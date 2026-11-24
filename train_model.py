"""
Training & Evaluation Pipeline for Uncertainty-Aware Ensemble Learning for Loan Default Prediction.

Implements:
- Data preprocessing (missing value imputation, IQR outlier capping, feature engineering,
  One-Hot Encoding, StandardScaler) via scikit-learn ColumnTransformer.
- 6 Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting,
  XGBoost, and Soft Voting Ensemble.
- Bootstrap Ensemble (25 resampled XGBoost/GBDT estimators) for epistemic uncertainty quantification.
- Evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and ROC Curve.
- Visualization export using Matplotlib and Seaborn to evaluation/.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

NUMERIC_BASE_COLS = [
    "applicant_income",
    "loan_amount",
    "loan_term",
    "existing_obligations",
]
CATEGORICAL_COLS = [
    "employment_status",
    "education",
    "marital_status",
    "property_area",
]


class CreditFeatureEngineerAndOutlierCapper(BaseEstimator, TransformerMixin):
    """Imputes numeric/binary fields, caps outliers via training IQR bounds, and engineers DTI/LTI."""

    def __init__(self) -> None:
        self.iqr_bounds_: Dict[str, Tuple[float, float]] = {}
        self.medians_: Dict[str, float] = {}
        self.credit_history_mode_: int = 1

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CreditFeatureEngineerAndOutlierCapper":
        df = X.copy()
        for col in ["applicant_income", "loan_amount", "loan_term", "existing_obligations"]:
            series = pd.to_numeric(df[col], errors="coerce")
            med = float(series.median())
            self.medians_[col] = med
            filled = series.fillna(med)
            q1 = float(filled.quantile(0.25))
            q3 = float(filled.quantile(0.75))
            iqr = q3 - q1
            self.iqr_bounds_[col] = (max(0.0, q1 - 1.5 * iqr), q3 + 2.2 * iqr)

        ch = pd.to_numeric(df["credit_history"], errors="coerce").dropna()
        self.credit_history_mode_ = int(ch.mode().iloc[0]) if len(ch) > 0 else 1
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in ["applicant_income", "loan_amount", "loan_term", "existing_obligations"]:
            series = pd.to_numeric(df[col], errors="coerce").fillna(self.medians_[col])
            low, high = self.iqr_bounds_[col]
            if col != "loan_term":
                series = series.clip(lower=low, upper=high)
            df[col] = series

        df["credit_history"] = pd.to_numeric(df["credit_history"], errors="coerce").fillna(
            self.credit_history_mode_
        )
        monthly_income = np.maximum(df["applicant_income"] / 12.0, 1.0)
        monthly_installment = df["loan_amount"] / np.maximum(df["loan_term"], 1.0)
        df["monthly_installment"] = monthly_installment
        df["dti_ratio"] = (df["existing_obligations"] + monthly_installment) / monthly_income
        df["lti_ratio"] = df["loan_amount"] / np.maximum(df["applicant_income"], 1.0)
        return df


def build_preprocessor() -> Pipeline:
    numeric_features = [
        "applicant_income",
        "loan_amount",
        "loan_term",
        "credit_history",
        "existing_obligations",
        "monthly_installment",
        "dti_ratio",
        "lti_ratio",
    ]
    col_tf = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )
    return Pipeline([("engineer", CreditFeatureEngineerAndOutlierCapper()), ("columns", col_tf)])


def train_and_evaluate() -> Dict[str, Any]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    feature_cols = [
        "applicant_income",
        "loan_amount",
        "loan_term",
        "credit_history",
        "employment_status",
        "education",
        "marital_status",
        "property_area",
        "existing_obligations",
    ]
    X = df[feature_cols]
    y = df["loan_status"].astype(int).values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 1. Base & Ensemble Models
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=42)
    rf = RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=4, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=140, learning_rate=0.08, max_depth=3, random_state=42)
    xgb = XGBClassifier(
        n_estimators=160,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb), ("xgb", xgb)],
        voting="soft",
        weights=[0.12, 0.23, 0.30, 0.35],
    )

    named_models = [
        ("Logistic Regression", "logistic_regression", lr),
        ("Decision Tree", "decision_tree", dt),
        ("Random Forest", "random_forest", rf),
        ("Gradient Boosting", "gradient_boosting", gb),
        ("XGBoost", "xgboost", xgb),
        ("Soft Voting Ensemble", "soft_voting_ensemble", ensemble),
    ]

    metrics_list: List[Dict[str, Any]] = []
    roc_plot_data = []

    for display_name, key, model in named_models:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = float(roc_auc_score(y_test, y_prob))

        roc_plot_data.append((display_name, fpr, tpr, roc_auc))
        metrics_list.append(
            {
                "model": display_name,
                "key": key,
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "roc_auc": round(roc_auc, 4),
                "confusion_matrix": cm.tolist(),
                "roc_curve": [
                    {"fpr": round(float(fx), 4), "tpr": round(float(tx), 4)}
                    for fx, tx in zip(fpr[::max(1, len(fpr)//30)], tpr[::max(1, len(tpr)//30)])
                ],
            }
        )

    # 2. Bootstrap Ensemble (25 resampled models) for Epistemic Uncertainty
    rng = np.random.RandomState(42)
    bootstrap_models: List[XGBClassifier] = []
    n_train = X_train.shape[0]
    for b in range(25):
        idx = rng.choice(n_train, size=n_train, replace=True)
        boot_clf = XGBClassifier(
            n_estimators=90,
            learning_rate=0.10,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=100 + b,
        )
        boot_clf.fit(X_train[idx], y_train[idx])
        bootstrap_models.append(boot_clf)

    # Save trained artifacts
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    joblib.dump(
        {
            "logistic_regression": lr,
            "decision_tree": dt,
            "random_forest": rf,
            "gradient_boosting": gb,
            "xgboost": xgb,
            "soft_voting_ensemble": ensemble,
        },
        os.path.join(MODELS_DIR, "models_bundle.joblib"),
    )
    joblib.dump(bootstrap_models, os.path.join(MODELS_DIR, "bootstrap_ensemble.joblib"))

    # Generate ROC Curve Matplotlib Plot
    plt.figure(figsize=(8, 6))
    for name, fpr, tpr, auc_val in roc_plot_data:
        lw = 2.6 if "Ensemble" in name else 1.6
        plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — Loan Default Prediction")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "roc_comparison.png"), dpi=150)
    plt.close()

    # Generate Confusion Matrix Heatmap for Soft Voting Ensemble
    ens_cm = np.array(metrics_list[-1]["confusion_matrix"])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        ens_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Repaid (0)", "Default (1)"],
        yticklabels=["Repaid (0)", "Default (1)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Soft Voting Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "ensemble_confusion_matrix.png"), dpi=150)
    plt.close()

    summary = {
        "dataset_summary": {
            "total_records": int(len(df)),
            "train_records": int(len(X_train_raw)),
            "test_records": int(len(X_test_raw)),
            "default_count": int(df["loan_status"].sum()),
            "repaid_count": int((1 - df["loan_status"]).sum()),
            "default_rate": round(float(df["loan_status"].mean()), 4),
        },
        "models": metrics_list,
    }
    with open(os.path.join(EVAL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete. Evaluation metrics saved to evaluation/metrics.json")
    return summary


if __name__ == "__main__":
    train_and_evaluate()
