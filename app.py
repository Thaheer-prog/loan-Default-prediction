"""
Flask Web Application for Uncertainty-Aware Ensemble Learning for Loan Default Prediction.

Provides HTML pages (Home, Dashboard, Loan Application Form, Model Comparison,
Prediction History, Documentation) as well as JSON REST endpoints and CSV/HTML report downloads.
"""

import csv
import io
import json
import math
import os
import uuid
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from database.db import (
    delete_prediction,
    get_all_predictions,
    get_dashboard_statistics,
    get_prediction_by_id,
    init_db,
    insert_prediction,
)
from train_model import EVAL_DIR, MODELS_DIR, train_and_evaluate

app = Flask(__name__)

VALID_EMPLOYMENT = {"Salaried", "Self-Employed", "Business Owner", "Freelancer", "Unemployed"}
VALID_EDUCATION = {"Graduate", "Post-Graduate", "Undergraduate", "High School"}
VALID_MARITAL = {"Married", "Single", "Divorced"}
VALID_PROPERTY = {"Urban", "Semiurban", "Rural"}

_ARTIFACTS: Dict[str, Any] = {}


def load_artifacts() -> Dict[str, Any]:
    if _ARTIFACTS:
        return _ARTIFACTS

    pre_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    bundle_path = os.path.join(MODELS_DIR, "models_bundle.joblib")
    boot_path = os.path.join(MODELS_DIR, "bootstrap_ensemble.joblib")
    metrics_path = os.path.join(EVAL_DIR, "metrics.json")

    if not (os.path.exists(pre_path) and os.path.exists(bundle_path) and os.path.exists(boot_path)):
        train_and_evaluate()

    _ARTIFACTS["preprocessor"] = joblib.load(pre_path)
    _ARTIFACTS["models"] = joblib.load(bundle_path)
    _ARTIFACTS["bootstrap"] = joblib.load(boot_path)

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            _ARTIFACTS["metrics"] = json.load(f)
    else:
        _ARTIFACTS["metrics"] = {"dataset_summary": {}, "models": []}
    return _ARTIFACTS


def validate_and_parse_input(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []

    def parse_opt_float(key: str) -> Any:
        val = payload.get(key)
        if val is None or str(val).strip() == "":
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            errors.append(f"Invalid numeric value for {key}.")
            return None

    income = parse_opt_float("applicant_income")
    loan_amount = parse_opt_float("loan_amount")
    loan_term = parse_opt_float("loan_term")
    credit_history = parse_opt_float("credit_history")
    obligations = parse_opt_float("existing_obligations")

    if income is not None and income < 0:
        errors.append("Applicant income cannot be negative.")
    if loan_amount is not None and loan_amount <= 0:
        errors.append("Loan amount must be greater than zero.")
    if loan_term is not None and (loan_term < 6 or loan_term > 360):
        errors.append("Loan term must be between 6 and 360 months.")
    if credit_history is not None and credit_history not in (0.0, 1.0):
        errors.append("Credit history must be 0 or 1.")
    if obligations is not None and obligations < 0:
        errors.append("Existing obligations cannot be negative.")

    cleaned = {
        "applicant_name": str(payload.get("applicant_name") or "Applicant").strip()[:80] or "Applicant",
        "applicant_income": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_history": credit_history,
        "employment_status": payload.get("employment_status") if payload.get("employment_status") in VALID_EMPLOYMENT else "Salaried",
        "education": payload.get("education") if payload.get("education") in VALID_EDUCATION else "Graduate",
        "marital_status": payload.get("marital_status") if payload.get("marital_status") in VALID_MARITAL else "Married",
        "property_area": payload.get("property_area") if payload.get("property_area") in VALID_PROPERTY else "Urban",
        "existing_obligations": obligations,
    }
    return cleaned, errors


def run_uncertainty_aware_prediction(cleaned: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = load_artifacts()
    preprocessor = artifacts["preprocessor"]
    models = artifacts["models"]
    bootstrap_models = artifacts["bootstrap"]

    engineer = preprocessor.named_steps["engineer"]
    notes: List[str] = []

    # Track imputations & IQR outlier adjustments transparently
    inc_val = cleaned["applicant_income"]
    if inc_val is None:
        inc_val = engineer.medians_["applicant_income"]
        notes.append(f"Missing income imputed with training median (${inc_val:,.0f}).")
    low_inc, high_inc = engineer.iqr_bounds_["applicant_income"]
    ood_count = 0
    if inc_val < low_inc or inc_val > high_inc:
        ood_count += 1
        notes.append(f"Income (${inc_val:,.0f}) capped via IQR bounds.")

    loan_val = cleaned["loan_amount"]
    if loan_val is None:
        loan_val = engineer.medians_["loan_amount"]
        notes.append(f"Missing loan amount imputed with median (${loan_val:,.0f}).")
    low_loan, high_loan = engineer.iqr_bounds_["loan_amount"]
    if loan_val < low_loan or loan_val > high_loan:
        ood_count += 1
        notes.append(f"Loan amount (${loan_val:,.0f}) capped via IQR bounds.")

    term_val = cleaned["loan_term"] if cleaned["loan_term"] is not None else engineer.medians_["loan_term"]
    ch_val = int(cleaned["credit_history"]) if cleaned["credit_history"] is not None else engineer.credit_history_mode_
    obl_val = cleaned["existing_obligations"] if cleaned["existing_obligations"] is not None else engineer.medians_["existing_obligations"]

    df_row = pd.DataFrame([{
        "applicant_income": cleaned["applicant_income"],
        "loan_amount": cleaned["loan_amount"],
        "loan_term": cleaned["loan_term"],
        "credit_history": cleaned["credit_history"],
        "employment_status": cleaned["employment_status"],
        "education": cleaned["education"],
        "marital_status": cleaned["marital_status"],
        "property_area": cleaned["property_area"],
        "existing_obligations": cleaned["existing_obligations"],
    }])

    X_proc = preprocessor.transform(df_row)

    p_lr = float(models["logistic_regression"].predict_proba(X_proc)[0, 1])
    p_dt = float(models["decision_tree"].predict_proba(X_proc)[0, 1])
    p_rf = float(models["random_forest"].predict_proba(X_proc)[0, 1])
    p_gb = float(models["gradient_boosting"].predict_proba(X_proc)[0, 1])
    p_xgb = float(models["xgboost"].predict_proba(X_proc)[0, 1])
    p_ens = float(models["soft_voting_ensemble"].predict_proba(X_proc)[0, 1])

    boot_probs = np.array([float(m.predict_proba(X_proc)[0, 1]) for m in bootstrap_models])
    boot_std = float(np.std(boot_probs))
    ci_low = float(np.quantile(boot_probs, 0.05))
    ci_high = float(np.quantile(boot_probs, 0.95))

    base_probs = [p_lr, p_dt, p_rf, p_gb, p_xgb]
    model_disagreement = float(np.std(base_probs))

    p_clip = min(max(p_ens, 1e-6), 1.0 - 1e-6)
    entropy = -(p_clip * math.log2(p_clip) + (1.0 - p_clip) * math.log2(1.0 - p_clip))

    norm_boot = min(1.0, boot_std / 0.14)
    norm_disagree = min(1.0, model_disagreement / 0.18)
    ood_penalty = min(0.22, ood_count * 0.11)
    uncertainty_score = min(0.98, max(0.03, 0.42 * norm_boot + 0.33 * norm_disagree + 0.25 * entropy + ood_penalty))
    confidence_score = min(0.97, max(0.02, 1.0 - uncertainty_score))

    crosses_boundary = ci_low < 0.5 <= ci_high
    is_uncertain = uncertainty_score >= 0.42 or crosses_boundary or model_disagreement >= 0.12

    monthly_income = max(inc_val / 12.0, 1.0)
    monthly_installment = loan_val / max(term_val, 1.0)
    dti_ratio = (obl_val + monthly_installment) / monthly_income
    lti_ratio = loan_val / max(inc_val, 1.0)

    reasons: List[str] = []
    if model_disagreement >= 0.09:
        reasons.append(
            f"Inter-model disagreement ({model_disagreement*100:.1f}% std dev): models range from "
            f"{min(base_probs)*100:.1f}% to {max(base_probs)*100:.1f}%."
        )
    if crosses_boundary:
        reasons.append(
            f"90% bootstrap interval [{ci_low*100:.1f}%, {ci_high*100:.1f}%] spans the 50% decision threshold."
        )
    if 0.35 <= p_ens <= 0.65:
        reasons.append(
            f"Borderline default probability ({p_ens*100:.1f}%) increases predictive Shannon entropy ({entropy*100:.1f}%)."
        )
    if ch_val == 0 and inc_val >= 75000:
        reasons.append("Conflicting signals: strong annual income paired with prior credit delinquency.")
    if ood_count > 0:
        reasons.append(f"Out-of-distribution input ({ood_count} feature(s) capped by IQR bounds).")
    if not reasons:
        reasons.append(
            f"Strong consensus across base models (disagreement {model_disagreement*100:.1f}%) "
            f"with tight bootstrap bounds [{ci_low*100:.1f}%, {ci_high*100:.1f}%]."
        )

    if p_ens >= 0.60:
        risk_category = "High Risk"
    elif p_ens >= 0.30 or is_uncertain:
        risk_category = "Medium Risk"
    else:
        risk_category = "Low Risk"

    return {
        "application_ref": f"APP-{uuid.uuid4().hex[:8].upper()}",
        "applicant_name": cleaned["applicant_name"],
        "applicant_income": inc_val,
        "loan_amount": loan_val,
        "loan_term": int(term_val),
        "credit_history": ch_val,
        "employment_status": cleaned["employment_status"],
        "education": cleaned["education"],
        "marital_status": cleaned["marital_status"],
        "property_area": cleaned["property_area"],
        "existing_obligations": obl_val,
        "dti_ratio": round(dti_ratio, 4),
        "lti_ratio": round(lti_ratio, 4),
        "prediction": "Likely to Default" if p_ens >= 0.5 else "Likely to Repay",
        "default_probability": round(p_ens, 4),
        "confidence_score": round(confidence_score, 4),
        "uncertainty_score": round(uncertainty_score, 4),
        "bootstrap_std": round(boot_std, 4),
        "model_disagreement": round(model_disagreement, 4),
        "entropy_score": round(entropy, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "is_uncertain": is_uncertain,
        "uncertainty_reasons": reasons,
        "model_breakdown": {
            "logistic_regression": round(p_lr, 4),
            "decision_tree": round(p_dt, 4),
            "random_forest": round(p_rf, 4),
            "gradient_boosting": round(p_gb, 4),
            "xgboost": round(p_xgb, 4),
            "soft_voting_ensemble": round(p_ens, 4),
        },
        "risk_category": risk_category,
        "preprocessing_notes": notes,
    }


@app.route("/")
def home():
    metrics = load_artifacts()["metrics"]
    return render_template("index.html", metrics=metrics)


@app.route("/dashboard")
def dashboard():
    stats = get_dashboard_statistics()
    history = get_all_predictions()
    return render_template("dashboard.html", stats=stats, recent=history[:10])


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    result = None
    error = None
    if request.method == "POST":
        cleaned, errors = validate_and_parse_input(request.form.to_dict())
        if errors:
            error = " ".join(errors)
        else:
            result = run_uncertainty_aware_prediction(cleaned)
            record_id = insert_prediction(result)
            result["id"] = record_id
    return render_template("predict.html", result=result, error=error)


@app.route("/models")
def models_page():
    metrics = load_artifacts()["metrics"]
    return render_template("models.html", metrics=metrics)


@app.route("/history")
def history_page():
    records = get_all_predictions()
    return render_template("history.html", records=records)


@app.route("/documentation")
def documentation_page():
    metrics = load_artifacts()["metrics"]
    return render_template("documentation.html", metrics=metrics)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    cleaned, errors = validate_and_parse_input(payload)
    if errors:
        return jsonify({"error": " ".join(errors)}), 400
    result = run_uncertainty_aware_prediction(cleaned)
    if payload.get("save", True):
        result["id"] = insert_prediction(result)
    return jsonify(result), 200


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify({"records": get_all_predictions()}), 200


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def api_delete_history(record_id: int):
    if not delete_prediction(record_id):
        return jsonify({"error": "Record not found"}), 404
    return jsonify({"deleted": True, "id": record_id}), 200


@app.route("/reports/download/<int:record_id>")
def download_single_report(record_id: int):
    record = get_prediction_by_id(record_id)
    if not record:
        return Response("Record not found", status=404)
    html = render_template("report.html", record=record)
    return Response(
        html,
        mimetype="text/html",
        headers={"Content-Disposition": f'attachment; filename="{record["application_ref"]}_report.html"'},
    )


@app.route("/reports/export-csv")
def export_csv_report():
    records = get_all_predictions()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Application Ref", "Applicant Name", "Income", "Loan Amount", "Term",
        "Credit History", "Employment", "DTI Ratio", "Prediction",
        "Default Probability", "Confidence Score", "Uncertainty Score",
        "Uncertain Flag", "Risk Category", "Created At"
    ])
    for r in records:
        writer.writerow([
            r["application_ref"], r["applicant_name"], r["applicant_income"],
            r["loan_amount"], r["loan_term"], r["credit_history"],
            r["employment_status"], f"{r['dti_ratio']*100:.2f}%", r["prediction"],
            f"{r['default_probability']*100:.2f}%", f"{r['confidence_score']*100:.2f}%",
            f"{r['uncertainty_score']*100:.2f}%", "YES" if r["is_uncertain"] else "NO",
            r["risk_category"], r["created_at"],
        ])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="loan_predictions_report.csv"'},
    )


if __name__ == "__main__":
    init_db()
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
