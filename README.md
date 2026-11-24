# Uncertainty-Aware Ensemble Learning for Loan Default Prediction

An end-to-end Artificial Intelligence and Machine Learning B.Tech engineering project that predicts loan default probability **and** explicitly quantifies predictive uncertainty using **Soft Voting Ensembles**, **Bootstrap Resampling**, **Inter-Model Disagreement**, and **Predictive Shannon Entropy**.

> **Educational Prototype Disclaimer:** This system is an educational B.Tech engineering prototype built on a clearly labelled synthetic demonstration dataset (`dataset/loan_data.csv`). It is **not** a validated real-world lending or underwriting decision system.

---

## Key Features

- **6 Machine Learning Algorithms**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and Soft Voting Ensemble.
- **Explicit Separation of Probability vs. Uncertainty**:
  - **Default Probability ($P(\text{Default})$)**: Calibrated ensemble likelihood of borrower default.
  - **Uncertainty Score ($U$) & Prediction Confidence ($1 - U$)**: Computed from 14–25 bootstrap resampled estimators, standard deviation across model families, predictive entropy, and out-of-distribution IQR bounds.
  - **Diagnostic Explanations**: Explains *why* a prediction is flagged uncertain (model disagreement, CI crossing the 50% threshold, conflicting profile features, or outlier capping).
- **Automated Data Preprocessing**: Missing value median/mode imputation, IQR outlier winsorization, engineered Debt-to-Income (DTI) and Loan-to-Income (LTI) ratios, one-hot encoding, and standard scaling.
- **SQLite Persistence & Reports**: Stores applicant inputs, probabilities, confidence intervals, and uncertainty rationales; exports individual HTML reports and full CSV logs.

---

## Project Folder Structure

```
Loan-Default-Prediction/
├── app.py                     # Flask application & REST APIs
├── train_model.py             # Scikit-learn + XGBoost training, Bootstrap & Matplotlib plots
├── requirements.txt           # Python package dependencies
├── README.md                  # Project setup and usage guide
├── dataset/
│   ├── generate_dataset.py    # Deterministic 1,200-record demonstration generator
│   └── loan_data.csv          # Synthetic dataset with missing values & outliers
├── models/                    # Saved preprocessor, model bundle, and bootstrap ensemble
├── evaluation/                # Metrics JSON, ROC curve & confusion matrix PNGs
├── database/
│   ├── schema.sql             # SQLite DDL
│   ├── db.py                  # SQLite CRUD operations
│   └── loan_default.db        # Persistent SQLite database
├── templates/                 # Bootstrap 5 Jinja2 HTML views
├── static/
│   ├── css/style.css          # Banking analytics blue/white theme
│   └── js/app.js              # Interactive table filtering & scenario presets
├── docs/
│   └── PROJECT_DOCUMENTATION.md
└── tests/
    └── test_app.py            # Automated unit & integration tests
```

---

## Installation & Execution (Python / Flask)

```bash
cd Loan-Default-Prediction

# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate synthetic dataset
python dataset/generate_dataset.py

# 4. Train all 6 ML models, 25 bootstrap estimators & export evaluation charts
python train_model.py

# 5. Run unit & integration tests
python -m unittest discover -s tests -v

# 6. Launch Flask server
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Verified Benchmark Results (80/20 Stratified Split, 241 Test Records)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 78.84% | 66.67% | 45.45% | 54.05% | 77.35% |
| Decision Tree | 80.50% | 67.92% | 54.55% | 60.50% | 79.19% |
| Random Forest | 80.50% | 73.17% | 45.45% | 56.07% | 80.71% |
| Gradient Boosting | 78.42% | 64.58% | 46.97% | 54.39% | 76.56% |
| XGBoost | 79.67% | 68.09% | 48.48% | 56.64% | 76.48% |
| **Soft Voting Ensemble** | **80.50%** | **70.21%** | **50.00%** | **58.41%** | **78.25%** |
