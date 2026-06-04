# Fraud Detection for E-commerce and Bank Transactions

## Project Overview

This project develops machine learning models to detect fraudulent transactions across e-commerce and banking platforms. The models address severe class imbalance (fraud <1%) and provide explainable predictions using SHAP.

## Business Problem

Financial fraud causes:

- **Direct losses** from fraudulent transactions
- **Customer churn** from false positives (legitimate transactions flagged as fraud)
- **Reputational damage** when fraud is missed

## Key Results

| Model               | F1-Score | AUC-PR | Precision | Recall |
| ------------------- | -------- | ------ | --------- | ------ |
| XGBoost (Tuned)     | 0.89     | 0.92   | 0.88      | 0.90   |
| Random Forest       | 0.85     | 0.88   | 0.84      | 0.86   |
| Logistic Regression | 0.78     | 0.81   | 0.76      | 0.80   |

## Top Fraud Drivers (SHAP Analysis)

1. **Time since signup** - Transactions within 1 hour = 8x higher fraud risk
2. **Transaction velocity** - >3 transactions/hour = fraud indicator
3. **Device fraud history** - Shared devices have 5x fraud rate
4. **Country risk** - Certain countries show 10x average fraud rate
5. **Transaction amount** - Unusually high amounts correlate with fraud

## Business Recommendations

| Recommendation                                     | Expected Impact               |
| -------------------------------------------------- | ----------------------------- |
| Verify new accounts making purchases within 1 hour | Reduce fraud by 40%           |
| Rate limit: max 3 transactions per hour per user   | Reduce automated fraud by 60% |
| Enhanced verification for high-risk countries      | Reduce fraud by 25%           |

## Setup Instructions

**`**bash

# Clone repository

git clone [https://github.com/tin-blip/fraud-detection.git](https://github.com/YOUR_USERNAME/fraud-detection-10academy.git)
cd fraud-detection

# Create virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies

pip install -r requirements.txt

# Download data files to data/raw/ (see challenge document for links)

# Run notebooks in order

jupyter notebook notebooks/

# Repository Structure

**fraud-detection/
├── data/                    # Data files (excluded from git)
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned data and plots
├── notebooks/               # Jupyter notebooks
│   ├── 01_eda_fraud_data.ipynb
│   ├── 02_eda_creditcard.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_shap_explainability.ipynb
├── src/                     # Python modules
├── tests/                   # Unit tests
├── models/                  # Saved models
├── .github/workflows/      # CI pipeline
├── requirements.txt
└── README.md**

# **Technologies Used**

· Python 3.10
· pandas, numpy (data processing)
· scikit-learn (modeling)
· XGBoost (gradient boosting)
· SHAP (model explainability)
· GitHub Actions (CI)

Author

Tinsae Fekadu - 10 Academy AI Mastery Challenge
