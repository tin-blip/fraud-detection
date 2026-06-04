# Fraud Detection for E-commerce and Bank Transactions

## Project Overview

This project develops machine learning models to detect fraudulent transactions across e-commerce platforms. The system handles highly imbalanced data (9.5% fraud rate) and provides explainable predictions using SHAP analysis.

## Business Impact

- **Reduce financial losses** by catching fraudulent transactions
- **Minimize false positives** to improve customer experience
- **Provide explainable decisions** for compliance and trust

## Key Results

| Model               | F1-Score | Precision | Recall | AUC-PR |
| ------------------- | -------- | --------- | ------ | ------ |
| XGBoost (Tuned)     | 0.89     | 0.88      | 0.90   | 0.92   |
| Random Forest       | 0.85     | 0.84      | 0.86   | 0.88   |
| Logistic Regression | 0.78     | 0.76      | 0.80   | 0.81   |

## Top Fraud Drivers (SHAP Analysis)

1. **Time since signup** - Transactions within 1 hour have 8x higher fraud risk
2. **Transaction velocity** - >3 transactions/hour indicates automated fraud
3. **Device fraud history** - Shared devices have 5x fraud rate
4. **Country risk** - Certain countries show 10x average fraud rate
5. **Transaction amount** - Unusually high amounts correlate with fraud

## Business Recommendations

| Recommendation                                     | Expected Impact               |
| -------------------------------------------------- | ----------------------------- |
| Verify new accounts making purchases within 1 hour | Reduce fraud by 40%           |
| Rate limit: max 3 transactions per hour per user   | Reduce automated fraud by 60% |
| Enhanced verification for high-risk countries      | Reduce fraud by 25%           |

## Repository Structure

**
fraud-detection/
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
│   ├── preprocess.py
│   ├── features.py
│   ├── models.py
│   └── visualize.py
├── tests/                   # Unit tests
├── scripts/                 # CLI scripts
├── models/                  # Saved models
├── .github/workflows/       # CI/CD pipeline
├── requirements.txt
└── README.md
**

## Setup Instructions

**# Clone repository
git clone https://github.com/tin-blip/fraud-detection.git
cd fraud-detection

# Create virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies

pip install -r requirements.txt

# Download data files to data/raw/

# (See challenge document for Google Drive links)

# Run notebooks in order

jupyter notebook notebooks/**

Technologies Used

· Python 3.10 - Core language
· pandas, numpy - Data processing
· scikit-learn - ML models
· XGBoost - Gradient boosting
· SHAP - Model explainability
· pytest - Unit testing
· GitHub Actions - CI/CD

Submission Information

· Challenge: 10 Academy - Artificial Intelligence Mastery
· Dates: Week 5 & 6 (June 4 - June 16, 2026)
· Author: Tinsae Fekadu

Branches

· main - Final submission
· task-1-eda-preprocessing - EDA and feature engineering
· task-2-modeling - Model training and evaluation
· task-3-shap - SHAP analysis and recommendations

License

This project is for educational purposes as part of the 10 Academy AI Mastery program.
