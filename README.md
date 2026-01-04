# Credit_Risk_Analysis_and_Default_prediction_using_Explainable_Machine_Learning
##  Project Overview
This project focuses on developing a robust predictive system to identify loan default risk using a comprehensive dataset of 1.3 Million Lending Club loans.

Unlike standard "accuracy-focused" projects, this system is engineered for Business Utility. It features an interactive Streamlit Dashboard that allows financial institutions to adjust risk thresholds dynamically and view real-time SHAP reason codes for every loan decision, ensuring transparency and regulatory compliance.


##  Key Technical Features
### 1. High-Scale Data Engineering
Feature Engineering: Transformed 151 raw features into 52 engineered predictors (e.g., loan_to_income, fico_cat_enc).

Memory Optimization: Handled 1.3M+ records using optimized data types to ensure high-performance training and inference.

Class Imbalance: Solved using scale_pos_weight and threshold moving to prioritize Recall for high-risk defaults.

### 2. Comparative Model Suite
I evaluated multiple architectures to find the optimal balance between speed and generalization:

Logistic Regression: Established a fast, interpretable baseline.

XGBoost: High predictive power with robust handling of minority classes.

LightGBM (Final Model): Chosen for its superior training speed and lowest Generalization Gap (0.0039).

### 3. Explainable AI (XAI)
Integrated SHAP (SHapley Additive exPlanations) to provide "Reason Codes" for decisions.

Top Risk Drivers: Identified int_rate, grade, term, and dti as the primary predictors of default.

Local Interpretability: Every prediction in the dashboard is explained by its specific SHAP contributors.

##  Interactive Dashboard (Streamlit)
The project includes a live decision support system with:

Dynamic Thresholding: Toggle between "Best-F1" (Profit Max) or "Recall ≈ 0.60" (Risk Averse).

Real-time Inference: Adjust borrower details and get an instant Approve/Reject decision.

Explainability UI: View horizontal bar charts showing exactly why a loan was flagged as risky.

##  Key Findings & Metrics
| Metric | Result | Impact |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.732 | Strong discriminative ability between good and bad loans. |
| **Optimized Recall** | 60.00% | Catches 6 out of 10 potential defaults. |
| **Overfitting Gap** | 0.0039 | Near-perfect generalization from training to test data. |
