Built a predictive model to classify churn behavior using Random Forest, Bagging, Boosting, SMOTE, and hyperparameter tuning.

Please refer to [code/notebook.ipynb](code/notebook.ipynb) to view the code.


# Thera Bank – Credit Card Churn Prediction

Predicting which credit card users are likely to leave, using EDA, feature engineering, and classification models with sampling and tuning.  
**Azin Faghihi | February 2025**  
AIML course – Bagging, Boosting, Hyperparameter Tuning

---

## Project Summary

Thera Bank is losing credit card customers.  
We use historical customer data (~10K rows) to build a predictive model for churn.  
The project includes preprocessing, class balancing, model selection, and hyperparameter tuning.

---

## Problem

Churn is costly.  
The goal is to identify high-risk customers early, so the bank can intervene and retain them.

---

## Approach

- **EDA**:
  - Analyzed categorical + numerical variables (univariate + bivariate)
  - Identified churn-related features like transaction count, credit limit, income
  - Found correlations between credit limit, open-to-buy, utilization, etc.

- **Preprocessing**:
  - Imputed missing values using most frequent value
  - Dropped collinear/outlier-heavy features (e.g., Credit_Limit, Months_on_book)
  - Created balanced training sets using oversampling and undersampling

- **Modeling**:
  - Tried multiple classifiers (AdaBoost, Gradient Boosting, etc.)
  - Tuned hyperparameters using validation recall
  - Selected final model with best recall and minimal overfitting

---

## Key Observations

- Attrition_Flag is imbalanced (~84% retained)
- Higher churn linked to:
  - Lower income and credit limit
  - Fewer transactions
  - Lower revolving balance
  - Lower utilization ratio
- Recall is prioritized to reduce false negatives (i.e., missing a churning customer)
- AdaBoost on oversampled data was selected as final model

---

## Data

- 10,127 rows × 24 columns  
- 3,380 missing values (Education, Income, Marital Status)  
- No duplicates  
- Numerical features: e.g., Age, Credit Limit, Transaction Count  
- Categorical features: e.g., Gender, Marital Status, Card Category

---

## Final Model

- **Model**: AdaBoost  
- **Class balance**: SMOTE oversampling  
- **Metric**: Recall on validation and test  
- **Feature importance** matched EDA insights

---

## Tools

- Python (Pandas, Scikit-learn, imbalanced-learn)  
- Matplotlib / Seaborn  
- Jupyter Notebook

---

## Note

This project was part of the AIML course by Great Learning.  
**Proprietary content – do not redistribute.**
