Analyzed customer attributes and built a decision tree model to predict loan acquisition likelihood and guide marketing strategies.

Please refer to [code/notebook.ipynb](code/notebook.ipynb) to view the code.

# AllLife Bank – Personal Loan Campaign

Predicting which liability customers are likely to purchase a personal loan using decision tree models.  
**Azin Faghihi | January 2025**  
AIML course – Decision Tree Modeling

---

## Project Summary

We analyze data from a past personal loan campaign to identify customer traits linked to loan acceptance.  
The modeling uses decision trees with pre- and post-pruning.  
Due to class imbalance, we optimize the model for F1 score.

---

## Problem

AllLife Bank wants to grow its asset customers (loan holders) without losing liability customers (depositors).  
The goal is to predict which customers are likely to accept a personal loan offer, and prioritize those segments.

---

## Approach

- **EDA**:
  - Explored relationships between customer attributes and loan acceptance
  - Found key predictors: income, education, family size, credit card spending, CD account
  - Income, mortgage, and credit card spend are correlated

- **Preprocessing**:
  - No missing or duplicate values
  - Converted negative values in experience to absolute
  - Dropped ID and Experience columns (highly correlated with Age)
  - Extracted SCF code from ZIP
  - No outlier treatment applied during modeling

- **Modeling**:
  - Used decision trees with `class_weight="balanced"` to handle class imbalance (~10% loan acceptance)
  - Conducted hyperparameter tuning (max depth, min samples, etc.)
  - Applied both pre-pruning and cost-complexity post-pruning
  - Final model optimized for F1, using Gini criterion

---

## Key Observations

- Most customers:
  - Don’t have CD or securities accounts
  - Don’t own other credit cards
  - Have graduate or professional degrees
  - Are not living alone
- Customers accepting loans tend to:
  - Have higher income, mortgage, and credit card spend
  - Come from larger families
  - Have CD accounts
- Age and experience are linearly correlated → only Age used in model

---

## Data

- 5,000 rows × 14 columns  
- No missing or duplicate values  
- Categorical: Education, CD Account, Online, ZIP, etc.  
- Numerical: Age, Income, Mortgage, Credit Card Spend, etc.  
- Target: `Personal_Loan` (binary)

---

## Final Model

- **Model**: Decision Tree (Gini, max depth = 6)  
- **Optimized for**: F1 Score  
- **Performance**:
  - F1 on test: 0.90  
  - Balanced precision and recall  
  - Important features matched EDA insights

---

## Tools

- Python (Pandas, Scikit-learn)  
- Matplotlib / Seaborn  
- Jupyter Notebook

---

## Note

This project was created as part of the Great Learning AIML course.  
**Proprietary content – not for redistribution.**
