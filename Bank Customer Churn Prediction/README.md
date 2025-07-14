Developed an artificial neural network from scratch to identify high-risk churn customers using TensorFlow, and Keras.

Please refer to [code/notebook.ipynb](code/notebook.ipynb) to view the code.

# Bank Churn Prediction – Neural Networks

Classifying bank customers who are likely to leave using a neural network model.  
**Azin Faghihi | March 2025**  
AIML course – Neural Networks

---

## Project Summary

We built a neural network classifier to predict customer churn using a dataset of ~10K bank customers.  
The target class (`Exited`) is imbalanced, so we use sampling strategies and recall as the evaluation metric.  
Multiple NN architectures are tested; the final model achieves high recall with minimal overfitting.

---

## Problem

Customer churn leads to revenue loss.  
The goal is to identify potential churners so the bank can retain them through timely intervention.

---

## Approach

- **EDA**:  
  - Univariate + bivariate analysis of categorical and numerical variables  
  - Churn is linked to age, balance, number of products, and activity level  
  - No missing or duplicate values  

- **Preprocessing**:  
  - Dropped ID and name fields  
  - Created dummy variables  
  - Normalized numerical values using `StandardScaler`  
  - Optional scaling of dollar amounts for visual clarity  
  - Used oversampling and undersampling to balance classes

- **Modeling**:  
  - Tried multiple NN architectures with:
    - Dense layers ([64, 32, 16, 8]), 3 hidden layers
    - Dropout layers
    - SGD and Adam optimizers (tuned learning rate + momentum)
    - `ReLU` and `tanh` activations
  - Evaluated models based on train/validation recall gap
  - Trained on sampled data

---

## Key Observations

- About 80% of customers stay, making this an imbalanced classification problem  
- Customers with higher balance, older age, and more products tend to churn more  
- Activity level, geography, and gender also correlate with churn  
- Final model uses SGD (lr = 1e-3, momentum = 0.95), trained for ~1 hour  
- Test recall: ~0.76  
- Chosen model shows minimal overfitting with smooth loss/recall curves

---

## Data

- 10,000 rows × 13 columns  
- No missing or duplicate entries  
- Categorical features: Gender, Geography, IsActiveMember, etc.  
- Numerical features: Age, Balance, Salary, Credit Score, etc.  
- Target: `Exited` (binary)

---

## Tools

- Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras)  
- Jupyter Notebook  
- Matplotlib / Seaborn

---

## Note

This project was part of the Great Learning AIML course.  
**Proprietary content – not for distribution.**
