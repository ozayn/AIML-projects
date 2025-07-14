Built an image classifier to distinguish plant seedlings and weeds using TensorFlow, image processing, and transfer learning.

Please refer to [code/notebook.ipynb](code/notebook.ipynb) to view the code.

# Plant Seedling Classification

Classifying plant seedlings into 12 categories using CNNs.  
Part of the Great Learning AIML course – Computer Vision track.  
**Azin Faghihi | March 2025**

---

## Project Summary

We built a deep learning pipeline to classify ~5K BRG images of plant seedlings.  
Two CNN models were trained and compared. The final model showed improved performance and less overfitting.

---

## Problem

Manual classification of plant seedlings is time-consuming and error-prone.  
We aim to automate this using image classification with CNNs.

---

## Approach

- **EDA**: Checked image samples and class distribution  
- **Preprocessing**:  
  - Resize: 128×128 → 64×64  
  - Convert BGR → RGB  
  - Normalize pixel values to [0, 1]  
  - One-hot encode labels  
  - Split: 80% train / 10% val / 10% test  
- **Modeling**:  
  - Built two CNNs with different architectures  
  - Used batch normalization, dropout, data augmentation  
  - Compared metrics: accuracy, precision, recall, f1-score  
  - Analyzed overfitting via loss curves  

---

## Results

- **Final model**: Model 2  
- Better train/val accuracy  
- Higher precision, recall, f1 for most classes  
- Lower overfitting  
- Visualized predictions and confusion matrix

---

## Key Observations

- Lowering learning rate helped  
- Fewer layers + batch norm + augmentation → better generalization  
- Hyperparameter tuning and regularization made a difference  
- Model performance varied across classes

---

## Data

- 4,750 images, 12 classes  
- Resized to 64×64  
- Labels one-hot encoded  
- Images originally in BGR format

---

## Tools

- Python, TensorFlow/Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn

---

## Next Steps

- Try transfer learning  
- Tune learning rate schedules  
- Handle underrepresented classes  
- Explore deployment options

---

## Note

This project was done as part of the AIML course by Great Learning.  
Content is proprietary – not for distribution.

---

