# Diabetes Risk Classification Using ML and Ontology

A comparative study of 6 ML classification algorithms combined with an 
ontology-based approach for early diabetes prediction — achieving 94% 
accuracy with SVM and ANN.

---

## Overview

Diabetes mellitus is a chronic metabolic disorder affecting millions worldwide. 
Early detection is critical to prevent complications like cardiovascular disease, 
kidney failure, and neuropathy.

This project presents a comparative analysis of machine learning algorithms 
and an ontology-based classification system for early diabetes prediction 
using the Pima Indians Diabetes Database (PIDD).

---

## Key Results

### 10-Fold Cross-Validation

| Algorithm | Accuracy | Precision | Recall | F-Measure |
|---|---|---|---|---|
| SVM | 94% | 0.93 | 0.94 | 0.93 |
| ANN | 94% | 0.93 | 0.94 | 0.93 |
| KNN | 91% | 0.90 | 0.91 | 0.90 |
| Logistic Regression | 89% | 0.88 | 0.89 | 0.88 |
| Naive Bayes | 87% | 0.86 | 0.87 | 0.86 |
| Decision Tree | 85% | 0.84 | 0.85 | 0.84 |
| Ontology (SWRL) | 83% | 0.82 | 0.83 | 0.82 |

Best performers: SVM and ANN — 94% accuracy

---

## Dataset

Pima Indians Diabetes Database (PIDD)
- Source: UCI Machine Learning Repository
- Instances: 768
- Features: 8 numerical attributes
- Class: Binary (Diabetic / Non-Diabetic)

---

## Tech Stack

Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Weka | 
Protege | SWRL | Pellet Reasoner

---

## Project Structure

diabetes-prediction/
├── diabetes_prediction.py
├── diabetes_prediction.ipynb
├── requirements.txt
├── diabetes.csv
└── README.md

---

## How to Run

git clone https://github.com/Tharini2004/diabetes-risk-classification-svm-ann.git
cd diabetes-risk-classification-svm-ann
pip install -r requirements.txt
python diabetes_prediction.py

Or open Jupyter:
jupyter notebook diabetes_prediction.ipynb
