#  Diabetes Prediction Using Machine Learning Algorithms and Ontology

> A comparative study of 6 ML classification algorithms combined with an ontology-based approach for early diabetes prediction.

**Author:** Tharini G (1NC22CI061)  
**Institution:** Nagarjuna College of Engineering and Technology, Bengaluru  
**Department:** Computer Science and Engineering (AI & ML)  
**Academic Year:** 2024–25  
**Guide:** Prof. Shivalila  

---

##  Overview

Diabetes mellitus is a chronic metabolic disorder affecting millions worldwide. Early detection is critical to prevent complications like cardiovascular disease, kidney failure, and neuropathy.

This project presents a **comprehensive comparative analysis** of machine learning algorithms and an **ontology-based classification** system for early diabetes prediction using the Pima Indians Diabetes Database (PIDD).

---

##  Objectives

- Compare 6 ML classification algorithms on the PIDD dataset
- Build an ontology-based classifier using Protégé + SWRL rules
- Evaluate performance using Accuracy, Precision, Recall, F-Measure
- Identify the most effective method for diabetes prediction

---

## Dataset

**Pima Indians Diabetes Database (PIDD)**
- Source: [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Instances: 768
- Features: 8 numerical attributes
- Class: Binary (Diabetic / Non-Diabetic)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |

---

##  Algorithms Used

| Algorithm | Tool |
|---|---|
| Support Vector Machine (SVM) | Weka / Scikit-learn |
| K-Nearest Neighbors (KNN) | Weka / Scikit-learn |
| Artificial Neural Network (ANN) | Weka / Scikit-learn |
| Naive Bayes (NB) | Weka / Scikit-learn |
| Logistic Regression (LR) | Weka / Scikit-learn |
| Decision Tree (DT) | Weka / Scikit-learn |
| Ontology-Based Classification | Protégé + SWRL + Pellet Reasoner |

---

##  Tools & Technologies

- **Python** — Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Weka** — ML algorithm implementation and evaluation
- **Protégé** — Ontology development
- **Cellfie Plugin** — Data import into ontology
- **SWRLTab Plugin** — Rule extraction from Decision Tree
- **Pellet Reasoner** — SWRL rule execution and inference

---

##  Results

### 10-Fold Cross-Validation

| Algorithm | Accuracy | Precision | Recall | F-Measure |
|---|---|---|---|---|
| SVM | **94%** | 0.93 | 0.94 | 0.93 |
| ANN | **94%** | 0.93 | 0.94 | 0.93 |
| KNN | 91% | 0.90 | 0.91 | 0.90 |
| Logistic Regression | 89% | 0.88 | 0.89 | 0.88 |
| Naive Bayes | 87% | 0.86 | 0.87 | 0.86 |
| Decision Tree | 85% | 0.84 | 0.85 | 0.84 |
| Ontology (SWRL) | 83% | 0.82 | 0.83 | 0.82 |

>  **Best performers: SVM and ANN with 94% accuracy**

---

##  Project Structure

```
diabetes-prediction/
│
├── diabetes_prediction.py       # Main Python script
├── diabetes_prediction.ipynb    # Jupyter Notebook (detailed walkthrough)
├── requirements.txt             # Python dependencies
├── diabetes.csv                 # Dataset (PIDD)
├── report/
│   └── Technical_Seminar_Report.pdf
└── README.md
```

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Tharini2004/diabetes-risk-classification-svm-ann.git
cd diabetes-risk-classification-svm-ann
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Python script
```bash
python diabetes_prediction.py
```

### 4. Or open the Jupyter Notebook
```bash
jupyter notebook diabetes_prediction.ipynb
```

---

##  Methodology

1. **Data Collection** — Pima Indians Diabetes Database (768 instances, 8 features)
2. **Data Preprocessing** — Handle missing values, normalize features, convert to ARFF
3. **ML Classification** — Apply 6 algorithms using Weka & Scikit-learn
4. **Ontology Development** — Build ontology in Protégé, import data via Cellfie, extract SWRL rules
5. **Evaluation** — Compare using Accuracy, Precision, Recall, F-Measure via 10-fold CV and 66% split

---

##  References

1. El Massari et al., "Diabetes Prediction Using ML Algorithms and Ontology," Journal of ICT Standardization, 2022
2. Khanam & Foo, "A comparison of ML algorithms for diabetes prediction," ICT Express, 2021
3. Khaleel & Al-Bakry, "Diagnosis of diabetes using ML algorithms," Materials Today, 2021

---

## 📄 License

This project is for academic purposes — Nagarjuna College of Engineering and Technology, 2024–25.
