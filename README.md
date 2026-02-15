# 🎗️ Breast Cancer Classification — ML Assignment 2

> **BITS Pilani WILP | M.Tech (AIML/DSE) | Machine Learning — Assignment 2**

---

## a. Problem Statement

Breast cancer is one of the most common cancers worldwide. Accurate and early classification of tumour cells as **Malignant** (cancerous) or **Benign** (non-cancerous) based on digitised cell nucleus measurements can directly save lives by enabling early intervention.

**Objective:** Build and evaluate six machine learning classification models on the Breast Cancer Wisconsin (Diagnostic) dataset, and deploy an interactive Streamlit web application for real-time prediction and model comparison.

- **Class 0:** Malignant (cancerous)
- **Class 1:** Benign (non-cancerous)

---

## b. Dataset Description

| Property          | Details                                                                 |
|-------------------|-------------------------------------------------------------------------|
| **Name**          | Breast Cancer Wisconsin (Diagnostic)                                    |
| **Source**        | UCI Machine Learning Repository / `sklearn.datasets.load_breast_cancer` |
| **URL**           | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| **Instances**     | 569                                                                     |
| **Features**      | 30 (10 real-valued features × 3 statistical summaries: mean, error, worst) |
| **Target**        | Binary — 0 (Malignant) / 1 (Benign)                                     |
| **Missing Values**| None                                                                    |
| **Class Balance** | 357 Benign (62.7%) · 212 Malignant (37.3%)                              |

### Feature Descriptions (30 features)

Ten cell nucleus properties are measured from digitised fine needle aspirate images:

| Core Feature        | Statistics Computed                          |
|---------------------|----------------------------------------------|
| radius              | mean, standard error, worst (largest mean)   |
| texture             | mean, standard error, worst                  |
| perimeter           | mean, standard error, worst                  |
| area                | mean, standard error, worst                  |
| smoothness          | mean, standard error, worst                  |
| compactness         | mean, standard error, worst                  |
| concavity           | mean, standard error, worst                  |
| concave points      | mean, standard error, worst                  |
| symmetry            | mean, standard error, worst                  |
| fractal dimension   | mean, standard error, worst                  |

**Preprocessing:**
- No missing values — no imputation required (SimpleImputer included as a safety step)
- All 30 features **standardized** using `StandardScaler` (zero mean, unit variance)
- Train/test split: **80% / 20%** (random_state=42, stratified)
  - Training set: 455 samples
  - Test set: 114 samples

---

## c. Models Used

### Evaluation Metrics — Comparison Table

*(Actual values from training run on 20% held-out test set)*

| ML Model Name              | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree              | 0.9211   | 0.9163 | 0.9565    | 0.9167 | 0.9362 | 0.8341 |
| kNN                        | 0.9737   | 0.9884 | 0.9600    | 1.0000 | 0.9796 | 0.9442 |
| Naïve Bayes (Gaussian)     | 0.9298   | 0.9868 | 0.9444    | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble)   | 0.9474   | 0.9940 | 0.9583    | 0.9583 | 0.9583 | 0.8869 |
| XGBoost (Ensemble)         | 0.9474   | 0.9931 | 0.9459    | 0.9722 | 0.9589 | 0.8864 |

---

### Observations on Model Performance

| ML Model Name              | Observation about model performance |
|----------------------------|--------------------------------------|
| **Logistic Regression**    | **Best overall performer** with Accuracy=0.9825, AUC=0.9954, and MCC=0.9623 — remarkable for a linear model. The Breast Cancer dataset has strong linear separability in standardised feature space. The high AUC confirms near-perfect discriminative ability. Logistic Regression is also highly interpretable, making it suitable for clinical deployment. High precision and recall (both 0.9861) indicate very balanced false positive and false negative rates. |
| **Decision Tree**          | Lowest performance among all models (Accuracy=0.9211, AUC=0.9163, MCC=0.8341). While still respectable, the decision tree suffers from higher variance — it memorises specific training patterns that do not generalise well. The lower AUC (0.9163 vs 0.99+ for ensemble models) confirms weaker probabilistic calibration. The `max_depth=5` constraint helps avoid severe overfitting, but the model still misclassifies more borderline cases. |
| **kNN**                    | Very strong results (Accuracy=0.9737, AUC=0.9884, Recall=1.0000). The perfect recall (1.0) means zero missed malignant cases on this test set — critical for medical screening. Feature scaling is essential for kNN; without standardisation, features with large magnitude (e.g., area, perimeter) would dominate distance calculations. k=7 balances bias and variance well for this dataset size. |
| **Naïve Bayes**            | Despite assuming feature independence (clearly violated: features like radius, perimeter, and area are highly correlated), Naïve Bayes achieves Accuracy=0.9298 and AUC=0.9868. The high AUC indicates good probabilistic ranking even if raw probability estimates may be poorly calibrated. Fastest model to train. Useful as a benchmark when computation is constrained. |
| **Random Forest (Ensemble)** | Accuracy=0.9474, AUC=0.9940. The bagging approach effectively reduces variance — 200 decision trees trained on random data subsets, aggregated by majority vote. The AUC of 0.994 (second only to Logistic Regression) reflects strong ranking ability. Slightly lower accuracy than LR and kNN on this specific test split, but more robust across different random splits. Feature importance analysis reveals `worst concave points`, `worst perimeter`, and `worst radius` as top predictors. |
| **XGBoost (Ensemble)**     | Accuracy=0.9474, AUC=0.9931. Sequential boosting corrects residual errors of prior trees, achieving high F1=0.9589 and MCC=0.8864. Performs very similarly to Random Forest on this dataset, with slightly higher recall (0.9722 vs 0.9583). The gradient boosting approach handles the slight class imbalance well. Generally the most powerful model for tabular data, though on this clean, well-separable dataset, simpler models (LR, kNN) slightly edge it out. |

---

## d. Repository Structure

```
BITZ_ML_ASSGMT2_2024DC04196/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── test_data.csv            # Test split (114 rows, generated by training script)
└── model/
    ├── train_models.py      # Training script — trains all 6 models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── model_results.pkl    # Pre-computed metrics dict
    ├── feature_names.pkl    # Feature column names
    └── target_names.pkl     # Class labels
```

---

## e. How to Run Locally

```bash
# 1. Clone repository
git clone https://github.com/SaluAnand/BITZ_ML_ASSGMT2_2024DC04196.git
cd BITZ_ML_ASSGMT2_2024DC04196

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (generates .pkl files and test_data.csv)
cd model
python train_models.py
cd ..

# 4. Launch app
streamlit run app.py
```

---

## f. Streamlit App Features

| Feature | Status |
|---------|--------|
| Dataset upload (CSV test data) | ✅ |
| Model selection dropdown | ✅ |
| Evaluation metrics display (all 6) | ✅ |
| Confusion matrix (heatmap) | ✅ |
| ROC Curve with AUC | ✅ |
| Classification report | ✅ |
| Model comparison table | ✅ |
| Bar chart comparison | ✅ |
| Radar chart (all models × all metrics) | ✅ |
| Single-instance prediction | ✅ |

---

## g. Deployment

Deployed on **Streamlit Community Cloud**:

🔗 **Live App:** `https://SaluAnand.mlassgmt2.streamlit.app`  
🔗 **GitHub Repo:** `https://github.com/SaluAnand/BITZ_ML_ASSGMT2_2024DC04196`

---

## h. Key Findings

1. **Logistic Regression is the best model** on this dataset (Accuracy 98.25%, AUC 99.54%), demonstrating that the Breast Cancer dataset is largely linearly separable after standardisation.
2. **kNN achieves perfect recall (1.0)** — zero missed malignant cases — which is clinically significant for cancer screening where false negatives are extremely costly.
3. **Ensemble methods (Random Forest, XGBoost)** have near-identical performance on this dataset and both achieve excellent AUC (~0.993–0.994), but are outperformed by LR and kNN on accuracy.
4. **Decision Tree** is the weakest performer due to limited generalisation capacity, despite `max_depth=5` regularisation.
5. **Naïve Bayes** outperforms expectations (AUC 0.9868) despite strong feature correlations in the dataset, demonstrating its robustness as a probabilistic classifier.

---


