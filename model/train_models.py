"""
ML Assignment 2 - Model Training Script
Dataset: Breast Cancer Wisconsin (Diagnostic) — sklearn built-in
30 features, 569 instances, binary classification
"""

import os
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# 1. Load dataset
print("Loading Breast Cancer Wisconsin dataset...")
bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['target'] = bc.target  # 0=Malignant, 1=Benign

print(f"Dataset shape: {df.shape}")
print(f"Features: {len(bc.feature_names)}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# 2. Split
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test split for Streamlit upload demo
test_df = X_test.copy()
test_df['target'] = y_test.values
test_df.to_csv('../test_data.csv', index=False)
print(f"\nTest data saved: ../test_data.csv ({len(test_df)} rows)")

# 3. Preprocessing
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 4. Models
models = {
    'logistic_regression': Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ]),
    'decision_tree': Pipeline([
        ('pre', preprocessor),
        ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    'knn': Pipeline([
        ('pre', preprocessor),
        ('clf', KNeighborsClassifier(n_neighbors=7))
    ]),
    'naive_bayes': Pipeline([
        ('pre', preprocessor),
        ('clf', GaussianNB())
    ]),
    'random_forest': Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))
    ]),
    'xgboost': Pipeline([
        ('pre', preprocessor),
        ('clf', XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42
        ))
    ]),
}

# 5. Train, evaluate, save
results = {}

for name, pipeline in models.items():
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'auc':       round(roc_auc_score(y_test, y_prob), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred), 4),
        'f1':        round(f1_score(y_test, y_pred), 4),
        'mcc':       round(matthews_corrcoef(y_test, y_pred), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['Malignant', 'Benign']
        )
    }
    results[name] = metrics

    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"  Accuracy:{metrics['accuracy']}  AUC:{metrics['auc']}  F1:{metrics['f1']}  MCC:{metrics['mcc']}")

with open('model_results.pkl', 'wb') as f:
    pickle.dump(results, f)

feature_names = list(X.columns)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

target_names = ['Malignant', 'Benign']
with open('target_names.pkl', 'wb') as f:
    pickle.dump(target_names, f)

# 6. Summary
print("\n\n" + "="*80)
print("RESULTS SUMMARY — Breast Cancer Wisconsin Dataset")
print("="*80)
print(f"{'Model':<28} {'Accuracy':>9} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'MCC':>8}")
print("-"*80)
for name, m in results.items():
    print(f"{name:<28} {m['accuracy']:>9} {m['auc']:>8} {m['precision']:>10} {m['recall']:>8} {m['f1']:>8} {m['mcc']:>8}")
print("="*80)
print("\nAll models and test data saved successfully!")
