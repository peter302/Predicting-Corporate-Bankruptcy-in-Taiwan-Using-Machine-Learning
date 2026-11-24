### Corporate Bankruptcy Prediction (Taiwan)

### Overview
This project applies machine learning to predict corporate bankruptcy using financial indicators from Taiwanese companies. The goal is to support financial risk assessment and early-warning analysis through data-driven classification models.

Completed as part of the Applied Data Science Lab – WorldQuant University.

### Problem
Can financial ratio data be used to accurately identify companies at risk of bankruptcy?

### Data
- **Source**: Taiwan Economic Journal (via WorldQuant University)
- **Records**: 6,137 companies
- **Features**: 95 financial indicators
- **Target**: Bankruptcy status (True / False)
- **Challenge**: Highly imbalanced classes (~94% healthy, ~6% bankrupt)

### Approach
1. Data loading and validation
2. Stratified train–test split (80/20)
3. Pipeline: median imputation → SMOTE → Random Forest
4. GridSearchCV optimising **F1 on the minority class** (not accuracy)
5. Evaluation via classification report (precision, recall, F1 per class)

### Key design decisions

**Oversampling inside the pipeline** — the original notebook applied `RandomOverSampler` before cross-validation, which caused duplicate rows to appear in CV validation folds and artificially inflated reported scores. SMOTE is now wrapped inside `imblearn.Pipeline` so it is applied only to training folds during CV.

**F1 as the search metric** — the original code used `GridSearchCV` with default accuracy scoring. On a 94/6 split, a model that predicts all-False scores 94% accuracy while missing every bankruptcy. F1 on the minority class is used instead, which penalises the model for missing real bankruptcies.

**SMOTE over RandomOverSampler** — SMOTE generates synthetic interpolated minority samples rather than duplicating existing ones, which generalises better to unseen data.

### Results
Evaluated on the held-out test set (20%), using F1 on the bankrupt class as the primary metric.

| Class   | Precision | Recall | F1   |
|---------|-----------|--------|------|
| Healthy | 0.99      | 0.99   | 0.99 |
| Bankrupt| 0.74      | 0.68   | 0.71 |

The bankrupt-class recall improved from **0.41 → ~0.68** after fixing oversampling leakage and switching the search metric to F1.

### Tools
- Python
- pandas, scikit-learn, imbalanced-learn
- Random Forest Classifier with SMOTE oversampling

### Usage
```bash
# Train
python train.py --data data/taiwan-bankruptcy-data.json.gz --out model.pkl

# Predict
python predict.py --data data/taiwan-bankruptcy-data-test-features.json.gz --model model.pkl
```

### Author
Peter Kuria
