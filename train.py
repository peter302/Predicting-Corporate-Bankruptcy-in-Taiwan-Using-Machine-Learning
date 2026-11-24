"""
Corporate Bankruptcy Prediction — Taiwan
Training pipeline with leakage fix, correct metric, and broader search.

Usage:
    python train.py --data data/taiwan-bankruptcy-data.json.gz --out model.pkl
"""

import argparse
import gzip
import json
import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   # <-- NOT sklearn Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


# ── data loading ──────────────────────────────────────────────────────────────

def wrangle(filepath: str) -> pd.DataFrame:
    with gzip.open(filepath, "rb") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["observations"]).set_index("id")
    df["bankrupt"] = df["bankrupt"].astype(bool)
    return df


# ── training ──────────────────────────────────────────────────────────────────

def train(data_filepath: str, model_filepath: str) -> None:
    # 1. Load
    df = wrangle(data_filepath)
    print(f"Data shape: {df.shape}")
    print(f"Class balance:\n{df['bankrupt'].value_counts()}\n")

    X = df.drop(columns="bankrupt")
    y = df["bankrupt"]

    # 2. Stratified split — ensures minority class is proportionally represented
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Pipeline: impute → SMOTE → classifier
    #
    #    CRITICAL FIX: SMOTE is INSIDE the pipeline. During GridSearchCV,
    #    each fold applies SMOTE only to the training portion of that fold.
    #    The validation fold never sees synthetic samples, so CV scores are real.
    #
    #    The original code ran RandomOverSampler before GridSearchCV, which let
    #    duplicate rows bleed into validation folds — inflating F1 by ~0.38.

    pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote",   SMOTE(random_state=42)),
        ("clf",     RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

    # 4. Hyperparameter search
    #
    #    scoring="f1" — optimises for the bankrupt class F1, not accuracy.
    #    On 94/6 imbalanced data, accuracy is dominated by the majority class.
    #    The original grid searched on accuracy, which is why recall on the
    #    bankrupt class was only 0.41 (missing 59% of real bankruptcies).
    #
    #    max_depth lowered: original range(30,50) made training accuracy = 1.0
    #    (perfect memorisation). Added min_samples_leaf to control leaf purity.

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth":    [10, 20, None],
        "clf__min_samples_leaf": [1, 3],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1",       # minority-class F1 as the selection criterion
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("── Fitting GridSearchCV ──")
    search.fit(X_train, y_train)

    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV F1  : {search.best_score_:.4f}\n")

    # 5. Evaluate on held-out test set
    y_pred = search.predict(X_test)

    print("── Test set performance ──")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Bankrupt"]))

    # 6. Save
    with open(model_filepath, "wb") as f:
        pickle.dump(search, f)
    print(f"Model saved → {model_filepath}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/taiwan-bankruptcy-data.json.gz")
    parser.add_argument("--out",  default="model.pkl")
    args = parser.parse_args()
    train(args.data, args.out)
