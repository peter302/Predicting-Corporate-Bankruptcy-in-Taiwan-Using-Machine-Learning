"""
Corporate Bankruptcy Prediction — Taiwan
Inference module. Replaces the original my_predictor.py.

Usage (standalone):
    python predict.py --data data/taiwan-bankruptcy-data-test-features.json.gz \
                      --model model.pkl

Usage (as a module):
    from predict import make_predictions
    preds = make_predictions("data/test.json.gz", "model.pkl")
"""

import argparse
import gzip
import json
import pickle

import pandas as pd


def _load_features(filepath: str) -> pd.DataFrame:
    with gzip.open(filepath, "rb") as f:
        raw = json.load(f)
    return pd.DataFrame(raw["observations"]).set_index("id")


def make_predictions(data_filepath: str, model_filepath: str) -> pd.Series:
    df = _load_features(data_filepath)

    with open(model_filepath, "rb") as f:
        model = pickle.load(f)

    preds = model.predict(df)
    return pd.Series(preds, index=df.index, name="bankrupt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="data/taiwan-bankruptcy-data-test-features.json.gz")
    parser.add_argument("--model", default="model.pkl")
    args = parser.parse_args()

    predictions = make_predictions(args.data, args.model)
    print(f"Predictions shape: {predictions.shape}")
    print(predictions.head(10))
    print(f"\nBankrupt predicted: {predictions.sum()} / {len(predictions)}")
