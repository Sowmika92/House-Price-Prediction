#!/usr/bin/env python3
"""
Train a simple house price model and save the pipeline.

Usage:
  python src/train.py --data houses.csv --output models/house_price_model.joblib
"""
import argparse
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def load_data(path: Path):
    df = pd.read_csv(path)
    return df

def build_pipeline(numeric_features, categorical_features, random_state=42):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline

def main(args):
    data_path = Path(args.data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    # Basic validation
    required_cols = {"area", "bedrooms", "location", "age", "price"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"Dataset must contain columns: {required_cols}")

    X = df[["area", "bedrooms", "location", "age"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = ["area", "bedrooms", "age"]
    categorical_features = ["location"]

    pipeline = build_pipeline(numeric_features, categorical_features, random_state=42)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE: ${mae:,.2f}")
    print(f"Test RMSE: ${rmse:,.2f}")
    print(f"Test R2: {r2:.3f}")

    # Save pipeline
    joblib.dump(pipeline, out_path)
    print(f"Saved model pipeline to {out_path}")

    # Show a couple predictions for manual inspection
    print("\nSample predictions:")
    sample = X_test.reset_index(drop=True).head(5)
    sample_preds = pipeline.predict(sample)
    for i, row in sample.reset_index(drop=True).iterrows():
        print(f"  Input: {row.to_dict()} -> Predicted price: ${sample_preds[i]:,.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="houses.csv", help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default="models/house_price_model.joblib", help="Output path for saved model pipeline")
    args = parser.parse_args()
    main(args)
