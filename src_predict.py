#!/usr/bin/env python3
"""
Load the saved pipeline and predict a house price.

Usage examples:
  python src/predict.py --model models/house_price_model.joblib --area 1500 --bedrooms 3 --location B --age 10
  python src/predict.py --model models/house_price_model.joblib --example
"""
import argparse
import joblib
import numpy as np
import pandas as pd

def predict_from_args(pipeline, area, bedrooms, location, age):
    X = pd.DataFrame([{
        "area": float(area),
        "bedrooms": int(bedrooms),
        "location": location,
        "age": float(age),
    }])
    pred = pipeline.predict(X)[0]
    return pred

def main(args):
    pipeline = joblib.load(args.model)
    if args.example:
        examples = [
            {"area": 1500, "bedrooms": 3, "location": "B", "age": 10},
            {"area": 900, "bedrooms": 2, "location": "A", "age": 20},
            {"area": 2100, "bedrooms": 5, "location": "C", "age": 2},
        ]
        for ex in examples:
            p = predict_from_args(pipeline, **ex)
            print(f"Input: {ex} -> Predicted price: ${p:,.0f}")
    else:
        if None in (args.area, args.bedrooms, args.location, args.age):
            raise SystemExit("Provide all of --area --bedrooms --location --age, or use --example")
        pred = predict_from_args(pipeline, args.area, args.bedrooms, args.location, args.age)
        print(f"Predicted price: ${pred:,.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/house_price_model.joblib", help="Path to saved model pipeline")
    parser.add_argument("--area", type=float, help="Area (sqft)")
    parser.add_argument("--bedrooms", type=int, help="Number of bedrooms")
    parser.add_argument("--location", type=str, help="Location code (e.g., A, B, C)")
    parser.add_argument("--age", type=float, help="House age in years")
    parser.add_argument("--example", action="store_true", help="Run example predictions")
    args = parser.parse_args()
    main(args)
