```markdown
# House Price Prediction (Simple)

A minimal example to predict house prices from features like area, bedrooms, location, and age.
This uses scikit-learn with a preprocessing pipeline and a Random Forest regressor.

Contents
- houses.csv — sample dataset
- src/train.py — train and save the model (saves to `models/house_price_model.joblib`)
- src/predict.py — load saved model and run predictions
- requirements.txt — Python packages

Quickstart
1. Create and activate a Python virtual environment (recommended):
   - python -m venv venv
   - source venv/bin/activate  # macOS / Linux
   - venv\Scripts\activate     # Windows

2. Install requirements:
   - pip install -r requirements.txt

3. Train:
   - python src/train.py --data houses.csv --output models/house_price_model.joblib

   This prints train/test metrics (MAE, RMSE, R2) and saves the pipeline.

4. Predict:
   - Examples:
     - python src/predict.py --model models/house_price_model.joblib --area 1500 --bedrooms 3 --location B --age 10
     - python src/predict.py --model models/house_price_model.joblib --example

Dataset
- CSV columns: area (sqft), bedrooms (int), location (categorical), age (years), price (USD)

Notes & next steps
- This is a toy example. For real data:
  - Use more samples and richer features (lot size, bathrooms, year built, proximity to amenities, school ratings).
  - Consider hyperparameter tuning (GridSearchCV) and cross-validation.
  - Try other models (XGBoost, LightGBM).
  - Add feature engineering (log price, interactions).
```
