from preprocessing import full_preprocessing_pipeline
from xgboost import XGBClassifier
import joblib
import os

# Get processed data
X_train, X_test, y_train, y_test = full_preprocessing_pipeline("data/raw_data.csv")

# Train model
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_rain_model.pkl")

print("Training complete and model saved.")
