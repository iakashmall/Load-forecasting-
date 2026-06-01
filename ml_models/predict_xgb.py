from xgb_model import XGBoostForecaster
from data.data_loader import load_data, train_test_split
from data.preprocessing import build_feature_matrix
import pandas as pd

# Load dataset
df = load_data()

# Split
train_df, test_df = train_test_split(df, test_months=3)

# Build features
X_test, y_test = build_feature_matrix(test_df)

# Load trained model
model = XGBoostForecaster.load()

# Predict
preds = model.predict_one_step(X_test)

print(preds[:10])

#Comparison of ctial vs predicted values
comparison = pd.DataFrame({
    "Actual": y_test.values[:len(preds)],
    "Predicted": preds
})

print(comparison.head(10))

#Plotting actual vs predicted values
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

plt.plot(y_test.values[:200], label="Actual")
plt.plot(preds[:200], label="Predicted")

plt.xlabel("Time")
plt.ylabel("Power Load (kW)")
plt.title("XGBoost Load Forecast")

plt.legend()
plt.grid(True)

plt.show()

