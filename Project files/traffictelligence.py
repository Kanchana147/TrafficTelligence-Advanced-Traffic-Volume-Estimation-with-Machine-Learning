import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("traffic volume.csv")

target_col = 'traffic_volume'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['Time'], format='%d-%m-%Y %H:%M:%S')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['second'] = df['datetime'].dt.second
holiday_mapping = {
    'None': 7,
    'Columbus Day': 1,
    'Veterans Day': 10,
    'Thanksgiving Day': 9,
    'Christmas Day': 0,
    "New Year's Day": 6,
    "Washington's Birthday": 11,
    'Memorial Day': 5,
    'Independence Day': 2,
    'State Fair': 8,
    'Labor Day': 3,
    'Martin Luther King Jr Day': 4
}

weather_mapping = {
    'Clouds': 1,
    'Clear': 0,
    'Rain': 4,
    'Drizzle': 2,
    'Mist': 5,
    'Haze': 3,
    'Thunderstorm': 10,
    'Snow': 8,
    'Squall': 9,
    'Smoke': 7
}

df['holiday'] = df['holiday'].map(holiday_mapping).fillna(7)
df['weather'] = df['weather'].map(weather_mapping).fillna(1)


df = df.dropna()  

X = df[['holiday', 'temp', 'rain', 'snow', 'weather',
        'year', 'month', 'day', 'hour', 'minute', 'second']]
y = df[target_col]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model has been trained successfully.")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ Model Testing Done.")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

results = X_test.copy()
results["actual"] = y_test
results["predicted"] = y_pred
results.to_csv("results.csv", index=False)
print("✅ Results saved to results.csv.")

os.makedirs("Flask", exist_ok=True)
with open("Flask/model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("✅ Model saved to Flask/model.pkl.")
print("✅ Saved model expects", model.n_features_in_, "features.")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=results, x="actual", y="predicted", alpha=0.5)
plt.plot([results["actual"].min(), results["actual"].max()],
         [results["actual"].min(), results["actual"].max()],
         color="red", linestyle="--")
plt.title("Actual vs Predicted Traffic Volume")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

plt.figure(figsize=(8, 6))
results["residuals"] = results["actual"] - results["predicted"]
sns.histplot(data=results, x="residuals", kde=True, color="green")
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()
