# ==========================================
# Predictive Analytics Using Historical Data
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("D:/Internship/Thiranex/Predictive_Analytics/dataset/synthetic_historical_sales.csv")

print("\nDataset Loaded Successfully!\n")
print(df.head())

# ==========================================
# Data Cleaning
# ==========================================

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
df["Advertising_Spend"] = df["Advertising_Spend"].fillna(
    df["Advertising_Spend"].mean()
)

df["Temperature"] = df["Temperature"].fillna(
    df["Temperature"].mean()
)

# ==========================================
# Feature Engineering
# ==========================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Save a copy for plotting later
plot_df = df.copy()

# ==========================================
# Select Features and Target
# ==========================================

X = df.drop(columns=["Sales", "Date"])
y = df["Sales"]

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "sales_prediction_model.pkl")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Model Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========\n")
print(f"Mean Absolute Error      : {mae:.2f}")
print(f"Mean Squared Error       : {mse:.2f}")
print(f"Root Mean Squared Error  : {rmse:.2f}")
print(f"R² Score                 : {r2:.4f}")

print("\nModel Trained Successfully!")
print("Prediction Completed Successfully!")
print("Model Saved Successfully!")

# ==========================================
# Visualization 1
# Historical Sales Trend
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(plot_df["Date"], plot_df["Sales"])

plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# Visualization 2
# Actual vs Predicted
# ==========================================

plt.figure(figsize=(12,5))

plt.plot(
    y_test.values[:50],
    label="Actual Sales",
    marker="o"
)

plt.plot(
    y_pred[:50],
    label="Predicted Sales",
    marker="x"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Samples")
plt.ylabel("Sales")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# Forecast Next 7 Days
# ==========================================

last_date = plot_df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7
)

avg_customers = df["Customers"].mean()
avg_ad = df["Advertising_Spend"].mean()
avg_discount = df["Discount_Percent"].mean()
avg_temp = df["Temperature"].mean()
avg_holiday = 0

future_df = pd.DataFrame({
    "Customers": [avg_customers] * 7,
    "Advertising_Spend": [avg_ad] * 7,
    "Discount_Percent": [avg_discount] * 7,
    "Temperature": [avg_temp] * 7,
    "Holiday": [avg_holiday] * 7,
    "Year": future_dates.year,
    "Month": future_dates.month,
    "Day": future_dates.day
})

future_sales = model.predict(future_df)

forecast = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_sales
})

print("\n========== NEXT 7 DAYS SALES FORECAST ==========\n")
print(forecast)

# ==========================================
# Visualization 3
# Future Forecast
# ==========================================

plt.figure(figsize=(10,5))

plt.plot(
    forecast["Date"],
    forecast["Predicted Sales"],
    marker="o"
)

plt.title("Next 7 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")

plt.xticks(rotation=45)

plt.grid(True)
plt.tight_layout()
plt.show()

print("\nProject Completed Successfully!")