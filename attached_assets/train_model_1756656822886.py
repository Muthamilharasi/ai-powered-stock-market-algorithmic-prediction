import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 1️⃣ Load your CSV file
df = pd.read_csv("HIST_CSV.csv")  # replace with your CSV file path

# 2️⃣ Preprocess data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Extract day, month, year as features
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Features (adjust according to your CSV)
X = df[['Closing Volume', 'Volume', '50-Day Moving Average', '200-Day Moving Average', 'Day', 'Month', 'Year']]

# Target (what we want to predict)
y = df['Closing Volume']  # or use 'Close' column if available

# 3️⃣ Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4️⃣ Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Test the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Model trained! Test MSE: {mse:.2f}")

# 6️⃣ Save the model
joblib.dump(model, "stock_model.pkl")
print("💾 Model saved as stock_model.pkl")
