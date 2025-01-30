import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import os

# Load dataset
df = pd.read_csv("restocking_data.csv")

# ------------------------------
# 1. Handling Missing Values
# ------------------------------
print("\nChecking for missing values:")
print(df.isnull().sum())

# Fill missing values if any
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical missing values

# ------------------------------
# 2. Encoding Categorical Variables
# ------------------------------
categorical_features = ["Product_Category", "Sales_Trend", "Promotion_Active"]

# Convert categorical variables using One-Hot Encoding
column_transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features)
    ],
    remainder="passthrough"
)

# Separate features and target
X = df.drop("Restock_Needed", axis=1)
y = df["Restock_Needed"]

# Apply one-hot encoding transformation
X_encoded = column_transformer.fit_transform(X)

# ------------------------------
# 3. Feature Scaling (Min-Max Scaling)
# ------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_encoded)

# ------------------------------
# 4. Splitting Data into Training and Testing Sets
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ------------------------------
# 5. Save Preprocessed Data
# ------------------------------
processed_data_folder = "processed_data"
os.makedirs(processed_data_folder, exist_ok=True)

np.save(os.path.join(processed_data_folder, "X_train.npy"), X_train)
np.save(os.path.join(processed_data_folder, "X_test.npy"), X_test)
np.save(os.path.join(processed_data_folder, "y_train.npy"), y_train)
np.save(os.path.join(processed_data_folder, "y_test.npy"), y_test)

print("\n✅ Preprocessed data saved successfully!")
