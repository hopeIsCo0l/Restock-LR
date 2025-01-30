import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# ------------------------------
# 1. Load Preprocessed Data
# ------------------------------
processed_data_folder = "processed_data"

X_train = np.load(os.path.join(processed_data_folder, "X_train.npy"))
X_test = np.load(os.path.join(processed_data_folder, "X_test.npy"))
y_train = np.load(os.path.join(processed_data_folder, "y_train.npy"))
y_test = np.load(os.path.join(processed_data_folder, "y_test.npy"))

print(f"\n✅ Loaded preprocessed data successfully!")
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# ------------------------------
# 2. Initialize the Logistic Regression Model
# ------------------------------
model = LogisticRegression(
    solver="lbfgs",  # Efficient for small datasets
    max_iter=200,    # Increase iterations to ensure convergence
    C=1.0,           # Regularization strength (default)
    random_state=42  # Ensures reproducibility
)

# ------------------------------
# 3. Train the Model
# ------------------------------
print("\n🔄 Training the model...")
model.fit(X_train, y_train)

print("\n✅ Model training completed!")

# ------------------------------
# 4. Evaluate Model Performance
# ------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# ------------------------------
# 5. Save the Trained Model
# ------------------------------
model_folder = "trained_model"
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, "logistic_regression_model.pkl")
joblib.dump(model, model_path)

print(f"\n💾 Model saved successfully at: {model_path}")
