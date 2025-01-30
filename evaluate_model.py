import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ------------------------------
# 1. Load the Preprocessed Data
# ------------------------------
processed_data_folder = "processed_data"

X_test = np.load(os.path.join(processed_data_folder, "X_test.npy"))
y_test = np.load(os.path.join(processed_data_folder, "y_test.npy"))

print(f"\n✅ Loaded test dataset successfully! Test data shape: {X_test.shape}")

# ------------------------------
# 2. Load the Trained Model
# ------------------------------
model_path = "trained_model/logistic_regression_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}. Train the model first.")

model = joblib.load(model_path)
print(f"\n✅ Loaded trained model from: {model_path}")

# ------------------------------
# 3. Make Predictions
# ------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Get probability scores for positive class

# ------------------------------
# 4. Compute Evaluation Metrics
# ------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n📊 Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

# ------------------------------
# 5. Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Restock", "Restock Needed"], yticklabels=["No Restock", "Restock Needed"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ------------------------------
# 6. ROC Curve
# ------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(alpha=0.7)
plt.show()
