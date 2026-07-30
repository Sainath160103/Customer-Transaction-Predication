import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
train = pd.read_csv(
    r"E:\Customer_Transaction_Prediction\Data\train.csv"
)

# One input feature
X = train[["var_0"]]

# Target
y = train["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create scaler
scaler = StandardScaler()

# Scale training data
X_train_scaled = scaler.fit_transform(X_train)

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Create model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train model
model.fit(
    X_train_scaled,
    y_train
)

# Test model
y_pred = model.predict(
    X_test_scaled
)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(
    model,
    "one_feature_model.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "one_feature_scaler.pkl"
)

print("Model saved successfully!")