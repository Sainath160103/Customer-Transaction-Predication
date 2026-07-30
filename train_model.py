import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
train = pd.read_csv(
    r"E:\Customer_Transaction_Prediction\Data\train.csv"
)

# Select 200 features
features = [
    col for col in train.columns
    if col.startswith("var_")
]

X = train[features]
y = train["target"]

print("Number of features:", len(features))
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

# Test
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Save model
joblib.dump(
    model,
    "customer_transaction_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model saved successfully!")