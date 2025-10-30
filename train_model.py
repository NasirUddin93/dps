import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Create a more realistic and larger dataset
np.random.seed(42)

# Generate synthetic health data
n_samples = 1000

data = {
    'age': np.random.randint(18, 80, n_samples),
    'glucose': np.random.normal(100, 30, n_samples),  # Normal fasting glucose ~70-100
    'blood_pressure': np.random.normal(120, 20, n_samples),  # Normal BP ~120/80
    'disease': np.zeros(n_samples)
}

# Create more realistic disease risk logic
df = pd.DataFrame(data)

# Higher risk for: age > 50, glucose > 140, blood_pressure > 140
df['disease'] = (
    (df['age'] > 50).astype(int) * 0.4 +
    (df['glucose'] > 140).astype(int) * 0.3 +
    (df['blood_pressure'] > 140).astype(int) * 0.3
)

# Convert to binary classification (0 = low risk, 1 = high risk)
df['disease'] = (df['disease'] > 0.5).astype(int)

print("Dataset sample:")
print(df.head())
print(f"\nDisease distribution:\n{df['disease'].value_counts()}")

X = df[['age', 'glucose', 'blood_pressure']]
y = df['disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nFeature Importance:\n{feature_importance}")

# Save model
if not os.path.exists("model"):
    os.makedirs("model")

joblib.dump(model, "model/disease_model.pkl")
print("Model saved successfully in 'model/disease_model.pkl'")