from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(data.target_names.tolist(), "classes.pkl")

print("Model trained and saved successfully!")