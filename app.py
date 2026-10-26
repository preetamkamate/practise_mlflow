from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
classes = joblib.load("classes.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ML Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or "features" not in data:
        return jsonify({"error": "Provide features as a list"}), 400

    features = data["features"]

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Exactly four numeric features are required"}), 400

    try:
        features = [float(x) for x in features]
    except (TypeError, ValueError):
        return jsonify({"error": "Features must be numeric"}), 400

    prediction = model.predict([features])[0]

    return jsonify({
        "prediction": classes[int(prediction)]
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)