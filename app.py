from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # Validate features key exists
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in request body"}), 400

    features = data["features"]

    # Validate exactly 4 values
    if len(features) != 4:
        return jsonify({
            "error": f"Expected exactly 4 feature values, got {len(features)}"
        }), 400

    # Validate all values are numeric
    for i, val in enumerate(features):
        if not isinstance(val, (int, float)):
            return jsonify({
                "error": f"Non-numeric value at index {i}: '{val}'. All features must be numeric."
            }), 400

    X = np.array([features])
    pred_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return jsonify({
        "predicted_class": target_names[pred_idx],
        "probabilities": {name: round(float(p), 4) for name, p in zip(target_names, proba)}
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)

    if "samples" not in data:
        return jsonify({"error": "Missing 'samples' key in request body"}), 400

    samples = data["samples"]
    if not isinstance(samples, list) or len(samples) == 0:
        return jsonify({"error": "'samples' must be a non-empty list of feature arrays"}), 400

    # Validate each sample
    for i, sample in enumerate(samples):
        if len(sample) != 4:
            return jsonify({
                "error": f"Sample {i}: expected 4 features, got {len(sample)}"
            }), 400
        for j, val in enumerate(sample):
            if not isinstance(val, (int, float)):
                return jsonify({
                    "error": f"Sample {i}, index {j}: non-numeric value '{val}'"
                }), 400

    X = np.array(samples)
    pred_indices = model.predict(X)
    probas = model.predict_proba(X)

    results = []
    for pred_idx, proba in zip(pred_indices, probas):
        results.append({
            "predicted_class": target_names[int(pred_idx)],
            "probabilities": {name: round(float(p), 4) for name, p in zip(target_names, proba)}
        })

    return jsonify({"predictions": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
