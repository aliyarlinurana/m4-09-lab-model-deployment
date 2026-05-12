![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Model Deployment

## Overview

A machine learning model that only lives inside a Jupyter Notebook can't serve predictions to users, applications, or business processes. Deployment bridges the gap between experimentation and real-world impact — it's the step that turns a trained model into a running service that other software can call.

In this lab you will walk through the complete deployment workflow for a scikit-learn classifier: train and serialize the model, wrap it in a Flask REST API with proper input validation, test the API with automated requests, and document it so that another developer could run it without your help.

This is deliberately a "first deployment" exercise — a single-process Flask server rather than a containerized production stack. The goal is to internalize the core pattern (serialize → serve → validate → test) so that you can layer on production concerns (Docker, CI/CD, monitoring) later.

## Learning Goals

By the end of this lab, you should be able to:

- Serialize a trained scikit-learn model with `joblib` and verify round-trip predictions.
- Build a Flask REST API with `/predict` and `/health` endpoints.
- Validate incoming request data and return meaningful error responses.
- Test an API programmatically using the `requests` library.

## Setup and Context

This lab spans two files: a Jupyter Notebook for model training and testing, and a standalone Python script for the Flask application. Both files live in the same project directory and share the serialized model file.

You'll train on the well-known Iris dataset (150 samples, 4 features, 3 species) — a small dataset that makes iteration fast while still exercising the full deployment pipeline.

## Requirements

### Fork and clone

1. Fork this repository to your own GitHub account.
2. Clone the fork to your local machine.
3. Navigate into the project directory.

### Python environment

```bash
pip install flask joblib requests scikit-learn numpy pandas
```

## Getting Started

1. Create a Jupyter Notebook called **`m4-09-model-deployment.ipynb`** (for Tasks 1, 3, and 4).
2. Create a Python file called **`app.py`** (for Task 2).
3. In the notebook's first cell, import your core libraries:

```python
import numpy as np
import pandas as pd
import joblib
import requests
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

4. Work through the tasks in order. Each task builds on the previous one.
5. Include markdown cells between code cells to explain your observations.

## Tasks

### Task 1: Train & Serialize

1. Load the Iris dataset (`load_iris()`). Split into 80/20 train/test (`random_state=42`).
2. Train a `RandomForestClassifier(n_estimators=100, random_state=42)` on the training set.
3. Evaluate on the test set — print the **accuracy** and the full **classification report**.
4. Save the trained model to a file called `model.joblib` using `joblib.dump()`.
5. In a new cell, load the model back with `joblib.load()` and predict on the same test set. Verify that the predictions are **identical** to the original (use `np.array_equal`).
6. Also save the **target names** (species labels) to `target_names.joblib` — the API will need them to return human-readable class names.

### Task 2: Build a Flask API

Create a file called **`app.py`** with the following structure:

1. **`/health` endpoint** (GET): Returns `{"status": "healthy"}` with a 200 status code. This is used by load balancers and monitoring tools to check if the service is alive.

2. **`/predict` endpoint** (POST): Accepts a JSON body with a `features` key containing a list of 4 numeric values (sepal length, sepal width, petal length, petal width).
   - Load the model and target names from the `.joblib` files.
   - Validate the input:
     - Check that the `features` key exists.
     - Check that it contains exactly 4 values.
     - Check that all values are numeric.
     - Return a 400 error with a descriptive message if validation fails.
   - Return a JSON response with `predicted_class` (string name) and `probabilities` (dict mapping class names to probabilities).

3. **`/predict_batch` endpoint** (POST): Accepts a JSON body with a `samples` key containing a list of feature arrays. Returns predictions for all samples in a single response.

4. Run the app with `app.run(debug=True, port=5000)`.

Here's a skeleton to get you started:

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

# Implement /predict and /predict_batch below

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

### Task 3: Test the API

**Before starting this task, open a terminal and run `python app.py` to start the Flask server.**

Back in your notebook:

1. **Health check**: Send a GET request to `http://localhost:5000/health` and verify the response.
2. **Single prediction**: Send a POST request to `/predict` with a valid Iris sample (e.g., `[5.1, 3.5, 1.4, 0.2]`). Print the predicted class and probabilities.
3. **Error handling**: Send at least three malformed requests and verify that the API returns appropriate 400 errors:
   - Missing `features` key.
   - Wrong number of features (e.g., 3 instead of 4).
   - Non-numeric values (e.g., `["a", "b", "c", "d"]`).
4. **Batch prediction**: Send a POST request to `/predict_batch` with 5 samples from the Iris test set. Verify that all predictions match what the model produces locally.
5. Document all test results in the notebook with clear output and markdown explanations.

### Task 4: Documentation & Reflection

1. Create a `README.md` file (separate from this lab README) inside a `deployment/` folder that documents your deployed model:
   - **What the model does** (one paragraph).
   - **How to run** (step-by-step commands).
   - **API specification** (endpoints, methods, request/response formats).
   - **Example requests and responses** (curl or Python `requests` examples).

2. In your notebook, write a reflection addressing:
   - What would need to change for a **production deployment**? (Think about: WSGI servers, containerization, environment variables, HTTPS.)
   - How would you handle **model versioning**? (What if you retrain and the new model is worse?)
   - What **monitoring** would you add? (Prediction latency, input drift, error rates.)
   - How would the architecture change if the model needed to handle **1,000 requests per second**?

## Submission

### What to submit
- `m4-09-model-deployment.ipynb` — completed notebook (Tasks 1, 3, 4).
- `app.py` — Flask application (Task 2).
- `deployment/README.md` — API documentation (Task 4).
- `model.joblib` and `target_names.joblib` — serialized model artifacts.

### Definition of done (checklist)
- [ ] Model is trained, evaluated, serialized, and round-trip verified.
- [ ] Flask app has `/health`, `/predict`, and `/predict_batch` endpoints with input validation.
- [ ] All API tests (valid, invalid, batch) are documented in the notebook.
- [ ] API documentation is complete with example requests/responses.
- [ ] Reflection addresses production deployment, versioning, monitoring, and scaling.
- [ ] The notebook runs top-to-bottom without errors (`Kernel → Restart & Run All`).

### How to submit (Git workflow)

```bash
git add .
git commit -m "lab: complete model deployment"
git push origin main
```

Then open a **Pull Request** on the original repository with a brief description of your work.
