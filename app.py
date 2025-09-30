from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/titanic_model.joblib")

app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expect JSON data
    df = pd.DataFrame([data])  # Convert JSON to DataFrame

    # Ensure columns match training features
    # For simplicity, assume you send preprocessed features
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][prediction]

    result = {
        "prediction": int(prediction),
        "probability": float(probability)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
import requests

data = {
    "Pclass": 3,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Sex_male": 1,
    "Embarked_Q": 0,
    "Embarked_S": 1
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expect JSON data
    df = pd.DataFrame([data])  # Convert JSON to DataFrame

    # Ensure columns match training features
    # For simplicity, assume you send preprocessed features
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][prediction]

    result = {
        "prediction": int(prediction),
        "probability": float(probability)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
import requests

data = {
    "Pclass": 3,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Sex_male": 1,
    "Embarked_Q": 0,
    "Embarked_S": 1
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)
print(response.json())