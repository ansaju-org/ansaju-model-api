from flask import Flask, jsonify, request
import os
import tensorflow as tf
import numpy as np
import joblib

SECRET_TOKEN = os.environ.get('SECRET_TOKEN') or 'ansaju'
MODEL_PATH = os.environ.get('MODEL_PATH') or 'model_ansaju.h5'

# model = joblib.load(MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)

def require_auth(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("x-api-token", "")
        if token != SECRET_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    data = request.get_json()
    features = data['features']
    
    input_array = np.array([features])
    pred_proba = model.predict(input_array)
    pred_index = np.argmax(pred_proba)
    pred_label = label_encoder.inverse_transform([pred_index])[0]

    return jsonify(prediction=pred_label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)