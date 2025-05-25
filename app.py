from flask import Flask, jsonify, request
import joblib
import os

SECRET_TOKEN = os.environ.get('SECRET_TOKEN') or 'ansaju'
MODEL_PATH = os.environ.get('MODEL_PATH') or 'model_ansaju.pkl'

model = joblib.load(MODEL_PATH)

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

    prediction = model.predict([features])
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run()