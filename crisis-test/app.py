from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_crisis_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([[
        data['hydration_level'],
        data['physical_activity'],
        data['temperature_exposure'],
        data['sleep_quality'],
        data['medication_adherence'],
        data['pain'],
        data['weakness'],
        data['headache'],
        data['dizziness'],
        data['jaundice'],
        data['swelling'],
        data['shortness_of_breath'],
        data['fever'],
        data['chest_pain'],
        data['vision_problems'],
        data['erection'],
        data['stroke_symptoms']
    ]])
    prediction = model.predict(features)
    return jsonify({'crisis_risk': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
