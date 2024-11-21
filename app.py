from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model = joblib.load('random_forest_crisis_model.pkl')

required_fields = [
    'hydration_level', 'physical_activity', 'temperature_exposure', 'sleep_quality',
    'medication_adherence', 'pain', 'weakness', 'headache', 'dizziness',
    'jaundice', 'swelling', 'shortness_of_breath', 'fever', 'chest_pain',
    'vision_problems', 'erection', 'stroke_symptoms'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received data: {data}")

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        input_data = pd.DataFrame([data])

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        risk = "At risk of crisis" if prediction[0] == 1 else "Not at risk"
        confidence = prediction_proba[0][prediction[0]]

        logging.info(f"Prediction: {risk}, Confidence: {confidence * 100:.2f}%")

        return jsonify({
            'prediction': risk,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True)