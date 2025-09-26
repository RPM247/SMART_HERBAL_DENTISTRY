import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

IC50 = 100      
HILL_N = 1.0     
LOCAL_VOLUME = 0.1 
RELEASE_FRAC = 0.5 

def map_mmp8_to_target_inhibition(mmp8_value):
    """Maps the predicted MMP-8 value to a required target inhibition level."""
    if mmp8_value < 50:
        return 0.4  
    elif mmp8_value < 120:
        return 0.7  
    else:
        return 0.9  

def required_concentration(ic50, target_effect, hill_coefficient):
    """Calculates the required drug concentration for a target effect using the Hill equation."""
    if target_effect >= 1.0:
        return float('inf')
    base = target_effect / (1 - target_effect)
    if base < 0:
        return float('nan')
    return ic50 * (base ** (1 / hill_coefficient))

def required_loading(c_req, volume, release_fraction):
    """Calculates the required total dose in milligrams."""
    total_ng = (c_req * volume) / release_fraction
    total_mg = total_ng / 1_000_000
    return total_mg


app = Flask(__name__)

clf_model = joblib.load('xgboost_classifier_pipeline.joblib')
reg_model = joblib.load('xgboost_regressor_pipeline.joblib')
label_encoder = joblib.load('group_label_encoder.joblib')

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_values = request.form.to_dict()

    input_data = pd.DataFrame({
        'age': [float(form_values['age'])],
        'gender': [form_values['gender']],
        'diabetic_status': [form_values['diabetic_status']],
        'oral_temp': [float(form_values['oral_temp'])],
        'salivary_ph': [float(form_values['salivary_ph'])],
        'IL1B': [float(form_values['il1b'])],
        'TNF': [float(form_values['tnf'])],
        'hemoglobin': [float(form_values['hemoglobin'])],
        'albumin': [float(form_values['albumin'])],
        'bacterial_growth': [float(form_values['bacterial_growth'])]
    })
    
    group_pred_encoded = clf_model.predict(input_data)
    predicted_group = label_encoder.inverse_transform(group_pred_encoded)[0]

    regression_pred = reg_model.predict(input_data)
    predicted_mmp8_value = regression_pred[0][0]
    predicted_mmp8_text = f"{predicted_mmp8_value:.2f} ng/ml"
    predicted_score = f"{regression_pred[0][1]:.1f} / 100"

    target_inhibition = map_mmp8_to_target_inhibition(predicted_mmp8_value)
    req_conc = required_concentration(IC50, target_inhibition, HILL_N)
    dose_mg = required_loading(req_conc, LOCAL_VOLUME, RELEASE_FRAC)
    predicted_dosage = f"{dose_mg:.6f} mg"

    return render_template('predict.html', 
                           prediction_group=f'Predicted Group: {predicted_group}',
                           prediction_mmp8=f'Predicted MMP-8: {predicted_mmp8_text}',
                           prediction_score=f'Inflammation Score: {predicted_score}',
                           prediction_dosage=f'Mimosa Pudica Dosage: {predicted_dosage}')

if __name__ == '__main__':
    app.run(debug=True)