from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load the trained model and preprocessors
try:
    with open('Diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('pca.pkl', 'rb') as file:
        pca = pickle.load(file)
except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model = None
    scaler = None
    pca = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        hba1c_level = float(request.form['hba1c_level'])
        blood_glucose_level = int(request.form['blood_glucose_level'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        gender = request.form['gender']
        smoking_history = request.form['smoking_history']

        # Create a DataFrame with the input data (original features)
        input_df_original = pd.DataFrame({
            'age': [age], 'hypertension': [hypertension], 'heart_disease': [heart_disease],
            'smoking_history': [smoking_history], 'bmi': [bmi], 'HbA1c_level': [hba1c_level],
            'blood_glucose_level': [blood_glucose_level], 'gender': [gender]
        })

        # One-Hot Encode the input data
        input_df_encoded = pd.get_dummies(input_df_original, columns=['gender', 'smoking_history'], drop_first=True)

        # Get the feature names that the scaler was fitted on (excluding the target)
        expected_columns = [col for col in scaler.feature_names_in_ if col != 'diabetes']

        # Add any missing columns with 0 and reorder
        for col in expected_columns:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0
        input_df_encoded = input_df_encoded[expected_columns]

        # Standardize the input data using the loaded scaler
        input_X_scaled = scaler.transform(input_df_encoded)

        # Apply PCA transformation using the loaded PCA
        input_X_pca = pca.transform(input_X_scaled)

        prediction_result = "Error during prediction"
        if model is not None and pca is not None and scaler is not None:
            try:
                prediction = model.predict(input_X_pca)
                if prediction[0] == 1:
                    prediction_result = "High risk of Diabetes"
                else:
                    prediction_result = "Low risk of Diabetes"
            except Exception as e:
                prediction_result = f"Error making prediction: {e}"
        elif model is None:
            prediction_result = "Model not loaded."

        return render_template('result.html', prediction=prediction_result)

    return "Error: Invalid request method"

if __name__ == '__main__':
    app.run(debug=True)
    