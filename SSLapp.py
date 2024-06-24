import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and preprocessor
with open('best_decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Define the key features
features = {
    'anxiety_level': {'label': 'Anxiety Level', 'type': 'slider', 'min': 0, 'max': 10, 'value': 5},
    'self_esteem': {'label': 'Self Esteem', 'type': 'slider', 'min': 0, 'max': 10, 'value': 5},
    'mental_health_history': {'label': 'Do you have a history of mental health issues?', 'type': 'selectbox', 'options': ['No', 'Yes']},
    'sleep_quality': {'label': 'Sleep Quality', 'type': 'slider', 'min': 0, 'max': 10, 'value': 5},
    'academic_performance': {'label': 'Academic Performance', 'type': 'slider', 'min': 0, 'max': 10, 'value': 5},
    'social_support': {'label': 'Social Support', 'type': 'slider', 'min': 0, 'max': 10, 'value': 5},
}

# Create a form for user input
st.title('Student Stress Level Prediction')

st.markdown("""
Welcome to the Student Stress Level Prediction App. 
Please provide your information below to predict your stress level.
""")

user_inputs = {}
for feature, config in features.items():
    if config['type'] == 'slider':
        user_inputs[feature] = st.slider(config['label'], min_value=config['min'], max_value=config['max'], value=config['value'])
    elif config['type'] == 'selectbox':
        user_inputs[feature] = st.selectbox(config['label'], options=config['options'])

# Convert categorical inputs to numerical
user_inputs['mental_health_history'] = 1 if user_inputs['mental_health_history'] == 'Yes' else 0

# Convert user inputs into a DataFrame
user_data = pd.DataFrame([user_inputs])

# Preprocess the user inputs
preprocessed_data = preprocessor.transform(user_data)

# Make predictions
if st.button('Predict'):
    prediction = model.predict(preprocessed_data)
    st.write(f'Predicted Stress Level: {prediction[0]}')

st.markdown("""
---
**Note:** This prediction is based on the information you provided and may not be fully accurate. 
For a comprehensive assessment, please consult a professional.
""")
