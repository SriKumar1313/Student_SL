import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and preprocessor
with open('best_decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# Define the features based on your dataset
features = ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
            'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
            'noise_level', 'living_conditions', 'safety', 'basic_needs',
            'academic_performance', 'study_load', 'teacher_student_relationship',
            'future_career_concerns', 'social_support', 'peer_pressure',
            'extracurricular_activities', 'bullying']

# Create a form for user input
st.title('Student Stress Level Prediction')

user_inputs = {}
for feature in features:
    user_inputs[feature] = st.number_input(f'Enter {feature}', min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# Convert user inputs into a DataFrame
user_data = pd.DataFrame([user_inputs])

# Preprocess the user inputs
preprocessed_data = preprocessor.transform(user_data)

# Make predictions
if st.button('Predict'):
    prediction = model.predict(preprocessed_data)
    st.write(f'Predicted Stress Level: {prediction[0]}')
