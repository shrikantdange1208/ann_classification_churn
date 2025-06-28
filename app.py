import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

## Load the pre-trained model
model = tf.keras.models.load_model('ann_model.h5')

## Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
    
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geography = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    

## Streamlit App
st.title('Customer Churn Prediction')

#User input
geography = st.selectbox('Geography', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## One-hot encode Geography
geography_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))

## Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

## Scale the input data
input_data_scaled = scaler.transform(input_data)


## Predict churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(f'Prediction Probability: {prediction_probability:.2f}')
if prediction_probability > 0.5:
    st.warning('The model predicts that the customer is likely to churn.')
else:
    st.success('The model predicts that the customer is not likely to churn.')