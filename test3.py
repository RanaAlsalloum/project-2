import joblib
import pandas as pd
import streamlit as st
import pickle as p

with open('final_model.pickle', 'rb') as f:
    pipe = p.load(f)

# Load the trained model and scaler
model = joblib.load(open("rf_model.joblib", 'rb'))
scaler = joblib.load(open("scaler.joblib", 'rb'))


# Set up the title of the app
st.title("Customer Spending Prediction")

# Create input fields for user data
avg_session_length = st.number_input("Avg. Session Length:", min_value=0.0)
time_on_app = st.number_input("Time on App:", min_value=0.0)
time_on_website = st.number_input("Time on Website:", min_value=0.0)
length_of_membership = st.number_input("Length of Membership:", min_value=0.0)



# Create a button to predict
if st.button("Predict"):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[avg_session_length, time_on_app, time_on_website, length_of_membership]])

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display the result
    st.success(f"Predicted Yearly Amount Spent: ${round(prediction, 2)}")

