import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open('Customer_churn_model.pkl', 'rb'))  
scaler = pickle.load(open('scaler.pkl', 'rb'))


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("churn.css")


# Streamlit App UI
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict whether they will churn or stay.")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=560)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=100, value=46)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=6)
balance = st.number_input("Balance", min_value=0.0, max_value=260000.0, value=45000.07)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card?", [1, 0])  # 1 = Yes, 0 = No
is_active_member = st.selectbox("Is Active Member?", [1, 0])  # 1 = Yes, 0 = No
estimated_salary = st.number_input("Estimated Salary", min_value=1.00, value=200000.0)

# Predict button
if st.button("Predict"):
    # Gender Encoding
    gender_map = {'Female': 0, 'Male': 1}
    gender_encoded = gender_map.get(gender)

    # Geography One-Hot Encoding
    geography_germany = 1 if geography == 'Germany' else 0
    geography_spain = 1 if geography == 'Spain' else 0

    # Prepare data for scaling (only the features used during scaler training)
    try:
        scaled_features = np.array([[credit_score, age, balance, num_of_products, estimated_salary, tenure]])
        scaled_values = scaler.transform(scaled_features)

        # Combine scaled + categorical data (if model expects 11 features)
        final_input = np.concatenate((scaled_values[0], [gender_encoded, has_cr_card, is_active_member, geography_germany, geography_spain])).reshape(1, -1)

        # Prediction
        prediction = model.predict(final_input)

        # Result
        if prediction[0] == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")

    except ValueError as e:
        st.error(f"❌ Error: {e}")
        st.warning("Scaler may not match — retrain with correct features.")




