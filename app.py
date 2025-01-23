import streamlit as st
import joblib

# Load the trained logistic regression model
model = joblib.load("logistic_model_clean.pkl")

st.title("Customer Purchase Prediction App")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
gender_0, gender_1 = (1, 0) if gender == "Female" else (0, 1)  # One-hot encoding
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
number_of_purchases = st.number_input("Number of Purchases", min_value=0, value=5)
product_category = st.number_input("Product Category", min_value=0, value=0)  # Adjust based on your dataset
time_spent = st.number_input("Time Spent on Website (minutes)", min_value=0, value=20)
loyalty_program = st.selectbox("Loyalty Program", ["No", "Yes"])
loyalty_0, loyalty_1 = (1, 0) if loyalty_program == "No" else (0, 1)  # One-hot encoding
discounts_availed = st.number_input("Discounts Availed", min_value=0, value=0)

# Combine all features into a list
features = [
    age,                # Age
    gender_0,           # Gender=0
    gender_1,           # Gender=1
    annual_income,      # AnnualIncome
    number_of_purchases, # NumberOfPurchases
    product_category,   # ProductCategory
    time_spent,         # TimeSpentOnWebsite
    loyalty_0,          # LoyaltyProgram=0
    loyalty_1,          # LoyaltyProgram=1
    discounts_availed   # DiscountsAvailed
]

# Predict when the button is clicked
if st.button("Predict Purchase Status"):
    prediction = model.predict([features])[0]
    result = "Will Make a Purchase" if prediction == 1 else "Will Not Make a Purchase"
    st.success(f"Prediction: {result}")

