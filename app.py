import streamlit as st
import pickle
import pandas as pd

def main():
    st.title('Streamlit Loan Eligibility Prediction App')

    left, right = st.columns((2, 2))

    # User inputs based on provided columns
    ApplicantIncome = left.number_input('Applicant Income', step=1000.0, format="%.2f")
    CoapplicantIncome = right.number_input('Coapplicant Income', step=1000.0, format="%.2f")
    LoanAmount = left.number_input('Loan Amount', step=1000.0, format="%.2f")
    Loan_Amount_Term = right.number_input('Loan Amount Term (in months)', step=1, min_value=1)
    Credit_History = st.selectbox('Credit History', (1.0, 0.0))

    Gender = left.selectbox('Gender', ('Male', 'Female'))
    Married = right.selectbox('Married', ('Yes', 'No'))

    Dependents = st.selectbox('Dependents', ('No Dependents', 'One Dependent', 'Two Dependents', 'Three or More Dependents'))

    Education = left.selectbox('Education', ('Graduate', 'Not Graduate'))
    Self_Employed = right.selectbox('Self-Employed', ('Yes', 'No'))

    Property_Area = st.selectbox('Property Area', ('Rural', 'Semiurban', 'Urban'))

    button = st.button('Predict')

    # If predict button is clicked
    if button:
        # Map dependents to one-hot encoding
        Dependents_mapping = {
            'No Dependents': [1, 0, 0, 0],
            'One Dependent': [0, 1, 0, 0],
            'Two Dependents': [0, 0, 1, 0],
            'Three or More Dependents': [0, 0, 0, 1]
        }
        Dependents_encoded = Dependents_mapping[Dependents]

        # Call predict function
        result = predict(ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
                         Credit_History, Gender, Married, Dependents_encoded, Education,
                         Self_Employed, Property_Area)

        st.success(f'You are {result} for the loan')

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    train_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict(applicant_income, coapplicant_income, loan_amount, loan_amount_term,
            credit_history, gender, married, dependents_encoded, education,
            self_employed, property_area):
    # Encode inputs
    gen = 0 if gender == 'Male' else 1
    mar = 0 if married == 'Yes' else 1
    edu = 0 if education == 'Graduate' else 1
    sem = 0 if self_employed == 'Yes' else 1

    # Process property area (one-hot encoding)
    property_area_rural = 1 if property_area == 'Rural' else 0
    property_area_semiurban = 1 if property_area == 'Semiurban' else 0
    property_area_urban = 1 if property_area == 'Urban' else 0

    # Combine inputs in the correct order
    features = [
        applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history,
        gen, mar, *dependents_encoded, edu, sem, property_area_rural, property_area_semiurban, property_area_urban
    ]

    # Convert features to DataFrame for scaling
    columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
               'Credit_History', 'Gender', 'Married', 'Dependents_0', 'Dependents_1',
               'Dependents_2', 'Dependents_3_plus', 'Education', 'Self_Employed',
               'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']

    input_df = pd.DataFrame([features], columns=columns)

    # Scale numerical inputs
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    if input_df[numerical_cols].isnull().any().any():
        st.error("Please provide valid inputs for all fields.")
        return

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make predictions
    prediction = train_model.predict(input_df)
    verdict = 'Not Eligible' if prediction == 0 else 'Eligible'
    return verdict

if __name__ == '__main__':
    main()
