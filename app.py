# x-order 'department', 'team', 'targeted_productivity', 'over_time', 'incentive',
# 'no_of_workers', 'actual_productivity'
#scaler is exported as scaler.pkl
#model is exported as model.pkl
import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Employee Performance Prediction")
st.divider()
st.write("Please provide the following details:")
st.divider()

departments = ['sewing', 'finishing', "Engineering"]
department = st.selectbox("Department", departments)

targeted_productivity = st.number_input("Targeted Productivity", min_value=0, max_value=500, value=250)
over_time = st.number_input("Over Time (in hours)", min_value=0, max_value=20, value=0)
incentive = st.number_input("Incentive (in $)", min_value=0, max_value=10000, value=1000)
no_of_workers = st.number_input("Number of Workers", min_value=1, max_value=100, value=10)
actual_productivity = st.number_input("Actual Productivity", min_value=0, max_value=500, value=200)

# Encode department as needed (example: label encoding)
department_map = {dept: idx for idx, dept in enumerate(departments)}
department_encoded = department_map[department]

# Correct order: department, team, targeted_productivity, over_time, incentive, no_of_workers, actual_productivity
x = [department_encoded, targeted_productivity, over_time, incentive, no_of_workers, actual_productivity]

st.divider()
predictionbutton = st.button("Predict Performance")
if predictionbutton:
    x1 = np.array(x).reshape(1, -1)
    x_array = scaler.transform(x1)
    prediction = model.predict(x_array)[0]
    st.write("Predicted performance of employee:", prediction)
    if prediction < 0.5:
        st.write("Employee performance is below average.")
    elif 0.5 <= prediction < 0.75:
        st.write("Employee performance is average.")
    else:
        st.write("Employee performance is above average.")
