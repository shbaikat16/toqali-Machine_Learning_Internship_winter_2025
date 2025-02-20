import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# Load the trained models
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
log_reg_model = pickle.load(open('logistic_regression.pkl', 'rb'))

# Streamlit UI
st.title("Hotel Booking Cancellation Prediction")
st.write("Enter the details below to predict whether a booking will be canceled.")

# User input fields
number_of_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=1)
number_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
number_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, max_value=7, value=1)
number_of_week_nights = st.number_input("Number of Week Nights", min_value=0, max_value=14, value=2)
car_parking_space = st.selectbox("Car Parking Space", [0, 1])
lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=30)
repeated = st.selectbox("Repeated Guest", [0, 1])
P_C = st.selectbox("Previous Cancellation", [0, 1])
P_not_C = st.selectbox("Previous Not-Canceled", [0, 1])
average_price = st.number_input("Average Price", min_value=0, max_value=1000, value=100)
special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=5, value=0)
reservation_year = st.number_input("Reservation Year", min_value=2000, max_value=2100, value=2024)
reservation_month = st.number_input("Reservation Month", min_value=1, max_value=12, value=1)
reservation_day = st.number_input("Reservation Day", min_value=1, max_value=31, value=1)
reservation_dayofweek = st.number_input("Reservation Day of Week", min_value=0, max_value=6, value=0)

# Categorical inputs
meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
room_type = st.selectbox("Room Type",
                         ["Room Type 1", "Room Type 2", "Room Type 3", "Room Type 4", "Room Type 5", "Room Type 6",
                          "Room Type 7"])
market_segment = st.selectbox("Market Segment", ["Corporate", "Online", "Offline", "Complementary"])

# Convert categorical inputs to binaryyy features
meal_plan_features = [1 if meal_plan == f"Meal Plan {i}" else 0 for i in range(2, 4)] + [
    1 if meal_plan == "Not Selected" else 0]
room_type_features = [1 if room_type == f"Room Type {i}" else 0 for i in range(2, 8)]
market_segment_features = [1 if market_segment == seg else 0 for seg in
                           ["Complementary", "Corporate", "Offline", "Online"]]

# Create input array
input_features = np.array([
                              number_of_adults, number_of_children, number_of_weekend_nights,
                              number_of_week_nights, car_parking_space, lead_time, repeated, P_C, P_not_C,
                              average_price, special_requests, reservation_year, reservation_month, reservation_day,
                              reservation_dayofweek
                          ] + meal_plan_features + room_type_features + market_segment_features).reshape(1, -1)

# Model selection
model_choice = st.radio("Select Model", ["KNN", "Logistic Regression"])

if st.button("Predict Cancellation"):
    if model_choice == "KNN":
        prediction = knn_model.predict(input_features)[0]
    else:
        prediction = log_reg_model.predict(input_features)[0]

    result = "CANCELED" if prediction == 1 else "NOT CANCELED"
    st.success(f"Prediction: {result}")
