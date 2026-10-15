import pickle
import pandas as pd
import streamlit as st

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Road Traffic Prediction App")
st.write("Predict average daily traffic using Machine Learning")

# User inputs
length = st.number_input("Road Length (km)", min_value=0.0, value=4.3)

lanes = st.number_input("Number of Lanes", min_value=1, value=4)

speed = st.number_input("Speed Limit (km/h)", min_value=0, value=60)

capacity = st.number_input("Vehicle Capacity per Hour", min_value=0, value=5384)

bus_lane = st.selectbox("Has Bus Lane", [0, 1])

bike_lane = st.selectbox("Has Bike Lane", [0, 1])

toll = st.number_input("Toll Rate (Rp)", min_value=0, value=0)

# Prediction
if st.button("Predict Traffic"):

    input_data = pd.DataFrame([{
        "length_km": length,
        "lanes": lanes,
        "speed_limit_kmh": speed,
        "capacity_vehicles_per_hour": capacity,
        "has_bus_lane": bus_lane,
        "has_bike_lane": bike_lane,
        "toll_rate_rp": toll
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Average Daily Traffic: {prediction[0]:,.0f} vehicles/day"
    )