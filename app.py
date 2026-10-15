import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature names
with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load dataset
df = pd.read_csv("road_network.csv")

# Streamlit application
st.title("Road Traffic Prediction App")
st.write("Predict average daily traffic using machine learning")

# User inputs
length_km = st.number_input("Road Length (km)", min_value=0.0, value=4.0)

lanes = st.number_input("Number of Lanes", min_value=1, value=2)

speed_limit_kmh = st.number_input("Speed Limit (km/h)", min_value=0, value=60)

capacity_vehicles_per_hour = st.number_input(
    "Vehicle Capacity per Hour",
    min_value=0,
    value=2000
)

has_bus_lane = st.selectbox("Bus Lane", [0, 1])

has_bike_lane = st.selectbox("Bike Lane", [0, 1])

toll_rate_rp = st.number_input("Toll Rate (Rp)", min_value=0, value=0)

# Additional categorical inputs
road_type = st.selectbox(
    "Road Type",
    sorted(df["road_type"].dropna().unique())
)

surface_condition = st.selectbox(
    "Surface Condition",
    sorted(df["surface_condition"].dropna().unique())
)

district = st.selectbox(
    "District",
    sorted(df["district"].dropna().unique())
)

# Prediction
if st.button("Predict Traffic"):

    input_data = pd.DataFrame([{
        "length_km": length_km,
        "lanes": lanes,
        "speed_limit_kmh": speed_limit_kmh,
        "capacity_vehicles_per_hour": capacity_vehicles_per_hour,
        "has_bus_lane": has_bus_lane,
        "has_bike_lane": has_bike_lane,
        "toll_rate_rp": toll_rate_rp,
        "road_type": road_type,
        "surface_condition": surface_condition,
        "district": district
    }])

    # Convert categorical columns into numerical columns
    input_data = pd.get_dummies(
        input_data,
        columns=["road_type", "surface_condition", "district"],
        dtype=int
    )

    # Match the features used during training
    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict traffic
    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Average Daily Traffic: {prediction[0]:,.0f} vehicles"
    )