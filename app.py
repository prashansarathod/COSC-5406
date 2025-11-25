import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Airfare Price Prediction",
    page_icon="✈️",
    layout="wide"
)

# =========================================
# LOAD CSS
# =========================================
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================================
# LOAD MODEL + COLUMNS
# =========================================
model = pickle.load(open("final_rf_model.pkl", "rb"))
model_cols = pickle.load(open("model_columns.pkl", "rb"))

# =========================================
# PAGE TITLE
# =========================================
st.markdown('<h1 class="main-title">✈️ Airfare Price Prediction App</h1>', unsafe_allow_html=True)
st.write("### Fill in the flight details below to get an accurate estimate.")

# =========================================
# DROPDOWN OPTIONS
# =========================================
airline_list = [
    "IndiGo", "Air India", "Jet Airways", "SpiceJet", "Multiple carriers",
    "GoAir", "Vistara", "Air Asia", "Jet Airways Business",
    "Multiple carriers Premium economy", "Vistara Premium economy", "Trujet"
]

source_list = ["Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"]

destination_list = ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata", "Banglore"]

stop_list = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]

# =========================================
# INPUT SECTION
# =========================================
left, right = st.columns(2)

with left:
    airline = st.selectbox("✈ Airline", airline_list)
    source = st.selectbox("📍 Source", source_list)
    destination = st.selectbox("📌 Destination", destination_list)

with right:
    stops = st.selectbox("⏺ Total Stops", stop_list)
    date = st.date_input("📅 Date of Journey")
    dep_time = st.time_input("⏰ Departure Time")
    arr_time = st.time_input("🕒 Arrival Time")
    duration = st.number_input("⏳ Duration in Minutes", min_value=1)

# =========================================
# PREPROCESS INPUT
# =========================================
def preprocess():
    df = pd.DataFrame([{
        "Journey_Day": date.day,
        "Journey_Month": date.month,
        "Journey_Weekday": date.weekday(),
        "Is_Weekend": 1 if date.weekday() >= 5 else 0,

        "Dep_Hour": dep_time.hour,
        "Dep_Minute": dep_time.minute,
        "Arrival_Hour": arr_time.hour,
        "Arrival_Minute": arr_time.minute,

        "Duration_TotalMinutes": duration,
        "Duration_Hours": duration / 60,

        "Is_RedEye": 1 if dep_time.hour >= 22 or dep_time.hour <= 5 else 0,

        "Total_Stops_Num": {
            "non-stop": 0,
            "1 stop": 1,
            "2 stops": 2,
            "3 stops": 3,
            "4 stops": 4
        }[stops]
    }])

    # Add one-hot encoding columns
    for col in model_cols:
        if col.startswith("Airline_"):
            df[col] = 1 if col == f"Airline_{airline}" else 0
        elif col.startswith("Source_"):
            df[col] = 1 if col == f"Source_{source}" else 0
        elif col.startswith("Destination_"):
            df[col] = 1 if col == f"Destination_{destination}" else 0
        else:
            if col not in df.columns:
                df[col] = 0

    return df[model_cols]

# =========================================
# PREDICT
# =========================================
if st.button("Predict Fare 💰"):
    X = preprocess()
    y_log = model.predict(X)[0]
    price = int(np.expm1(y_log))

    st.markdown(
        f"<div class='result-card'>Estimated Airfare: ₹ {price}</div>",
        unsafe_allow_html=True
    )
