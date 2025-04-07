import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Earthquake Magnitude Prediction")

# User input fields
latitude = st.number_input("Latitude", value=0.0)
longitude = st.number_input("Longitude", value=0.0)
depth = st.number_input("Depth", value=0.0)
nst = st.number_input("Nst", value=0.0)
gap = st.number_input("Gap", value=0.0)
clo = st.number_input("Clo", value=0.0)
rms = st.number_input("RMS", value=0.0)

# Prediction button
if st.button("Predict Magnitude"):
    user_input = pd.DataFrame([[latitude, longitude, depth, nst, gap, clo, rms]],
                              columns=["Latitude", "Longitude", "Depth", "Nst", "Gap", "Clo", "RMS"])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    st.success(f"Predicted Earthquake Magnitude: {prediction}")
