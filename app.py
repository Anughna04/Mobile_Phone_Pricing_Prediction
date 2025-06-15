# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("mobile_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Title
st.title("📱 Mobile Price Range Predictor")
st.markdown("Predict whether a mobile phone is low, medium, high, or very high cost.")

# Define price mapping
price_map = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

def predict_and_display(df):
    df_scaled = scaler.transform(df[feature_names])
    predictions = model.predict(df_scaled)
    df["Predicted Price Range"] = [price_map[p] for p in predictions]
    st.dataframe(df)

    if len(df) == 1:
        st.success(f"🔮 The mobile is predicted to be **{price_map[predictions[0]]}**.")

# Choose input method
option = st.radio("Select input method:", ("Manual Entry", "Upload CSV"))

if option == "Manual Entry":
    st.header("Enter Mobile Features")

    battery_power = st.number_input("Battery Power (mAh)", 500, 2000, 1000)
    blue = st.selectbox("Bluetooth", [0, 1])
    clock_speed = st.number_input("Clock Speed (GHz)", 0.5, 3.0, 1.5)
    dual_sim = st.selectbox("Dual SIM", [0, 1])
    fc = st.slider("Front Camera (MP)", 0, 20, 5)
    four_g = st.selectbox("4G", [0, 1])
    int_memory = st.slider("Internal Memory (GB)", 2, 64, 16)
    m_deep = st.number_input("Mobile Depth (cm)", 0.1, 1.0, 0.5)
    mobile_wt = st.slider("Weight (g)", 80, 250, 150)
    n_cores = st.slider("No. of Cores", 1, 8, 4)
    pc = st.slider("Primary Camera (MP)", 0, 20, 10)
    px_height = st.slider("Pixel Height", 0, 1960, 800)
    px_width = st.slider("Pixel Width", 500, 2000, 1200)
    ram = st.slider("RAM (MB)", 256, 4000, 2000)
    sc_h = st.slider("Screen Height (cm)", 5, 20, 10)
    sc_w = st.slider("Screen Width (cm)", 2, 15, 5)
    talk_time = st.slider("Talk Time (hours)", 2, 20, 10)
    three_g = st.selectbox("3G", [0, 1])
    touch_screen = st.selectbox("Touch Screen", [0, 1])
    wifi = st.selectbox("WiFi", [0, 1])

    user_input = pd.DataFrame([[
        battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
        m_deep, mobile_wt, n_cores, pc, px_height, px_width, ram,
        sc_h, sc_w, talk_time, three_g, touch_screen, wifi
    ]], columns=feature_names)

    if st.button("Predict"):
        predict_and_display(user_input)

elif option == "Upload CSV":
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            missing_cols = [col for col in feature_names if col not in data.columns]
            if missing_cols:
                st.warning(f"CSV is missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("Predict for Uploaded Data"):
                    predict_and_display(data)
        except Exception as e:
            st.error(f"Error reading file: {e}")
