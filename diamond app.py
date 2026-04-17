import streamlit as st
import numpy as np
import pandas as pd
import joblib


price_model = joblib.load("best_model.pkl")
cluster_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = price_model.feature_names_in_

st.set_page_config(page_title="💎 Diamond Predictor", layout="wide")

st.title("💎 Diamond Price & Market Segment Predictor")


st.subheader("Enter Diamond Details")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", min_value=0.1, step=0.1, value=0.5)
    x = st.number_input("Length (x)", min_value=0.1, value=5.0)
    y = st.number_input("Width (y)", min_value=0.1, value=5.0)

with col2:
    z = st.number_input("Depth (z)", min_value=0.1, value=3.0)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
    color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])


def preprocess_input(carat, x, y, z, cut, color, clarity):

    input_df = pd.DataFrame([{
        'carat': carat,
        'x': x,
        'y': y,
        'z': z,
        'cut': cut,
        'color': color,
        'clarity': clarity
    }])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Align columns with training data
    input_df = input_df.reindex(columns=columns, fill_value=0)

    return input_df


col3, col4 = st.columns(2)


with col3:
    if st.button("Predict Price 💰"):

        input_data = preprocess_input(carat, x, y, z, cut, color, clarity)

        # 🔥 Predict (log → original scale)
        prediction_log = price_model.predict(input_data)
        prediction = np.exp(prediction_log)[0]

        st.success(f"💰 Predicted Price: ₹ {prediction:,.2f}")



with col4:
    if st.button("Predict Market Segment 🧠"):

        input_data = preprocess_input(carat, x, y, z, cut, color, clarity)

        # Scale data
        scaled_data = scaler.transform(input_data)

        cluster = cluster_model.predict(scaled_data)[0]

        # Cluster names
        cluster_names = {
            0: "Mid-range Balanced Diamonds ⚖️",
            1: "Premium Heavy Diamonds 💎",
            2: "Affordable Small Diamonds 💰"
        }

        cluster_label = cluster_names.get(cluster, "Unknown")

        st.success(f"📊 Cluster: {cluster}")
        st.info(f"🏷️ Segment: {cluster_label}")