import streamlit as st
import numpy as np
import joblib
import pandas as pd
import joblib

model = joblib.load('forest_cover_model.pkl')
columns = joblib.load('columns.pkl')

# Load model
model = joblib.load('forest_cover_model.pkl')

st.title("🌲 Forest Cover Type Prediction")

st.write("Enter input values:")



Elevation = st.number_input("Elevation", value=2500)
Aspect = st.number_input("Aspect", value=100)
Slope = st.number_input("Slope", value=10)

Horizontal_Distance_To_Hydrology = st.number_input("Horizontal Distance to Hydrology", value=200)
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance to Hydrology", value=0)

Horizontal_Distance_To_Roadways = st.number_input("Distance to Roadways", value=1000)

Hillshade_9am = st.number_input("Hillshade 9am", value=200)
Hillshade_Noon = st.number_input("Hillshade Noon", value=220)
Hillshade_3pm = st.number_input("Hillshade 3pm", value=150)

Horizontal_Distance_To_Fire_Points = st.number_input("Distance to Fire Points", value=1000)

Soil_Type = st.number_input("Soil Type", value=10)

Wilderness_Area = st.selectbox(
    "Wilderness Area",
    ["Aspen", "Lodgepole Pine", "Spruce/Fir", "Krummholz", "Ponderosa Pine", "Douglas-fir", "Cottonwood/Willow"]
)

# Inputs
input_df = pd.DataFrame(columns=columns)
input_df.loc[0] = 0

# Fill values
input_df['Elevation'] = Elevation
input_df['Aspect'] = Aspect
input_df['Slope'] = Slope
input_df['Horizontal_Distance_To_Hydrology'] = Horizontal_Distance_To_Hydrology
input_df['Vertical_Distance_To_Hydrology'] = Vertical_Distance_To_Hydrology
input_df['Horizontal_Distance_To_Roadways'] = Horizontal_Distance_To_Roadways
input_df['Hillshade_9am'] = Hillshade_9am
input_df['Hillshade_Noon'] = Hillshade_Noon
input_df['Hillshade_3pm'] = Hillshade_3pm
input_df['Horizontal_Distance_To_Fire_Points'] = Horizontal_Distance_To_Fire_Points
input_df['Soil_Type'] = Soil_Type

# Prediction
if st.button("Predict"):
    input_data = np.array([[Elevation, Aspect, Slope,
                            Horizontal_Distance_To_Hydrology,
                            Vertical_Distance_To_Hydrology,
                            Horizontal_Distance_To_Roadways,
                            Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
                            Horizontal_Distance_To_Fire_Points,
                            Soil_Type]])

   
    prediction = model.predict(input_df)
    st.success(f"Prediction: {prediction[0]}")