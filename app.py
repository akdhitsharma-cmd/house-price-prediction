import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("house_model.pkl", "rb"))

st.title("House Price Prediction")
st.write("Enter house details:")
size = st.number_input("Size (sqft)", min_value=100)
rooms = st.number_input("Number of Rooms", min_value=1)
age = st.number_input("Age of House (years)", min_value=0)
location = st.slider("Location Score (1-10)", 1, 10)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[size, rooms, age, location]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Price: {int(prediction[0]):,}")