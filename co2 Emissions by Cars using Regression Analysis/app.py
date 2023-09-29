import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from pickle import load

# loading the saved model
loaded_model = load(open('E:/ExcelR/Data Science/DATA SCIENCE PROJECTS/Project-3-Regression/trained_model.sav', 'rb'))

def co2_emission_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def user_input_features():
    engine_size = st.sidebar.number_input("Engine size:")
    cylinders = st.sidebar.number_input("No. of Cylinders:")
    fuel_consumption_city = st.sidebar.number_input("City Fuel Consumption:")
    fuel_consumption_hwy = st.sidebar.number_input("Highway Fuel Consumption")
    fuel_consumption_comb = st.sidebar.number_input("Combined Fuel Consumption")
    fuel_consumption_comb_mpg = st.sidebar.number_input("Comined Fuel Consumption mpg")

    if st.button('Predict CO2 Emission'):
        emission = co2_emission_prediction([engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_comb_mpg])
        st.success(emission)

if __name__ == '__main__':
    st.title('Model Deployment: Regression Analysis')
    st.sidebar.header('User Input Parameters')
    user_input_features()
