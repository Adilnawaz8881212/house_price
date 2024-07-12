import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor

# Load the model
with open('xgboost_model.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Load the DataFrame
with open('df.pkl', 'rb') as df_file:
    df = pickle.load(df_file)

# Title of the app
st.title("Real Estate Price Prediction")

# Sidebar for user input
a = st.selectbox('Location Name', df['location'].unique())
b = st.number_input("Bedrooms", 0, 100)
c = st.number_input("Bath", 0, 10)
d = st.number_input("Size", 0, 100000)

if st.button("Predict price"):
    # Prepare input data
    user_input = pd.DataFrame([[a,b,c,d]], columns=['location', 'bedrooms', 'baths', 'size'])

    # Make Prediction
    predicted_price = int(np.exp(pipeline.predict(user_input)[0]))
    

    # Display the Prediction with a meaningful message
    st.title(f'Your Home Price is: {predicted_price} PKR')
 

