import streamlit as st
import pandas as pd
import pickle
import os
from datetime import datetime
import sys
import traceback



st.write("Python version:", sys.version)
st.set_page_config(page_title="Car-Price_prediction")
st.header("Welcome to Price-Prediction")
df=pd.read_csv("copy.csv")
objects={}
for i in df.columns:
    if df[i].dtype==object:
        objects[i] = list(df[i].unique())
        objects[i].sort()


rmodel = None
try:
    if os.path.exists("RFmod.pkl"):
        with open("RFmod.pkl", "rb") as file:
            rmodel = pickle.load(file)
        st.success("Model loaded successfully (pickle).")
    else:
        st.error(" No model file found. Please upload ur pickle file.")

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.code(traceback.format_exc())
    rmodel = None








with  st.container(border=True): 
    Company=st.selectbox('Company',options=objects['Make'])
    Model=st.selectbox('Model',options=objects['Model'])
    Year = st.number_input('Year_manufactured',min_value=2000,max_value=2025)
    current_year = datetime.now().year
    current_age = current_year - Year
    st.write(f"Car's Current Age: {current_age} years")
    Engine_Size= st.number_input('Engine Size',min_value=1.0,max_value=4.5,step=0.1)
    Mileage=st.number_input('Mileage',min_value=56.0,max_value=199867.0,step = 10.0)
    Fuel_Type=st.selectbox('Fuel type',options=objects['Fuel Type'])
    Transmission=st.radio('Transmission',options=objects['Transmission']) 
     
    input_vals=[[objects['Make'].index(Company),objects['Model'].index(Model),Year,
             Engine_Size,Mileage,objects['Fuel Type'].index(Fuel_Type),
             objects['Transmission'].index(Transmission),current_age]]

    c1, c2, c3 = st.columns([1.5, 1.6, 1])
    if c2.button('Submit'):
       if rmodel is None:
         st.error("Model is not loaded. Please upload the correct Random Forest pickle file.")
       else:
        try:
            
            feature_names = ['Make', 'Model', 'Year', 'Engine Size', 'Mileage', 'Fuel Type', 'Transmission', 'current_age']
            
            
            input_df = pd.DataFrame(input_vals, columns=feature_names)
            
            
            
            out = rmodel.predict(input_df)
            st.subheader(f"Total_Price in Indian currency: {out[0] * 10}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
 