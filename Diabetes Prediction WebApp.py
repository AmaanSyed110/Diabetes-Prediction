# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 22:31:02 2024

@author: shree
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open('D:/Git-Hub projects/Diabetes Prediction/trained_model.sav','rb'))

#creating function
def diabetes_prediction(input_data):
    #changing the input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)
    #reshaping the array
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
        return("The person is not diabetic")
    else:
        return("The person is diabetic")

def main():
    st.title('Diabetes prediction web app')
    
    
    gender=st.text_input('Gender (Male=1 Female=0)')
    age=st.text_input('Age')
    hypertension=st.text_input('Hypertension (Yes=1 No=0)')
    heart_disease=st.text_input('Heart Diseases (Yes=1 No=0)')
    smoking_history=st.text_input('Smoking History ( No info=0, current=1, ever=2, former=3, never=4, not current=5)')
    bmi=st.text_input('BMI')
    HbA1c_level=st.text_input('HbA1c_level')
    blood_glucose_level=st.text_input('blood glusoce level')
    
    diagnosis= ""
    #code for Prediction
    
    if st.button("Diabetes Test Results:"):
        diagnosis=diabetes_prediction([gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level])
        
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()