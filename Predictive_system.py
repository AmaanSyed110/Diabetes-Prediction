# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model
loaded_model=pickle.load(open('D:/Git-Hub projects/Diabetes Prediction/trained_model.pkl','rb'))

input_data=(0,55,1,0,2,22.5,7.2,120)
#changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshaping the array
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print("The person is not diabetic")
else:
  print("The person is diabetic")
