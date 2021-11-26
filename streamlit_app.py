# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:14:36 2021

@author: sngh9
"""

import streamlit as st
#import cv2 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import joblib
joblib_file = "SVM_model.pkl"
joblib_file2 = "mlp.pkl"
joblib_file3 = 'eclf1.pkl' 
def barometro(a):
    if a == 0:
        return 'No llena'
    else:
        return 'Llena'
st.title('Barómetro basado en modelos de inteligencia artificial')
uploaded_file = st.file_uploader("Escogé una imagen")
Modelos =st.sidebar.radio("Modelo para saber el estado de la llanta",('SVM', 'Red Neuronal', 'Bosque Aleatorio'))
#bytes_data = uploaded_file.getvalue()

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
     # To convert to a string based IO:
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)
     # To read file as string:
    #string_data = stringio.read()
    #st.write(string_data)
    
     # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)
    #image = Image.open(uploaded_file)
    #imagen = cv2.imread(uploaded_file)
    #plt.imshow(imagen)
    image = Image.open(uploaded_file)
    st.image(image, caption='Input', use_column_width=True)
    img_array = np.array(image)
    print(img_array.shape)
    imagen = img_array[ : , : , 0 ].flatten( )
    SVM_model = joblib.load(joblib_file)
    MLP_model = joblib.load(joblib_file2)
    RF_model = joblib.load(joblib_file3)
    
    if Modelos == 'SVM':
        a = SVM_model.predict(imagen.reshape(1, -1))
    elif Modelos == 'Red Neuronal':
        a = MLP_model.predict(imagen.reshape(1, -1))
    elif Modelos == 'Bosque Aleatorio':
        a = RF_model.predict(imagen.reshape(1, -1))
        

    st.text('Su llanta esta : '+ barometro(a[0]))
  
    #cv2.imread(img_array)