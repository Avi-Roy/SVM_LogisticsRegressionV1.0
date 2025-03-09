import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import joblib
import pickle
from sklearn.datasets import load_iris
import streamlit as st

def load_model(model_name,user_data):
    model_name =model_name+".pkl"
    model = joblib.load(model_name)
    predit = model.predict(user_data)
    if predit ==1 :
        return 'versicolor or class 1'
    elif predit == 2:
        return 'virginica or Class 2'
    else:
        return 'setosa or Clas 0'
    

def main():
    st.title("IRIS DATA CLASSIFICATIONS")
    st.write("Select Your Model Class , either Binary or Multi class classifications")

    model_name = st.selectbox("Select Model ", ['LogisticsBinary', 'LogisticsMultiClass', 'svm_binary' , 'svm_multiClassPredit'])
    sepal_length = st.number_input("Enter sepal length (cm)", min_value= 0 , max_value=10000, value=54)
    sepal_width = st.number_input("Enter sepal width (cm) ", min_value= 0.00 , max_value=1000.00, value=5.9)
    petal_length = st.number_input("Enter petal length (cm)", min_value= 0.0 , max_value=1000.0, value=6.9)
    petal_width = st.number_input("Enter petal width (cm)", min_value= 0.0 , max_value=1000.0, value=1.9)

    if st.button("Find Class of IRIS Flower"):
        user_data = {
            'sepal length (cm)' : sepal_length,
            'sepal width (cm)' : sepal_width,
            'petal length (cm)' : petal_length,
            'petal width (cm)' : petal_width
        }

        user_data = np.array(list(dict.values(user_data))).reshape(1,-1)     
        
        predit = load_model(model_name,user_data)
        st.success(f"Your IRIS class  is  {predit} ")
        if predit == 'versicolor or class 1':
            st.image('versicolor.jpeg')
        elif predit == 'virginica or Class 2':
            st.image('virginica.jpeg')
        else :
            st.image('setosa.jpeg')
              

if __name__ == "__main__":
    main()
