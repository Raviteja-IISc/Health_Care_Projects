import streamlit as st
import pickle
import numpy as np
#import joblib


def load_model():
    with open('Heart_Disease_Prediction/saved_steps.pkl','rb') as file:
        data=pickle.load(file)
        #data=joblib.load('Streamlit/saved_steps.pkl')
    return data


data =load_model()

clf_log_reg=data["model"]


def show_predict_page():
    st.title("Heart Diesease predictor")

    st.write("Key-in the patient's information.")

    age=st.slider("age",10,90,1)
    sex=st.selectbox("Sex",(0,1))
    cp=st.slider("cp",0,3,1)
    trestbps=st.slider("trestbps",100,160,1)
    chol=st.slider("chol",100,400,1)
    fbs=st.selectbox("fbs",(0,1))
    restecg=st.selectbox("restecg",(0,1))
    thalach=st.slider("thalach",100,400,1)
    exang=st.selectbox("exang",(0,1))
    oldpeak=st.slider("oldpeak",0.1,4.0,0.1)
    slope=st.selectbox("slope",(0,1,2,3))
    ca=st.selectbox("ca",(0,1,2))
    thal=st.selectbox("thal",(1,2,3))
    
    ok=st.button("Calculate the risk")

    if ok:
        X=np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal]])
        target=clf_log_reg.predict(X)
        if target==1:
            st.subheader("Heart issue")
            
        else:
            st.subheader("No problem")        


        
    











        
