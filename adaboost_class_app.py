import streamlit as st
import joblib

model = joblib.load("ada_classifier.pkl")

st.title("AdaBoost Classification")

sl = st.slider("Sepal Length",4.0,8.0,5.1)
sw = st.slider("Sepal Width",2.0,5.0,3.5)
pl = st.slider("Petal Length",1.0,7.0,1.4)
pw = st.slider("Petal Width",0.1,3.0,0.2)

if st.button("Predict"):
    pred = model.predict([[sl,sw,pl,pw]])
    st.success(pred[0])