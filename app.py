import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

@st.cache_resource
def load_model():
    try:
        model = joblib.load('iadc_code_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None
    return model, vectorizer

model, vectorizer = load_model()

if model is None or vectorizer is None:
    st.stop()

def main():
    st.title('IADC Code Predictor')
    st.write("This app predicts the IADC code based on the provided descriptions.")

    description = st.text_input('Enter the description:')

    if st.button('Predict'):
        if description:
            text_vectorized = vectorizer.transform([description])
            prediction = model.predict(text_vectorized)
            st.write('Predicted IADC Code:', prediction[0])
        else:
            st.write('Please enter a description.')

if __name__ == '__main__':
    main()
