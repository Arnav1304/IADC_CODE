import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the trained model and TF-IDF vectorizer
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('iadc_code_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Define the Streamlit app
def main():
    st.title('IADC Code Predictor')
    
    # Text input for description
    description = st.text_input('Enter the description:')
    
    if st.button('Predict'):
        if description:
            # Preprocess the input text
            text = description.lower()
            
            # Vectorize the text
            text_vectorized = vectorizer.transform([text])
            
            # Make prediction
            prediction = model.predict(text_vectorized)
            
            # Display prediction
            st.write('Predicted IADC Code:', prediction[0])
        else:
            st.write('Please enter a description.')

if __name__ == '__main__':
    main()
