import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Function to load data
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip')
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to train model
@st.cache_data
def train_model(data):
    # Column names based on the provided structure
    comments_column = 'comments'
    iadc_code_column = 'Sub code'

    # Preprocess the data
    data = data.dropna(subset=[comments_column, iadc_code_column])
    comments = data[comments_column].astype(str).values
    iadc_codes = data[iadc_code_column].astype(str).values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(comments, iadc_codes, test_size=0.2, random_state=42)

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Define the model
    model = LogisticRegression()

    # Define the grid of hyperparameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l2', 'none'],
        'max_iter': [100, 200, 300, 500]
    }

    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train_tfidf, y_train)

    # Save the best model and vectorizer for future use
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

    return best_model, vectorizer

# Streamlit App
def main():
    st.title("IADC Code Prediction")
    st.write("Enter a comment to predict the IADC code")

    # Load the dataset
    file_path = r'C:/Users/KIIT/Desktop/AI/DPRIiadc).csv'  # Use raw string to avoid unicode escape issues
    data = load_data(file_path)
    if data is not None:
        # Train the model
        best_model, vectorizer = train_model(data)
        
        # Text input
        user_input = st.text_area("Comment")

        if st.button("Predict"):
            if user_input:
                user_input_tfidf = vectorizer.transform([user_input])
                prediction = best_model.predict(user_input_tfidf)
                st.write(f"Predicted IADC Code: {prediction[0]}")
            else:
                st.write("Please enter a comment")
    else:
        st.write("Error loading dataset")

if __name__ == '__main__':
    main()
