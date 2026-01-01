import streamlit as st
import pickle
import os

# File paths
MODEL_FILE = "expense_model.pkl"
VECTORIZER_FILE = "expense_vectorizer.pkl"

# App title (split to satisfy Flake8 line length)
st.title(
    "💰 AI-Powered "
    "Expense Tracker"
)

# Load model and vectorizer
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    with open(MODEL_FILE, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_FILE, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
else:
    st.error(
        "Model or vectorizer not found. "
        "Please run train_model.py first."
    )
    st.stop()

# User input
description = st.text_input(
    "Enter expense description:"
)

# Prediction
if st.button("Predict Expense"):
    if description.strip() == "":
        st.warning(
            "Please enter a description to predict expense."
        )
    else:
        desc_vector = vectorizer.transform([description])
        predicted_amount = model.predict(desc_vector)[0]
        st.success(
            f"Predicted Expense Amount: ₹{predicted_amount:.2f}"
        )
