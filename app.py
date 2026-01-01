import streamlit as st
import pickle

# Load model
with open("expense_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open("expense_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

st.title("AI-Powered Expense Tracker")

st.write(
    "Enter your expense details below, and the AI model will predict the "
    "category for you."
)

user_input = st.text_area(
    "Enter expense details (e.g., Rent, Groceries, Travel, etc.):"
)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        st.success(f"Predicted category: {prediction[0]}")
