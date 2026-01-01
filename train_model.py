import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import pickle

CSV_FILE = "expenses.csv"
MODEL_FILE = "expense_model.pkl"
VECTORIZER_FILE = "expense_vectorizer.pkl"

needed = ["description", "amount"]

if not os.path.exists(CSV_FILE):
    print("CSV missing. Add expenses.csv to the project folder.")
    exit()

df = pd.read_csv(CSV_FILE)

if df.empty:
    print("CSV is empty. Please add some rows.")
    exit()

for col in needed:
    if col not in df.columns:
        print(f"Missing column: {col}. Please add it.")
        exit()

descriptions = df["description"].astype(str)
amounts = df["amount"].astype(float)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(descriptions)
y = amounts.values

model = LinearRegression()
model.fit(X, y)

with open(MODEL_FILE, "wb") as mf:
    pickle.dump(model, mf)

with open(VECTORIZER_FILE, "wb") as vf:
    pickle.dump(vectorizer, vf)

print("Model and vectorizer saved.")
