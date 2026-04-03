# Importing Important Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline


# Setting Text Cleaning Function

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Training the Model
def train_category_model():
    # Loading the datset as pandas DataFrame
    df = pd.read_csv("data/complaints.csv")

    # Selecting relevant columns
    df= df[["narrative", "product_5"]]

    # Dropping nulls in the dataset
    df = df.dropna()

    # Renaming columns
    df.columns = ["text", "label"]

    # Taking Sample of the dataset to train our model
    df = df.sample(80000, random_state=42).reset_index(drop= True)

    # Cleaning the text
    df["text"] = df["text"].apply(clean_text)

    #Defining X and y for the dataset
    X= df["text"]
    y= df["label"]

    # Trainig-Testing Split of the Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Creating the Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("clf", SGDClassifier(max_iter=200, class_weight= "balanced"))
    ])


    # Training the Model
    pipeline.fit(X_train, y_train)


    # Evaluation
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division= 0))

    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

    # Saving the Model
    import joblib

    joblib.dump(pipeline, "models/category_model.pkl")
    print("Model Saved Successfully!")

if __name__=="__main__":
    train_category_model()