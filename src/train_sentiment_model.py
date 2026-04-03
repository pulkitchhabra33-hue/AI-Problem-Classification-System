import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import joblib

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Setting Text Cleaning Function

def clean_text(text):
    text= str(text).lower()
    text= re.sub(r"http\S+", "", text)
    text= re.sub(r"@\w+", "", text)
    text= re.sub(r"[^a-zA-Z\s]", "", text)
    text= re.sub(r"\s+", " ", text).strip()
    return text 

# Training the Model
def train_sentiment_model():
    # Loading the datset as pandas DataFrame
    cols = ["target", "id", "date", "flag", "user", "text"] #defining the column names for the dataset
    df= pd.read_csv(r"C:\Users\pulki\AIP\data\training.1600000.processed.noemoticon.csv", encoding="latin-1", names= cols)
    df.head()

    # Selecting relevant columns
    df= df[["text", "target"]]

    # Convert Label (0-> Negative, 4-> Positive)
    df["target"]= df["target"].map({0: "Negative", 4: "Positive"})

    # Dropping nulls in the dataset
    df = df.dropna()

    # Taking Sample of the dataset to train our model
    df = df.sample(100000, random_state=42).reset_index(drop= True)

    # Cleaning the text
    df["text"] = df["text"].apply(clean_text)

    #Defining X and y for the dataset
    X= df["text"]
    y= df["target"]

    # Trainig-Testing Split of the Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Creating the Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words="english",
            min_df= 5,
            max_df= 0.9
        )),
        ("classifier", LogisticRegression(max_iter= 200))
    ])

    # Training the Model
    pipeline.fit(X_train, y_train)

    # Predicting on Test Set
    y_pred = pipeline.predict(X_test)

    # Evaluating the Model
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Saving model
    joblib.dump(pipeline, "models/sentiment_model.pkl")
    print("\nModel Saved Successfully!")

if  __name__=="__main__":
    train_sentiment_model()