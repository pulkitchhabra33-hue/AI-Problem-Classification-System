from unittest import result

from src.train_category_model import clean_text
import joblib

def predict_category(text):
    #Loading the trained category model for prediction
    category_model= joblib.load("models/category_model.pkl")

    #Predicting the category of the input text using the loaded model
    return category_model.predict([text])[0]

def predict_sentiment(text):
    #Loading the trained sentiment model for prediction
    sentiment_model= joblib.load("models/sentiment_model.pkl")

    #Predicting the sentiment of the input text using the loaded model
    return sentiment_model.predict([text])[0]


Category_map= {
    "credit reporting": "account_issue",
    "debt collection": "payment_issue",
    "credit card services": "payment_issue",
    "bank accounts and services": "account_issue",
    "loans": "payment_issue"
    }

def map_category(raw_category):
    raw_category= raw_category.strip().lower()
    if raw_category in Category_map:
        return Category_map[raw_category]
    return "other_issue"


def urgency_level(sentiment, category):
    sentiment= sentiment.strip().lower()

    if sentiment== "negative" and category in ["account_issue", "payment_issue"]:
        return "High"
    elif sentiment== "negative":
        return "Medium"
    else:
        return "Low"


risk_words= ["scam", "fraud", "worst", "never buy", "twitter"]

def detect_risk(text, sentiment, category):
    text= text.strip().lower()
    
    if any(word in text for word in risk_words):
        return "High Risk"
    elif sentiment== "negative" and category== "payment_issue":
        return "High Risk"
    elif sentiment.lower()== "negative":
        return "Medium Risk"
    else:
        return "Low Risk"

def generate_reply(category):
    if category == "payment_issue":
        return "We apologize. Your payment/refund issue is being reviewed."
    elif category == "account_issue":
        return "We're sorry for the trouble. Please try resetting your account or contact support for assistance."
    else:
        return "Thank you for your message. Our team will respond shortly."



def analyze_complaint(text):
    
    text= clean_text(text)

    raw_category= predict_category(text)
    category= map_category(raw_category)

    sentiment= predict_sentiment(text)
    urgency= urgency_level(sentiment, category)

    risk= detect_risk(text, sentiment, category)
    reply= generate_reply(category)

    return {
        "category" : category,
        "sentiment" : sentiment, 
        "urgency" : urgency,
        "risk" : risk,
        "reply" : reply
    }
    

# if __name__== "__main__":
#     text= "I didn't receive my refund, worst service ever"
#     result = analyze_complaint(text)
#     for key, value in result.items():
#         print(f"{key}: {value}")
