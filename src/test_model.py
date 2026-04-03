import joblib

#Load the dataset
model= joblib.load("models/category_model.pkl")

#Text Input
text= ["I did not receive my refund and no one is responding"]

prediction= model.predict(text)
print("prediction:", prediction[0])