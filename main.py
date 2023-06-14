from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

model = pickle.load(open('./model/model.pkl', 'rb'))
vectorization = pickle.load(open('./model/vector.pkl', 'rb'))

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not a Fake News"

@app.route('/predict', methods=['POST'])
def predict():
    news = request.json['news']
    result = manual_testing(news)
    
    return jsonify(result)

def manual_testing(news):
    vect = vectorization.transform([news]).toarray()
    pred_LR = model.predict(vect)

    return {"prediction": output_label(pred_LR[0])}

@app.route('/')
def index():

    return jsonify({"message": "Welcome to Fake News Detection API"})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
