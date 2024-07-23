# app.py
from flask import Flask, request, render_template
import pandas as pd
import joblib

import numpy as np
import pickle

# Load the trained model
model = joblib.load('news_classifier_model.pkl')
#filename = 'finalized_model.sav'
#model = pickle.load(open(filename, 'rb'))

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure you have downloaded the necessary NLTK data
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

final_data = pd.read_csv("Final_Data.csv")

print(final_data.head())


def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)

    # Convert to lowercase
    text = text.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = ' '.join([word for word in word_tokens if word not in stop_words])
    return filtered_text

def preprocess_text(text):
    # Tokenization
    word_tokens = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokens])
    return lemmatized_text

from sklearn.feature_extraction.text import TfidfVectorizer

content = final_data['preprocessed_text'].astype(str)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))

# Fit and transform the content to extract TF-IDF features
tfidf_features = tfidf_vectorizer.fit_transform(content)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# We have to start by splitting our dataset into training and test sets

X = final_data["preprocessed_text"] #Text data
y = final_data["category-label"] #Target variable (category)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Now we will vectorize the train and test data using TF-IDF

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training Logisitic Regression Model

#logModel = LogisticRegression(C=0.1, penalty='l2')
logModel = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
logModel.fit(X_train_tfidf,y_train)

result = {0: 'business', 1: 'education', 2: 'entertainment', 3: 'politics', 4: 'sports', 5: 'technology'}

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        news_text = clean_text(news)
        news_text = preprocess_text(news_text)
        news_tfidf = tfidf_vectorizer.transform([news_text])

        prediction = logModel.predict(news_tfidf)

        predicted_category = logModel.predict_proba(news_tfidf)

        top_two_indices = np.argsort(predicted_category)[0][-2::]
        print("Top prediction: {} with {}% accuracy".format(result.get(top_two_indices[1]).capitalize(),
                                                            round(predicted_category[0][top_two_indices[1]] * 100, 2)))
        print("Second prediction: {} with {}% accuracy".format(result.get(top_two_indices[0]).capitalize(),
                                                               round(predicted_category[0][top_two_indices[0]] * 100,
                                                                     2)))
        return render_template('result.html', prediction=result.get(int(prediction)), topPrediction = result.get(top_two_indices[1]).capitalize(), topAcc= round(predicted_category[0][top_two_indices[1]] * 100, 2),
                                secondPrediction=result.get(top_two_indices[0]).capitalize(), secAcc=round(predicted_category[0][top_two_indices[0]] * 100,
                                                                     2))

if __name__ == '__main__':
    app.run(debug=True)

