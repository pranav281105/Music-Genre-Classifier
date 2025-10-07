import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')

# Load model & vectorizer
lr_model = joblib.load('genre_classifier_lr.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

st.title("🎵 Music Genre Classification")
st.write("Paste your song lyrics below to predict the genre!")

user_lyrics = st.text_area("Enter Lyrics Here", height=200)

if st.button("Predict Genre"):
    if user_lyrics.strip() == "":
        st.warning("Please enter some lyrics!")
    else:
        clean_lyrics = preprocess(user_lyrics)
        features = tfidf_vectorizer.transform([clean_lyrics])
        prediction = lr_model.predict(features)[0]
        st.success(f"Predicted Genre: **{prediction}**")
