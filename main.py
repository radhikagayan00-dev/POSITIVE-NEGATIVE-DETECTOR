import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import nltk


nltk.download("stopwords")
nltk.download("punkt")


loaded_model = joblib.load('sentiment_model.joblib')


stop_words = set(stopwords.words("english"))


ps = PorterStemmer()


vectorizer = joblib.load('vectorizer.joblib')

def preprocess_text(text):
    
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    
    text = re.sub(r"\@\w+", "", text)
    
    
    text = re.sub(r"[^\w\s]", "", text)
    
    
    words = word_tokenize(text.lower())
    
    
    words = [ps.stem(w) for w in words if w.isalpha() and w not in stop_words]
    
    return " ".join(words)


def predict_sentiment(text):
    processed_text = preprocess_text(text)
    input_vectorized = vectorizer.transform([processed_text])
    prediction = loaded_model.predict(input_vectorized)
    return "Positive" if prediction[0] == 4 else "Negative"


while True:
    user_input = input("Enter a sentence (type 'exit' to quit): ").strip()

    if user_input.lower() == 'exit':
        break

    sentiment = predict_sentiment(user_input)
    print(f"Sentiment: {sentiment}")
