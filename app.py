import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)




# Load the trained model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Streamlit application code
st.title(":blue[Major Project]")

input_sms = st.text_area(":red[Enter the message]")

if st.button(':green[Submit]'):

    # Preprocess the input
    input_sms = transform_text(input_sms) # Convert to lowercase

    # Transform the input using the TF-IDF vectorizer
    transformed_sms = tfidf_vectorizer.transform([input_sms])

    # Predict
    result = model.predict(transformed_sms)[0]

    # Display the result
    if result == 1:
        st.header(":red[Not Safe]")
    else:
        st.header(":green[Safe]")
