import streamlit as st
import base64
import time
import threading
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from streamlit_modal import Modal

ps = PorterStemmer()

confirmationEdit = Modal("Result", key= "popUp_edit")


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


st.title(":blue[Text Phishing Detection System using Machine Learning Algorithm]")

input_sms = st.text_area(":red[Enter the message]")

 # Preprocess the input
input_sms = transform_text(input_sms) # Convert to lowercase

    # Transform the input using the TF-IDF vectorizer
transformed_sms = tfidf_vectorizer.transform([input_sms])

    # Predict
result = model.predict(transformed_sms)[0]

if st.button(':green[Submit]'):
    # 4. Display
    if result == 1:
        res = ":red[UnSafe]"
    else:
        res = ":green[Safe]"

    with st.status("Predicting .....", expanded = True) as status:
        st.write("Analyzing Text .....")
        time.sleep(0.3)
        st.write("Predicting Class")
        time.sleep(0.3)
        st.write("Finalizing results")
        time.sleep(0.3)
        status.update(label = f"{res}", state="complete", expanded=False)

    modal = Modal(key="Demo Key",title="Result")
  
    with modal.container():
        if res == ":red[UnSafe]":
            new_title = '<p style="font-family:sans-serif; text-align: center; color:Red; font-size: 42px;"><b>UnSafe</b></p>'
            st.markdown(new_title, unsafe_allow_html=True)
        else:
            new_title = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 42px;"><b>Safe</b></p>'
            st.markdown(new_title, unsafe_allow_html=True)
            