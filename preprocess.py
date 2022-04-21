import os
import tensorflow as tf
import gdown
import pandas as pd
import streamlit as st
import tensorflow_hub as hub
import torch
import librosa
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
# Dataset for Question-Answering
from tensorflow.python.framework.test_ops import none

df = pd.read_json("http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json")
df = df.drop("version", axis=1)

# dataset for sentiment analysis
df_sent = pd.read_csv("sms_spam.csv")

# model for sentiment analysis
sent_model_url = "https://drive.google.com/uc?id=1--eULExMNhEKGiY4zZmdSB7dvMwh0nOX"
sent_model_path = './Best_model_emotion.h5'

# downloading model and tokenzier for speech to txt
speech_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


class preprocess:
    def __init__(self, load_data, store_pred, sent_anal_app, speechtotext):
        self.load_data = load_data
        self.store_pred = store_pred
        self.sent_anal_app = sent_anal_app
        self.speechtotext = speechtotext


def load_data():
    global df
    # required columns in our dataset
    columns = ["text", "question", "answer"]

    comp_list = []
    for index, row in df.iterrows():
        for i in range(len(row["data"]["questions"])):
            temp_list = []
            temp_list.append(row["data"]["story"])
            temp_list.append(row["data"]["questions"][i]["input_text"])
            temp_list.append(row["data"]["answers"][i]["input_text"])
            comp_list.append(temp_list)
    new_df = pd.DataFrame(comp_list, columns=columns)
    # saving the dataframe to csv file for further loading
    new_df.to_csv("CoQA_data.csv", index=False)
    return new_df


def store_pred(text, pred, val):
    return none


def load_model():
    global sent_model_path
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(sent_model_path):
        if os.path.getsize(sent_model_path) >= 2770000:
            st.warning("model already there haha")

    else:
        try:
            weights_warning = st.warning("Downloading %s..." % sent_model_path)
            gdown.download(sent_model_url, output=sent_model_path)
            st.warning('download finished')
        finally:
            st.write('thanks for the patience')

    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4")
    sent_model = tf.keras.models.load_model(sent_model_path, custom_objects={'KerasLayer': hub_layer})
    return sent_model


@st.cache(suppress_st_warning=True)
def sent_anal_app(texts):
    # loading the model
    sent_model = load_model()
    # predicting using pretrained model
    sent_prediction = sent_model.predict([texts])

    if round(sent_prediction[0][0]) == 1:
        val = "neg"
        st.title("This is negative sentence 😞")
    else:
        st.title("This is Positive Sentence 😉 ")
        val = "pos"

    store_pred(text=texts, pred=sent_prediction[0][0], val=val)
    return


def speechtotext(audiofile):
    file = audiofile
    speech, rate = librosa.load(file, sr=16000)  # sampling rate is 16000
    # converting vectors into pytorch tensors
    input_values = speech_tokenizer(speech, return_tensors="pt").input_values

    # storing logits (Non-normalized prediction)
    logits = speech_model(input_values).logits

    # store predicted ids
    predicted_ids = torch.argmax(logits, dim=-1)
    # decoding the array
    transcription = speech_tokenizer.decode(predicted_ids[0])
    return transcription
