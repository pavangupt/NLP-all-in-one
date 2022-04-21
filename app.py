import streamlit as st
from keras.models import load_model
import helper
import preprocess
from PIL import Image
qa_image = Image.open('question_answering.jpg')
sa_img = Image.open('sentiment_analysis.jpg')
st_img = Image.open('speechtotext.png')


# sent_analysis_model= load_model("model_sent_analysis.h5")

def main():
    option = st.sidebar.selectbox("Which NLP Task You Want To Perform!",
                                  ("Question_Answering", "Sentiment_Analysis", "Speech-To-Text"))
    st.sidebar.write("You selected:--", option)
    # new_dfs = preprocess.load_data()
    if option == "Question_Answering":
        st.sidebar.image(qa_image, caption= "Question Answering")
        st.subheader("Question Answering Model")
        text = st.text_area("Enter Your Text/Paragraph")
        question = st.text_area("Enter Your Questions")
        if st.button("Predict"):
            answer = helper.question_answer(text, question)
            st.success(answer)

    if option == "Sentiment_Analysis":
        st.sidebar.image(sa_img, caption="Sentiment Analysis")
        st.title("Sentiment Analysis Model (Emotion Detection!)")

        texts = st.text_area("Enter the text", """This is the sample text! """)
        if st.button("Predict"):
            answer = preprocess.sent_anal_app(texts)
            st.success(answer)

    if option == "Speech-To-Text":
        st.sidebar.image(st_img, caption="Speech To Text Conversion")
        st.title('Speech to Text Conversion')
        audio_file = st.file_uploader("Please upload your Audio file (wav. format)!")

        st.audio(audio_file)
        st.write("Sample Audio for the reference!")
        if st.button('Convert'):
            if audio_file is not None:
                audio_text = preprocess.speechtotext(audio_file)
                st.success(audio_text)


if __name__ == "__main__":
    main()
