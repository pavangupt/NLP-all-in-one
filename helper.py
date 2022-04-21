import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

df = pd.read_csv("CoQA_data.csv")
random_num = np.random.randint(0, len(df))
question = df["question"][random_num]
text = df["text"][random_num]


class helper:
    def __init__(self, question_answer,question_answer_condition):
        self.question_answer = question_answer
        self.question_answer_condition = question_answer_condition



def question_answer(text: object, question: object) -> object:
    # tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)

    # string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segment IDs
    # first occurences of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    # number of tokens in segment A (question)
    num_seg_a = sep_idx + 1
    # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a

    # list of 0s and 1s for segment embeddings
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    # model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # tokens with the highest start and end scores
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

    return answer.capitalize()



def question_answer_condition(text, question):
    while True:
        question_answer(text, question)
        flag = True
        flag_N = False
        while flag:
            response = st.text_input('Do you want to ask another question based on this text (Y/N)?', 'Y')
            st.write("You have selected:-", response)
            if response[0] == "Y":
                question = st.text_area("Enter the next Questions")
                flag = False
            elif response[0] == "N":
                print("Thank you for your time!")
                flag = False
                flag_N = True
        if flag_N == True:
            break
