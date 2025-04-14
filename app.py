from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")

if openai_key and langchain_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true" # Langsmith tracking: for tracing and capturing all the monitoring results
else:
    raise ValueError("Keys are not present")


## For collecting question history in a list
if "question_history" not in st.session_state:
    st.session_state.question_history = []


## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a very helpful assistance and you should respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

## streamlit framework

st.title("LLM powered Chat Assistance Dashboard ")
st.markdown("Welcome to the intelligent assistant dashboard! Get the experience in interacting with the OpenAI's GPT-4o model  "
 " And feel free to explore :) ")
input_text=st.text_input("Search the topic that interests you!")



## openAI LLM
llm=ChatOpenAI(model='gpt-4o')
output_parser=StrOutputParser() #responsible for capturing the output
chain=prompt|llm|output_parser


# Question History Button
if st.button("Question History"):
    if st.session_state.question_history:
        for i, question in enumerate(st.session_state.question_history):
            st.write(i+1, question)
    else:
        st.write("Ah! You havent asked any questions so far!! :( ")

elif input_text:
    st.session_state.question_history.append(input_text)
    st.write(chain.invoke({"question": input_text}))

# Summarize the session chat
if st.button(" Summarize session chat"):
    if st.session_state.question_history:
        all_questions = "\n".join([f"- {q}" for q in st.session_state.question_history])
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a smart and helpful assistant. You will be given a list of user questions from a session."),
                ("user", "Here are the questions asked by the user in this session:\n{questions}\n\n" 
                         "Please summarize the session in a friendly, detailed, and informative manner. Group similar questions and highlight key themes if possible.")
            ]
        )
        summary_chain = summary_prompt | llm | output_parser
        summary = summary_chain.invoke({
            "questions": all_questions
        })
        st.subheader("Summarize Session Chats:")
        st.write(summary)

    else:
        st.info("Ah! There is no questions asked yet")