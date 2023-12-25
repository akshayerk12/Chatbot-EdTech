import streamlit as st

from langchain_helper import create_vector_db,get_qa_chain

st.title("CODE TECH ⚡💻⚡")
btn=st.button("Create Knowledgebase")
question=st.text_input("Please Enter Your Question")
if btn:
    pass
if question:
    chain=get_qa_chain()
    response=chain(question)
    st.header('Answer:')
    st.write(response["result"])