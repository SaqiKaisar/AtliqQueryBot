import streamlit as st
from langchain_helper import get_few_shot_query_chain, format_answer, run_query_on_db

st.title("AtliQ T Shirts: Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    llm, db, chain = get_few_shot_query_chain()
    result = run_query_on_db(question, db, chain, llm)

    st.header("Answer")
    st.markdown(f"```\n{result}\n```")
