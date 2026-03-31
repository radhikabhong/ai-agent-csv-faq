import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Load API key from .env

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# App UI

st.set_page_config(page_title="AI Agent - CSV FAQ")
st.title("AI Agent - CSV FAQ")

st.write("Upload CSV files and ask questions based ONLY on the data.")

# File upload

uploaded_files = st.file_uploader(
"Upload CSV files",
type=["csv"],
accept_multiple_files=True
)

dataframes = []

# Show previews

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        dataframes.append(df)
        st.subheader(f"Preview: {file.name}")
        st.dataframe(df.head())

# User question

question = st.text_input("Ask a question about your data")

# Answer generation

if st.button("Get Answer") and question and dataframes:

    
    system_prompt = """
    You are a smart data assistant.
    You ONLY answer from the provided CSV data.
    Do NOT use general knowledge.
    If the answer is not found, say:
    'I could not find this information in the uploaded files.'
    """

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=api_key
        )

        agent = create_pandas_dataframe_agent(
            llm,
            dataframes,
            verbose=True,
            agent_type="openai-functions",
            allow_dangerous_code=True
        )

        final_query = system_prompt + "\n\nQuestion: " + question

        response = agent.invoke({"input": final_query})

        st.success(response["output"])

    except Exception as e:
        st.error(f"Error: {e}")
    
