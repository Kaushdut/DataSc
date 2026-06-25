import pandas as pd
import streamlit as st

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama


# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="DataFrame ChatBot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DataFrame ChatBot")


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = None


# --------------------------------------------------
# Read File Function
# --------------------------------------------------
def read_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    return pd.read_excel(file)


# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    try:
        df = read_data(uploaded_file)

        st.session_state.df = df

        st.success("✅ File uploaded successfully")

        st.subheader("Data Preview")
        st.dataframe(df.head())

        with st.expander("Dataset Information"):
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write(df.columns.tolist())

    except Exception as e:
        st.error(f"Error loading file: {e}")


# --------------------------------------------------
# Display Chat History
# --------------------------------------------------
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# User Input
# --------------------------------------------------
user_prompt = st.chat_input(
    "Ask questions about your dataframe..."
)


# --------------------------------------------------
# Process Query
# --------------------------------------------------
if user_prompt:

    if st.session_state.df is None:
        st.warning("Please upload a CSV or Excel file first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    try:

        # --------------------------------------------------
        # LLM
        # --------------------------------------------------
        llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0
        )

        # --------------------------------------------------
        # Pandas Agent
        # --------------------------------------------------
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state.df,
            verbose=False,
            allow_dangerous_code=True
        )

        enhanced_prompt = f"""
You are working with a pandas dataframe.

Question:
{user_prompt}

Instructions:
- Execute pandas operations when needed.
- Return actual results from the dataframe.
- If rows are found, display the rows.
- Do not only explain what code should be used.
- Give concise answers.
"""

        with st.spinner("Thinking..."):

            response = agent.invoke(
                {"input": enhanced_prompt}
            )

            answer = response.get(
                "output",
                "No answer returned."
            )

        # --------------------------------------------------
        # Display Assistant Response
        # --------------------------------------------------
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    except Exception as e:

        with st.chat_message("assistant"):
            st.error(f"Error: {str(e)}")