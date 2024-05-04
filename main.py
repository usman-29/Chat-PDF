import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
import pickle
from dotenv import load_dotenv
import os

with st. sidebar:
    st.title("CHAT PDF")
    st .markdown(
        '''
        #About
        This app is an LLM-powered chatbot built using:
        - [Streamlit] (https://streamlit.io/)
        - [LangChain] (https://python.langchain.com/)
        - [OpenAI] (https://platform.openai.com/docs/models) LLM model
        ''')
    add_vertical_space(5)
    st.write("Chat with Pdf files")


def main():
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload your pdf file", type="pdf")

    if pdf:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = faiss.from_texts(chunks, embeddings=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, )


if __name__ == "__main__":
    load_dotenv()
    main()
