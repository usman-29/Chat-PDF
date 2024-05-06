import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import pickle
from dotenv import load_dotenv
import os


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
            VectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, )

        query = st.text_input("Ask question about your PDF file: ")

        if query:
            docs = VectorStore.similarity_search(query=query)
            llm = ChatOpenAI(model='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == "__main__":
    load_dotenv()
    main()
