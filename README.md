# Chat with PDF

This is a Streamlit application that allows users to interactively chat with a PDF document using natural language processing techniques.

## Features

- **PDF Upload:** Users can upload a PDF document.
- **Text Extraction:** The application extracts text from the uploaded PDF document.
- **Question-Answering:** Users can ask questions about the PDF document, and the application provides answers based on the content of the document.
- **Precomputed Data Storage:** To optimize performance, the application stores precomputed data in `.pkl` files for future use.

## How to Use

1. Clone the repository to your local machine:
```
git clone https://github.com/usman-29/Chat-PDF.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the application:
```
streamlit run main.py
```

4. Upload a PDF document using the file uploader.
5. Ask questions about the PDF document in the provided text input field.
6. The application will display answers based on the content of the PDF document.

## Technologies Used

- [Streamlit](https://streamlit.io/): For building the web application.
- [PyPDF2](https://pythonhosted.org/PyPDF2/): For reading PDF files.
- [langchain](https://github.com/) : [langchain](https://github.com/langchain/): For text processing and question-answering capabilities.
- [dotenv](https://github.com/theskumar/python-dotenv): For managing environment variables.
- [FAISS](https://github.com/facebookresearch/faiss): For efficient similarity search.
- [OpenAI GPT-3.5](https://openai.com/): For generating responses in the chat interface.
