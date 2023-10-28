# Ask the PDF

Inside this directory you will find one Jupyter notebook:

- **notebook.ipynb**: A Jupyter notebook containing the complete project documentation, code, and instructions. It guides you through the process of utilizing LangChain and the OpenAI API to answer questions based on the content of a PDF document. The notebook covers library installations, PDF reading, text splitting, embeddings generation, and how to interact with the OpenAI model.

And one Python script:

- **app.py**: A Python script that serves as a Streamlit web application. It allows users to upload a PDF document, input their questions, and obtain answers based on the content of the PDF. The script leverages the knowledge base created from the PDF and the ChatGPT model from OpenAI to provide responses to user queries.

## Introduction

Ask the PDF is a Python project that allows you to extract information from a PDF document by asking questions. It leverages LangChain and the OpenAI API to provide answers based on the content of the PDF. This application can help you quickly find specific information within a large document without having to manually search for it.

**Key Features:**

- Extract text from a PDF document.
- Split the document into smaller chunks for efficient processing.
- Generate embeddings for the document chunks.
- Ask questions related to the document's content and receive answers.

## Getting Started

These instructions will help you set up and run the project on your local machine.

### Prerequisites

To run this project, you need to have **Python 3.x** and the following Python libraries installed:

- PyPDF2
- LangChain
- faiss-cpu
- openai
- python-dotenv
- sentence_transformers

You also need to obtain an API key from OpenAI and set it as an environment variable. Make sure to replace `'OPENAI_API_KEY'` with your actual API key.

### Installing

1. Clone the repository to your local machine.

2. Install the required Python libraries using `pip`.

3. Set your OpenAI API key as an environment variable or provide it inside the Streamlit app.

### Usage

1. Run the application with Streamlit:

```bash
streamlit run app.py
```

2. Enter your OpenAI API key and upload a PDF document.
3. Ask a question related to the content of the PDF.
4. The application will provide you with an answer based on the document's content.

## Built With

- [PyPDF2](https://pythonhosted.org/PyPDF2/) - Python library for PDF document manipulation.
- [LangChain](https://www.langchain.com/) - Language processing toolkit.
- [faiss-cpu](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors.
- [OpenAI](https://openai.com/) - For question answering using the GPT-3.5 model.
- [Streamlit](https://streamlit.io/) - For creating the web application interface.
- [Sentence Transformers](https://www.sbert.net/) - Library for generating embeddings for text.
