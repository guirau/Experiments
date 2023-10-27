import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

st.header("Ask the PDF")
OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
pdf_obj = st.file_uploader("Load your document", type="pdf")

@st.cache_resource
def create_embeddings(pdf) -> FAISS:
    '''Extracts text from a PDF file, splits it into chunks, and creates embeddings for each chunk.

    Args:
        - pdf (Pdf): PDF object.

    Returns:
        - knowledge_base (FAISS): FAISS knowledge base containing embeddings for text chunks.
    '''

    # Load PDF file
    pdf_reader = PdfReader(pdf)

    # Store contents of PDF file in new variable 'text'
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Create text splitter object
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
        length_function = len
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    knowledge_base = FAISS.from_texts(chunks, embedding_model)

    return knowledge_base

# If PDF file is uploaded, create embeddings and ask the PDF
if pdf_obj:
    knowledge_base = create_embeddings(pdf_obj)
    user_question = st.text_input("Ask your question")

# If user question is provided, extract the context from the PDF and ask the question
    if user_question:
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        context = knowledge_base.similarity_search(user_question, top_k=3)
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type='stuff')
        answer = chain.run(input_documents=context, question=user_question)

        # Show answer
        st.write(answer)

        # Show total cost
        with get_openai_callback() as cb:
            answer = chain.run(input_documents=context, question=user_question)
            formatted_cb = " ".join(str(cb).split()[-4:])
            st.write(formatted_cb)