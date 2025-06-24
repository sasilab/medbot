from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone, FAISS

# Optional: Pinecone init
import os
import pinecone
from dotenv import load_dotenv

load_dotenv()

def create_embedder(model_name="all-MiniLM-L6-v2"):
    """
    Create a HuggingFace embedder.
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def create_chroma_vectorstore(lc_documents, model_name="all-MiniLM-L6-v2"):
    """
    Create a Chroma vectorstore from LangChain documents using HuggingFace embeddings.
    """
    embedder = create_embedder(model_name)
    vectorstore = Chroma.from_documents(lc_documents, embedding=embedder)
    return vectorstore

def create_faiss_vectorstore(lc_documents, model_name="all-MiniLM-L6-v2"):
    """
    Create a FAISS vectorstore from LangChain documents using HuggingFace embeddings.
    """
    embedder = create_embedder(model_name)
    vectorstore = FAISS.from_documents(lc_documents, embedding=embedder)
    return vectorstore

def create_pinecone_vectorstore(lc_documents, index_name, model_name="all-MiniLM-L6-v2"):
    """
    Create or connect to a Pinecone vectorstore.
    """
    # Get Pinecone API key and environment from env variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not set in .env")

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    # Create embedder
    embedder = create_embedder(model_name)

    # Connect to or create the index
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=384)  # 384 is dimension of MiniLM-L6-v2

    vectorstore = Pinecone.from_documents(lc_documents, embedding=embedder, index_name=index_name)
    return vectorstore
