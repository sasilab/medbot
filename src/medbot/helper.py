
import os
import pandas as pd
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def load_diagnosis_documents(csv_path):
    """
    Load a diagnosis CSV file and convert each row into a text document for RAG.

    Args:
        csv_path (str): The file path to the diagnosis CSV.

    Returns:
        list: A list of document strings (1 per row).
    """
    df = pd.read_csv(csv_path)

    documents = []
    for _, row in df.iterrows():
        content = (
            f"PatientID: {row['PatientID']}. "
            f"Diagnosis: {row['Diagnosis']}. "
            f"State: {row['State']}. "
            f"Status: {row['Status']}."
        )
        documents.append(content)

    return documents


def load_medications_documents(csv_path):
    """
    Load a medications CSV file and convert each row into a text document for RAG.

    Args:
        csv_path (str): The file path to the medications CSV.

    Returns:
        list: A list of document strings (1 per row).
    """
    df = pd.read_csv(csv_path)

    documents = []
    for _, row in df.iterrows():
        content = (
            f"PatientID: {row['PatientID']}. "
            f"Date: {row['Date']}. "
            f"Medication: {row['Medication']}. "
        )
        documents.append(content)

    return documents



def create_langchain_documents(documents):
    """
    Convert a list of text chunks into LangChain Document objects.

    Args:
        documents (list): List of text strings.

    Returns:
        list: List of LangChain Document objects.
    """
    return [Document(page_content=text) for text in documents]


def create_chroma_vectorstore(lc_documents, model_name="all-MiniLM-L6-v2"):
    """
    Create a Chroma vectorstore from LangChain documents using HuggingFace embeddings.

    Args:
        lc_documents (list): List of LangChain Document objects.
        model_name (str): The HuggingFace model to use for embeddings.

    Returns:
        Chroma: A Chroma vectorstore instance.
    """
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma.from_documents(lc_documents, embedding=embedder)
    return vectorstore



def create_chat_openai_llm(model_name="gpt-3.5-turbo"):
    """
    Create a ChatOpenAI LLM instance using API key from .env.

    Args:
        model_name (str): OpenAI model name.

    Returns:
        ChatOpenAI: An LLM instance.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

    return ChatOpenAI(model_name=model_name, openai_api_key=api_key)

def create_retrieval_qa_chain(llm, retriever, chain_type="stuff", k=5):
    """
    Create a RetrievalQA chain.

    Args:
        llm (ChatOpenAI): The LLM instance.
        retriever: The vectorstore retriever.
        chain_type (str): Type of chain ("stuff", "map_reduce", etc.).
        k (int): Number of top documents to retrieve.

    Returns:
        RetrievalQA: A QA chain ready to invoke.
    """
    retriever.search_kwargs = {"k": k}


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
    )
    return qa_chain

def interactive_med_query(qa_chain):
    """
    Runs an interactive terminal loop for medical queries.

    Args:
        qa_chain: The LangChain QA chain to invoke queries.
    """
    print("📌 Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("Ask your med query: ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("👋 Exiting medical query assistant.")
            break

        try:
            result = qa_chain.invoke({"query": user_input})
            print(f"Sasky's answer: {result['result']}\n")
        except Exception as e:
            print(f"⚠ Error: {e}\n")
