from langchain_core.embeddings import Embeddings
from headlamphealth.embedder import ChromaEmbedder
from typing import Optional
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from headlamphealth.ingest import InMemoryTextIngestor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from headlamphealth.chat_agent import AnxietyChain
from rich.console import Console


console = Console()

load_dotenv()


def load_anxiety_chain(
    empathy_csv: str,
    embedding_store_directory: str,
    embedding_function: Optional[Embeddings] = None,
    openai_model: Optional[str] = "gpt-4o",
):
    empathy_df = pd.read_csv(empathy_csv)

    llm = ChatOpenAI(model=openai_model, api_key=os.environ["OPENAI_API_KEY"])

    if embedding_function is None:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Prepare the data for post and the response
    post = empathy_df["post"]
    response = empathy_df["response"]
    metadatas = map(lambda response: {"response": response}, response)
    metadatas = list(metadatas)

    # This will give you the documents for embedding in the store
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    ingestor = InMemoryTextIngestor(
        texts=post, metadatas=metadatas, splitter=text_splitter
    )

    documents = ingestor.ingest()

    embedder = ChromaEmbedder(
        embedding_function=embedding_function,
        embedding_store_directory=embedding_store_directory,
    )

    with console.status("Storing the embeddings"):
        store = embedder.store_embeddings(documents)

    chat = AnxietyChain(llm=llm, vector_store=store)

    return chat
