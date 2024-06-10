import pandas as pd
from headlamphealth.ingest import InMemoryTextIngestor
from pandas import DataFrame
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from headlamphealth.embedder import ChromaEmbedder
from langchain_core.embeddings import Embeddings
import os
from typing import Optional
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()


class RecommendCards:
    def __init__(self, cards_csv: str, embedding_function: Optional[Embeddings] = None):
        """Recommend cards to the user
        based on anxiety levels

        Parameters
        ----------
        cards_csv : str
            File which contains cards
        embedding_function: Embeddings
            This is the embedding function that converts text
            to semantic vectors
        """
        self.cards_csv = cards_csv

        self.top_k_retrieval = 2

        # Read the information stored in csvs
        self.cards_df = self._read_df()

        # By default this text splitter is used
        # TODO: Remove this dependency and accept any textsplitter
        # to embed the cards

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200, add_start_index=True
        )

        if embedding_function is None:
            embedding_function = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

        self.embedding_function = embedding_function

        self.embedder = ChromaEmbedder(
            embedding_function=self.embedding_function,
            embedding_store_directory=f"{os.environ['STORES_DIR']}/anxiety_cards",
        )

    def _read_df(self) -> DataFrame:
        """Return the dataframe that contains the card information

        Returns
        -------
        DataFrame
            Contains information about the cards
        """
        return pd.read_csv(self.cards_csv)

    def _embed_cards(self) -> VectorStore:
        """This embeds all the cards into the vector store

        Returns
        -------
        VectorStore
            This is the vector store where all the embeddings
            are stored. This is created only one
        """
        data = self.cards_df["tip_information"].tolist()
        metadata = self.cards_df["tip_heading"].tolist()
        metadata = list(
            map(lambda heading_information: {"heading": heading_information}, metadata)
        )

        self.ingestor = InMemoryTextIngestor(
            texts=data, metadatas=metadata, splitter=self.splitter
        )
        documents = self.ingestor.ingest()
        store = self.embedder.store_embeddings(documents)
        return store

    def recommend_cards(self, query: str):
        """Recommend cards using the query

        Parameters
        ----------
        query : str
            The query is from the user's journal or
            we can use information that he has already stored
            from the previous encounters
        """

        store = self._embed_cards()
        retriever = store.as_retriever()
        docs = retriever.invoke(query, search_kwargs={"k": self.top_k_retrieval})
        return docs


if __name__ == "__main__":
    pass
