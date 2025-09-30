from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import json 
import os 

import chromadb
from chromadb.errors import NotFoundError
from chromadb import EmbeddingFunction, Documents
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from src.utils.data_handler import DataHandler
from src.utils.utils import format_collection_name



class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        """Tiny wrapper for chroma collections to use sentence-transformers models."""
        self.model = model

    def __call__(self, intput: Documents):
        return self.model.encode(intput, convert_to_numpy=True).tolist()


class Retriever(DataHandler): 
    def __init__(
            self, 
            data_dir: str = None, 
            data_filepath: str = None, 
            data: List[dict] = None, 
            model_dir: str = None
        ):
        super().__init__(data_dir=data_dir, data_filepath=data_filepath, data=data)
        """
        A document retrieval system using ChromaDB for vector similarity search.

        This class handles the creation and management of vector collections, document
        ingestion from JSONL files, and semantic search capabilities. 

        Attributes:
            client: ChromaDB client for vector database operations.
            collection_name: Name of the vector collection (derived from data filename).
            embedding_model: SentenceTransformers model for generating embeddings.
            embedding_function: ChromaDB-compatible wrapper for the embedding model.
            collection: ChromaDB collection instance for storing and querying vectors.
            model_dir: Directory for storing model files.
        """

        self.model_dir = self.root / Path(model_dir) if model_dir else None
        self.client = chromadb.Client()
        # self.client = chromadb.PersistentClient(path="./chroma_db")  # NOTE: for Persistent client
        self.collection_name = self.data_filename if self.data_filename else "temp_collection"

        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-MiniLM-L6-v2", # use same model as the semantic filtering step 
            cache_folder=self.model_dir / 'embedding'
            )

        self.embedding_function = CustomEmbeddingFunction(model=self.embedding_model) # embedding wrapper for chroma 
        
        # Create or get the collection
        try:
            self.collection = self.client.get_collection(
                name=format_collection_name(self.collection_name), # used to check if collection already exists 
                embedding_function=self.embedding_function
            )
            print(f"Using existing collection: {self.collection_name}")

        except NotFoundError:
            self.collection = self.client.create_collection(
                name=format_collection_name(self.collection_name),
                embedding_function=self.embedding_function
            )
            print(f"Created new collection: {self.collection_name}")
            # populate collection 
            self._load_vector_store()


    def _load_vector_store(self, batch_size: int = 100) -> None:
        """
        Load documents from self.data into ChromaDB collection in batches. 
        Each document is expected to have 'id', 'text', and optional metadata fields.
        """

        documents = []
        ids = []
        metadatas = []
        
        for i, chunk in enumerate(self.data):
            batch_count = 0
            # Extract text content
            text = chunk.get("text", "")
            # Extract metadata (excluding text field to avoid duplication)
            metadata = {k: ("None" if v is None else v) for k, v in chunk.items() if k != "text"}
            del metadata['merge_ids'] # removed the 'merge_ids' field
            # Use the chunk ID as provided in the data
            chunk_id = chunk.get("id")
            
            if not chunk_id or not text:
                print(f"Warning: Skipping entry at line {i+1} due to missing id or text")
                continue
            
            documents.append(text)
            ids.append(chunk_id)
            metadatas.append(metadata)

            # add batch to create embeddings 
            if len(documents) >= batch_size:

                self.collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas
                )
                batch_count += 1
                documents = []
                ids = []
                metadatas = []
        
        # Add remaining documents
        if documents:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            batch_count += 1
            
        print(f"Successfully loaded {self.collection.count()} documents into Chroma")


    def retrieve(self, query: str, top_k: int = 2, filter_criteria: Optional[Dict] = None) -> List[dict]:
        """
        Retrieves top_k most relevant documents for a given query.
        
        Performs vector similarity search against the stored document collection,
        optionally filtering results based on metadata criteria. Results are ranked
        by semantic similarity using the configured embedding model.

        Args:
            query: The search query
            top_k: Number of top results to return
            filter_criteria: Dictionary of metadata filters (e.g., {"subreddit": "changemyview"})
            
        Returns:
            List of retrieved documents with similarity scores & metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_criteria
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "document": {
                    "text": results["documents"][0][i],
                    **results["metadatas"][0][i]  # Include all metadata
                },
                "similarity": results["distances"][0][i] if "distances" in results else None,
                "id": results["ids"][0][i]
            })
            
        return formatted_results


    
