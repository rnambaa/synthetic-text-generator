import os 
from pathlib import Path
from typing import List
from huggingface_hub import hf_hub_download

from src.RAG.retriever import Retriever
from src.utils.data_handler import DataHandler

from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QA_generator(DataHandler): 
    """
    A Question-Answer generator that uses RAG to create contextual responses based on retrieved documents.
    
    This class a gen LLM  model (Mistral-7B) to produce first-person, authentic answers to questions 
    using the retrieved context as inspiration.
    
    Attributes:
        retriever: An instance of the retriever class for document retrieval
        gen_model: The generative language model (GPT4All with Mistral-7B)
        model_dir: Path to the directory containing model files
    """
    def __init__(
            self, 
            retriever, 
            data_dir: str = None, 
            data_filepath: str = None, 
            data: List[dict] = None, 
            model_dir: str = None,
        ): 
        super().__init__(data_dir=data_dir, data_filepath=data_filepath, data=data)
        
        self.retriever = retriever(
            data_dir=data_dir, 
            data_filepath=data_filepath, 
            data=data, 
            model_dir=model_dir 
        )

        self.gen_model = None 
        self.model_dir = self.root / Path(model_dir) if model_dir else None
        self.load_models()

    def load_models(self):
        """
        Downloads and initializes the generative LLM.
        Downloads and caches:
        - Mistral 7B Instruct (GGUF format, ~5-6GB) for keyword generation
        """
        # create model sub-dirs
        os.makedirs(self.model_dir / 'genLLM', exist_ok=True)

        gen_model_path = hf_hub_download(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", 
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf", # RAM: ~ 5-6GB
        cache_dir=self.model_dir / 'genLLM'
    )   
        self.gen_model = GPT4All(
            model=gen_model_path,
            verbose=False,  
            n_threads=os.cpu_count() or 4,    # Adjust for your CPU cores
            temp=0.3,                         # Temperature (creativity)
            top_p=0.95,                       # Nucleus sampling
            top_k=40,                         # Token sampling pool
            repeat_penalty=1.1,               # Penalize repeats
            max_tokens=200,                   # Max tokens to generate
            allow_download=False,             # Don't try to download model
        )

    def query_retrieve_texts(self, query: str, top_k: int) -> List[str]: 
        """
        Retrieve relevant text documents for a given query.
        
        Args:
            query: The search query to find relevant documents
            top_k: Number of top documents to retrieve
            
        Returns:
            List[str]: List of text content from the most relevant documents
        """
        documents = self.retriever.retrieve(query, top_k=top_k, filter_criteria=None)
        return [doc["document"]["text"].strip() for doc in documents]

    
    def generate(self, query: str, topic: str = "", n_documents=1) -> List[str]: 

        """
        Generate first-person answers to a question using retrieved context.
        
        The functiion returns one answer per retrieved document, 
        allowing for multiple perspectives on the same question. 
        The query used for retrieval is the same as the question itself.
        
        Args:
            query: The question to answer
            topic: The topic/domain for the questionnaire context
            n_documents: Number of documents to retrieve and use. 
        
        Returns:
            List[str]: List of generated answers, one for each retrieved document.
        """

        # retrieve texts for RAG 
        texts = self.query_retrieve_texts(query, top_k=n_documents)
        
        template = """
        You are writing as if you are personally answering a {topic} questionnaire. 
        Base your response on the context provided below, but do not copy it word for word. 
        Write in the first person, keep it natural and authentic, and limit your answer to 100 words.

        Context (opinions from others, for inspiration):
        {context}

        Question:
        {query}

        Answer (first-person, max 100 words):
        """

        prompt = PromptTemplate(template=template, input_variables=["topic", "context", "query"])
        context = "\n\n".join([t for t in texts])
        parser = StrOutputParser()

        chain = prompt | self.gen_model | parser 

        # NOTE: trying with multiple contexts
        results = []
        for text in texts:
            res = chain.invoke({"topic": topic,"context": text, "query": query})
            results.append(res)

        return results 

        # TODO: output parsing? 
