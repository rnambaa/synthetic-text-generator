import os 
from pathlib import Path
from typing import List
from huggingface_hub import hf_hub_download
from src.RAG.retriever import Retriever

from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QA_generator(): 
    def __init__(self, retriever, model_dir: str = None): 
        self.retriever = retriever # TODO figure out it I instantiate outside or inside the class 
        self.model_dir = Path(model_dir)

        self.gen_model = None 
        self.load_models()

    def load_models(self):
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
            max_tokens=512,                   # Max tokens to generate
            allow_download=False,             # Don't try to download model
        )

    def query_retrieve_texts(self, query) -> List[str]: 
        """return relevant texts as list"""
        documents = self.retriever.retrieve(query, top_k=10, filter_criteria=None)
        return [doc["document"]["text"] for doc in documents]

    
    def generate(self, query: str): 
        # retrieve texts for RAG 
        texts = self.query_retrieve_texts(query)
        
        # ---- Define your custom RAG prompt ----
        # template = """
        # You are an expert at answering DEI questions. Use the following context to inspire your answer. 

        # Context:
        # {context}

        # Question:
        # {query}

        # Answer:
        # """

        template = """
        You are writing as if you are personally answering a Diversity, Equity, and Inclusion (DEI) questionnaire. 
        Base your response on the context provided below, but do not copy it word for word. 
        Write in the first person, keep it natural and authentic, and limit your answer to 100 words.

        Context (opinions from others, for inspiration):
        {context}

        Question:
        {query}

        Answer (first-person, max 100 words):
        """


        prompt = PromptTemplate(template=template, input_variables=["context", "query"])
        context = "\n\n".join([t for t in texts])
        parser = StrOutputParser()


        chain = prompt | self.gen_model | parser 
        # chain.invoke({"context": context, "query": query})


        # NOTE: trying with multiple contexts
        results = []
        for text in texts:
            res = chain.invoke({"context": text, "query": query})
            results.append(res)

        return results 
    


        # TODO: eventually some kind of output parsing, cleaning and formatting 
