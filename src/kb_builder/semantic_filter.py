import os
import time 
from pathlib import Path
from typing import List, Union, Dict, Any 
import numpy as np

from huggingface_hub import hf_hub_download
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.llms import GPT4All
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List 
import re 

from src.utils.data_handler import DataHandler
from src.kb_builder.base_filter import BaseFilter
from src.utils.utils import min_max



class SemanticFilter(BaseFilter):
    """
    Advanced semantic filtering system using topic similarity and keyword extraction.
    
    This filter uses a combination of a generative LLM and an embedding model to:
    1. Generate keywords most representative of a topic
    2. Extract keywords from a given document text using KeyBERT
    3. Compute semantic similarity between topic and document keywords
    4. Filter documents based on similarity thresholds and token length
                        
    Attributes:
        topic: Topic of interest for filtering.
        n_keywords: Number of keywords to extract from topic and document text.
        similarity_threshold: Minimum similarity score to retain documents [0, 1].
                            Higher values = more restrictive filtering (0.2 - 0.4 recommended range)
        min_token_len: Minimum token count for document retention.
                        Documents shorter than this are discarded.

        model_dir: Path object for model storage directory.
        gen_model: GPT4All instance wrapping Mistral 7B for keyword generation.
        embedding_model: SentenceTransformers model for text embedding.
        
    Note:
        First initialization will download ~5-6GB for Mistral model and
        additional space for embedding model. Subsequent runs use cached models.
    """

    def __init__(
        self,
        data: List[dict] = None,
        data_dir: str = None,
        data_filepath: str = None, 
        model_dir: str = None,
        semantic_filter_config: Dict[str, dict] = None,
    ):
        super().__init__(data_dir=data_dir, data_filepath=data_filepath, data=data)

        self.topic = semantic_filter_config["topic"]
        self.n_keywords = semantic_filter_config["n_keywords"]
        self.similarity_threshold = semantic_filter_config["similarity_threshold"]
        self.min_token_len = semantic_filter_config["min_token_len"]

        self.model_dir = self.root / Path(model_dir) if model_dir else None
        self.gen_model = None
        self.embedding_model = None
        self.load_models()


    def load_models(self):
        """
        Download and initialize the generative LLM and embedding models.
        
        Downloads and caches:
        - Mistral 7B Instruct (GGUF format, ~5-6GB) for keyword generation
        - all-MiniLM-L6-v2 SentenceTransformers model for embeddings        
        """
        # create model sub-dirs
        os.makedirs(self.model_dir / 'genLLM', exist_ok=True)
        os.makedirs(self.model_dir / 'embedding', exist_ok=True)

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
        )                                     # NOTE: GPT4All wraps prompts in its own hidden system message, to make it more instruction friendly instead of raw completion. 


        self.embedding_model = SentenceTransformer(
            model_name_or_path="all-MiniLM-L6-v2", # auto downloads model from the SentenceTransformers hub
            cache_folder=self.model_dir / 'embedding'
            )
        print("Models successfully loaded.")

    def generate_topic_keywords(self, topic: str, n_keywords: int, max_retries=5) -> List[str]:
        """
        Generate a set of n keywords representing the given topic using a GenLLM (mistral 7B).

        Returns:
            List[str]: List containing the original topic plus n_keywords generated terms.
                      All keywords are cleaned (lowercase, no punctuation) and validated.
    
        Example: 
            topic: "climate change" -> ["climate change", "environment", "carbon", "temperature"]
        """

        template = """
        Extract exactly {n_keywords} single-word (unigram) keywords that best represent the topic: "{topic}".

        Rules:
        - Output only the keywords, lowercase, separated by commas.
        - Do not include numbers, punctuation, or multi-word phrases.

        Example (topic: Artificial Intelligence):
        learning, algorithms, neural, computation, intelligence

        Now output:
        """

        prompt = PromptTemplate(template=template,input_variables=['topic', 'n_keywords'])
        output_parser = CommaSeparatedListOutputParser() # simply splits on ','

        chain = prompt | self.gen_model | output_parser

        for attempt in range(max_retries): 
            # llm forward pass + parsing 
            result = chain.invoke({"topic": topic, "n_keywords" : n_keywords})
            
            if len(result) != n_keywords:
                print(f"Attempt {attempt} failed, non-intended number of keywords, retrying...")
                print(result)
                continue 

            for i, kwd in enumerate(result): 
                if len(kwd) > 20: 
                    print(f"Attempt {attempt} failed, keyword '{kwd}' too long, retrying...")
                    break 
                # clean keyword
                kwd = re.sub(r'[^\w\s]', '', kwd) # remove anything that isn't word, char or whitespace
                kwd = kwd.strip()                 # remove trailing whitespaces 
                result[i] = kwd

            time.sleep(0.5)

            result.insert(0, topic) # add topic itself to keywords 
            return result 

                
        raise RuntimeError(f"Condition not met after {max_retries} attempts.")
    

    def score_topic_similarity(self, data: List[dict], topic: str, n_keywords=5) -> Dict[str, Any]: 
        """
        uses KeyBERT to extract keywords from text and then computes cosine similarity between topic keywords and text keywords.

        Returns:
            Dict[str, Dict]: Mapping of document IDs to similarity information:
                {
                    "doc_id": {
                        "score": float,      # Normalized similarity score [0, 1]
                        "keywords": List[str] # Extracted document keywords
                    }
                }
        """
        
        # embed topic keywords as single string sentence 
        topic_keywords = self.generate_topic_keywords(topic, n_keywords)
        topic_keywords_emb = self.embedding_model.encode(' '.join(topic_keywords))

        n_text_keywords = len(topic_keywords)
        keyword_model = KeyBERT(self.embedding_model)

        topic_similarity_scores = {}

        for i, entry in enumerate(data): 
            # extract keywords 
            text_keywords = keyword_model.extract_keywords(
                entry["text"],
                keyphrase_ngram_range=(1, 1),   # Unigrams only, bigrams -> (1, 2)
                stop_words='english',
                top_n=n_text_keywords           # Top N keywords to return
            )

            # embed topic keywords as single string sentence 
            kwd_string  = ' '.join(k[0] for k in text_keywords)
            text_keywords_emb = self.embedding_model.encode(kwd_string)

            # compute cosine similarity 
            score = cosine_similarity([topic_keywords_emb], [text_keywords_emb])

            topic_similarity_scores[entry["id"]] = {"score": score[0][0], "keywords": kwd_string.split(' ')}

        # min-max normalize scores 
        max_score = np.max(list(i["score"] for i in topic_similarity_scores.values()))
        min_score = np.min(list(i["score"] for i in topic_similarity_scores.values()))
        for key in topic_similarity_scores:
            topic_similarity_scores[key]["score"] = float(min_max(topic_similarity_scores[key]["score"], min_score, max_score))

        return topic_similarity_scores
    

    def filter(self, save_results=True):
        """
        Execute the complete semantic filtering pipeline on loaded data.

        Applies a multi-stage filtering process:
        1. Text cleaning and normalization
        2. Topic similarity scoring using keyword extraction and embeddings
        3. Token length filtering to remove very short documents
        4. Similarity threshold filtering to keep only topically relevant content
        5. Optional saving of filtered results with added metadata

        Args:
            save_results: If True, saves filtered data to 'filtered' subdirectory.

        Returns:
            List[dict]: Filtered documents with added fields:
                - 'text_keywords': List of extracted keywords from the document
                - 'topic_similarity_score': Normalized similarity score [0, 1]
        """
        
        # clean text data 
        self.data = self.clean_text()
        # get topic similarity scores
        topic_similarity_scores = self.score_topic_similarity(self.data, self.topic, self.n_keywords)

        filtered_data = []
        for i, entry in enumerate(self.data): 

            # remove small entries 
            if self.count_tokens(entry["text"]) < self.min_token_len: 
                continue
            
            # remove low topic similarity entries 
            if entry["id"] in topic_similarity_scores:
                if topic_similarity_scores[entry["id"]]["score"] >= self.similarity_threshold: 
                    # add the found keywords and relevance score to entries  
                    entry["text_keywords"] = topic_similarity_scores[entry["id"]]["keywords"]
                    entry["topic_similarity_score"] = topic_similarity_scores[entry["id"]]["score"]

                    filtered_data.append(entry)


        # save filtered data to directory 
        if save_results: 
            os.makedirs(self.data_dir / "filtered", exist_ok=True)
            self.save_jsonl(data=filtered_data, path=f"{self.data_dir / "filtered" / self.data_filename}") 
            print(f"Saving filtered results to: {self.data_dir / "filtered" / self.data_filename}")
        
        print("Semantic filtering completed!")
        return filtered_data
                

    
