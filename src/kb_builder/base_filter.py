import os
import html 
import copy
from pathlib import Path
from typing import List, Union, Dict, Any 
from abc import abstractmethod
import tiktoken 

from typing import List 
import re 

from src.utils.data_handler import DataHandler


class BaseFilter(DataHandler):
    def __init__(self, data_dir: str = None, data_filepath: str = None, data: List[dict] = None):
        super().__init__(data_dir=data_dir, data_filepath=data_filepath, data=data)
        """
        Base class for filtering data entries based on various criteria.
        
        This abstract class provides common functionality for text cleaning and token counting,
        while requiring subclasses to implement their own filtering logic. Inherits from
        DataHandler to manage data loading and saving operations.

        TODO:
            - separate prompt template? 
        """

    @abstractmethod
    def filter(self):
        """Must be implemented in subclass."""
        raise NotImplementedError


    def clean_text(self): 
        """Clean text for embedding-friendly processing."""

        data = copy.deepcopy(self.data)

        for entry in data: 
            text = entry["text"]
            # Decode HTML entities like &#x200B; or &amp;
            text = html.unescape(text)
            # Remove URLs
            text = re.sub(r"http\S+|www\S+", "", text)
            # Remove Markdown links but keep link text
            text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
            # Remove leftover Markdown symbols for emphasis/bold/italics
            text = re.sub(r"[*_~`]", "", text)
            # Normalize quotes and dashes
            text = text.replace("“", '"').replace("”", '"')
            text = text.replace("‘", "'").replace("’", "'")
            text = text.replace("—", "-").replace("–", "-")
            # Remove invisible/zero-width characters
            text = re.sub(r"[\u200B\u200C\u200D\u200E\u200F\u2060\uFEFF]", "", text)
            # Remove excessive punctuation like !!! or ???
            text = re.sub(r"([!?.,])\1{2,}", r"\1", text)
            # Normalize whitespace but preserve paragraph breaks
            text = re.sub(r"[ \t]+", " ", text)  # spaces/tabs → single space
            text = re.sub(r"\n{3,}", "\n\n", text)  # 3+ newlines → 2
            text = text.strip()
            entry["text"] = text

        return data
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count the number of tokens in a text string using tiktoken encoding."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        return len(tokens)