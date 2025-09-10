from pathlib import Path
from typing import List, Union
import json 
import os 

from src.utils.data_handler import DataHandler

class Chunker(DataHandler): 
    def __init__(self, data_dir: str = None, data_filepath: str = None, data: List[dict] = None):
        super().__init__(data_dir=data_dir, data_filepath=data_filepath, data=data)
        """
        Base class for filters. Handles loading data.
        TODO: 
            - develop more complex chunking system. 
            - config? 
        """
    
    def chunk(self, save_results=True): 
        """
        Extermely simple chunking method, temporary.
        """
        chunks = []
        
        for entry in self.data:
            chunk = {
                "id": f"chunk_{entry['id']}_0",
                "source_id": entry['id'],
                "source_type": entry['type'],
                "subreddit": entry.get('subreddit', ''),
                "text": entry.get('text', ''),
                "chunk_index": 0,
                "chunk_count": 1,
                "merge_ids": [],
                "created_utc": entry.get('created_utc'),
                "score": entry.get('score', 0),
                "thread_id": entry.get('thread_id'),
                "parent_id": entry.get('parent_id')
            }
            chunks.append(chunk)

        # save chunks to directory 
        if save_results: 
            os.makedirs(self.data_dir / "chunks", exist_ok=True)
            self.save_jsonl(data=chunks, path=f"{self.data_dir / "chunks" / self.data_filename}") 
            print(f"Saving filtered results to: {self.data_dir / "chunks" / self.data_filename}")

        return chunks
