from pathlib import Path
from typing import List, Union
import json 
from abc import ABC, abstractmethod
import os 

class DataHandler(ABC): 
    def __init__(self, data_dir: str = None, data_filepath: str = None, data: List[dict] = None):
        """
        Handler class responsible for loading and saving data.

        Args:
            data_dir: Directory containing data files.
            data_filepath: Path to specific data file.
            data: List[dict] of data entries to process directly (instead of reading from storage).
        """

        self.data_dir = Path(data_dir) if data_dir else None
        self.data_filepath = Path(data_filepath) if data_filepath else None
        self.data_filename = self.data_filepath.name if self.data_filepath else None

        self.data = data
        
        if not data and self.data_filepath:
            self.load_data_from_file(self.data_filepath)

    def load_data_from_file(self, file_path: Union[str, Path]):
        """Loads data as a list of dicts from a single jsonl file specified by full path."""
        file_path = Path(file_path)
        self.data = []
        if file_path.suffix.lower() != '.jsonl':
            raise ValueError(f"File must be a JSONL file, got {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"Data loaded from {file_path}")

    def load_data_from_dir(self, dir_path: Union[str, Path]):
        """Loads data as a list of dicts from jsonl files in a directory."""
        dir_path = Path(dir_path)
        self.data = []
        for file in dir_path.glob("*.jsonl"):
            with open(file, "r") as f:
                for line in f:
                    self.data.append(json.loads(line))

        print(f"Data loaded from {dir_path}")

    def save_jsonl(self, data, path):
        """Saves data as a list of dicts to a single jsonl."""
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")