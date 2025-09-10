import yaml
import json
from abc import ABC, abstractmethod
from src.utils.data_handler import DataHandler

class BaseScraper(DataHandler):
    def __init__(self, config_path: str = None, data_dir: str = None):
        super().__init__(data_dir=data_dir)
        """Base class for all scrapers."""
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    @abstractmethod
    def scrape(self):
        """Must be implemented in subclass."""
        raise NotImplementedError

