import yaml
import json
from abc import ABC, abstractmethod
from typing import Dict
from src.utils.data_handler import DataHandler

class BaseScraper(DataHandler):
    def __init__(self, scraper_config: Dict[str, dict] = None, data_dir: str = None):
        super().__init__(data_dir=data_dir)
        """Base class for all scrapers."""
    
        self.scraper_config = scraper_config

    @abstractmethod
    def scrape(self):
        """Must be implemented in subclass."""
        raise NotImplementedError

