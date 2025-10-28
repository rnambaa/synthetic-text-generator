from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

class RedditScraperRequest(BaseModel):
    """Request model for the reddit scraper"""
    class SearchConfig(BaseModel):
        #  search fields
        subreddits: List[str]
        queries: List[str]
        # search params 
        sort: str = "top" 
        time_filter: str = "all" 
        limit: int = 20 
        rate_limit_delay: float = 0.1 

        # filters
        class FilterConfig(BaseModel): 
            n_comments: int = 20
            min_post_score: int = 20
            min_comment_score: int = 10
            min_char_length: int = 100
        
        filters: FilterConfig = FilterConfig()

    search: SearchConfig
    credentials: None = None # credentials are taken from configs/config.yaml


class SemanticFilterRequest(BaseModel): 
    """Request model for the semantic filter"""
    topic: str
    n_keywords: int = 5
    similarity_threshold: float = Field(0.4, gt=0, le=1)
    min_token_len: int = 70
    

class DataCollectionRequest(BaseModel): 
    """Request model for entire data collection pipeline"""
    reddit_scraper: RedditScraperRequest
    semantic_filter: SemanticFilterRequest


class QAGenerationRequest(BaseModel): 
    job_id: str # completed job id from data collection
    topic: str
    question: str 
    n_answers: int = 5