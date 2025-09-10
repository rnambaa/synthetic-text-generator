# add root dir 
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.scrapers.reddit_scraper import RedditScraper
from src.kb_builder.semantic_filter import SemanticFilter
from src.kb_builder.chunker import Chunker
from src.RAG.retriever import Retriever
from src.RAG.QA_format_generator import QA_generator




if __name__ == "__main__":

    # refactor tests

    # # scrape reddit & save raw data
    # scraper = RedditScraper(
    #     config_path="/Users/ryonamba/Documents/xN/text-gen-agent/src/scrapers/scraper_config.yaml",
    #     data_dir="/Users/ryonamba/Documents/xN/text-gen-agent/data"
    #     )
    # raw = scraper.scrape(save_results=True)

    # semantic_filter = SemanticFilter(
    #     # data_dir="/Users/ryonamba/Documents/xN/text-gen-agent/data",
    #     data_filepath = "/Users/ryonamba/Documents/xN/text-gen-agent/data/raw/reddit_data-2025-08-19 10:39.jsonl",
    #     model_dir="/Users/ryonamba/Documents/xN/text-gen-agent/models"
    # )
    # semantic_filter.filter()

    # chunker = Chunker(
    #     data_dir="/Users/ryonamba/Documents/xN/text-gen-agent/data",
    #     data_filepath = "/Users/ryonamba/Documents/xN/text-gen-agent/data/filtered/reddit_data-2025-08-19 10:39.jsonl"
    # )

    # chunker.chunk()

    retriever = Retriever(
        # data_dir="/Users/ryonamba/Documents/xN/text-gen-agent/data",
        data_filepath = "/Users/ryonamba/Documents/xN/text-gen-agent/data/chunks/reddit_data-2025-08-19 10:39.jsonl",
        model_dir="/Users/ryonamba/Documents/xN/text-gen-agent/models"
    )


    # query = """What is your personal association with 'diversity'?"""
    # query = """In your opinion, what is effective in DEI communication and implementation at this company, and what is not effective?"""
    # query = """If you could make a wish to optimize the DEI strategy at this company, which area would you choose to increase support in implementing the DEI strategy?"""
    query = """For you to participate actively in a DEI Workshop, the session must provide or avoid what exactly?"""

    generator = QA_generator(
        retriever=retriever,
        model_dir="/Users/ryonamba/Documents/xN/text-gen-agent/models"
    )

    res = generator.generate(query)

    print(res)





