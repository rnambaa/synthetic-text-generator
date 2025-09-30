import os 
from datetime import datetime
from pathlib import Path
import praw
import time
from typing import Dict 
from src.scrapers.base_scraper import BaseScraper

class RedditScraper(BaseScraper):
    def __init__(self, scraper_config: Dict[str, dict] = None, data_dir: str = None):
        super().__init__(scraper_config=scraper_config, data_dir=data_dir)
        """
        A Reddit scraper that collects posts and comments from specified subreddits.
        
        Inherits from BaseScraper and uses the PRAW library to interact with Reddit's API.
        Searches subreddits for posts matching configured queries, filters results by score
        and content length, and collects top comments for each post.
        
        Attributes:
            data_dir (Path): Directory path for storing scraped data
            reddit (praw.Reddit): PRAW Reddit instance for API interactions
        """

        self.data_dir = Path(data_dir)

        creds = self.scraper_config["credentials"]
        self.reddit = praw.Reddit(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            user_agent=creds["user_agent"]
        )

    def scrape(self, save_results=True):
        """
        Scrape Reddit posts and comments based on configured search parameters.
        
        Searches subreddits for posts matching queries defined in config, filters by 
        score and length, and collects top comments. Applies rate limiting between requests.
        
        Returns:
            list: List of dictionaries containing post and comment data. Each dictionary
                contains the following keys:
                - id (str): Unique identifier (reddit_<post_id> or reddit_comment_<comment_id>)
                - type (str): Content type ("post" or "comment")
                - source (str): Data source ("reddit")
                - subreddit (str): Subreddit name
                - title (str): Post title (posts only)
                - text (str): Post selftext or comment body
                - author (str): Author username
                - url (str): Post URL
                - score (int): Reddit score (upvotes - downvotes)
                - created_utc (float): Unix timestamp of creation
                - thread_id (str): ID of parent thread
                - parent_id (str): ID of parent post/comment (None for posts)
        """
         
        results = []
        cfg = self.scraper_config

        print("Collecting data...")

        for sub in cfg["search"]["subreddits"]:
            subreddit = self.reddit.subreddit(sub)

            for query in cfg["search"]["queries"]:
                submissions = subreddit.search(
                    query=query,
                    sort=cfg["search"]["sort"],
                    time_filter=cfg["search"]["time_filter"],
                    limit=cfg["search"]["limit"]
                )

                for post in submissions:
                    if post.score < cfg["search"]["filters"]["min_post_score"] or len(post.selftext) < cfg["search"]["filters"]["min_char_length"]:
                        continue

                    post_data = {
                        "id": f"reddit_{post.id}",
                        "type": "post",
                        "source": "reddit",
                        "subreddit": sub,
                        "title": post.title,
                        "text": post.selftext,
                        "author": str(post.author),
                        "url": post.url,
                        "score": post.score,
                        "created_utc": post.created_utc,
                        "thread_id": f"reddit_{post.id}",
                        "parent_id": None
                    }
                    results.append(post_data)

                    post.comment_sort = "top"
                    post.comments.replace_more(limit=0)
                    for comment in post.comments[:cfg["search"]["filters"]["n_comments"]]:
                        if comment.score < cfg["search"]["filters"]["min_comment_score"] or len(comment.body) < cfg["search"]["filters"]["min_char_length"]:
                            continue

                        comment_data = {
                            "id": f"reddit_comment_{comment.id}",
                            "type": "comment",
                            "source": "reddit",
                            "subreddit": sub,
                            "text": comment.body,
                            "author": str(comment.author),
                            "url": post.url,
                            "score": comment.score,
                            "created_utc": comment.created_utc,
                            "thread_id": f"reddit_{post.id}",
                            "parent_id": f"reddit_{post.id}"
                        }
                        results.append(comment_data)

                    time.sleep(cfg["search"]["rate_limit_delay"])  # Optional delay for rate limiting

        print("Removing duplicate posts...")

        unique_results = {}
        for r in results: 
            if r['id'] not in unique_results: 
                unique_results[r['id']] = r
        results = list(unique_results.values())

        # save raw results to directory
        if save_results: 
            os.makedirs(self.data_dir / "raw", exist_ok=True)
            dtime = datetime.now().strftime("%Y-%m-%d %H:%M")
            filename = f"reddit_data-{dtime}.jsonl"
            self.save_jsonl(data=results, path=f"{self.data_dir / "raw" / filename}")
            print(f"Saving results to: {self.data_dir / "raw" / filename}")

        print("Scraping completed!")
        return results
