import requests
import time 

"""
small script to test the api. Does data collection and QA data generation sequentially.
"""

if __name__ == "__main__": 

    data_collection_payload = {
        "reddit_scraper": {
            "search": {
                # --mandatory-- 
                "subreddits": ["MachineLearning"],
                "queries": ["AI", "GPT"],
                # --optional--
                "sort": "top",
                "time_filter": "all",
                "limit": 3,  # Keep small for testing
                "rate_limit_delay": 0.1,
                "filters": {
                    "n_comments": 3, # keep small for testing 
                    "min_post_score": 20, 
                    "min_comment_score": 10,
                    "min_char_length": 100
                }
            }
        },
        "semantic_filter": {
            # --mandatory--
            "topic": "artificial intelligence",
            # --optional--
            "n_keywords": 5,
            "similarity_threshold": 0.4,
            "min_token_len": 70
        }
    }

    # start data collection job 
    response = requests.post("http://127.0.0.1:8000/api/v1/data-collection", json=data_collection_payload)
    job_id = response.json().get("job_id")

    # wait until collection job is completed 
    status = None
    while not status == "completed": 
        time.sleep(10)
        status = requests.get(f"http://127.0.0.1:8000/jobs/{job_id}").json().get("status")

    data_generation_payload = {
        "job_id": response.json().get("job_id"), 
        "topic": "artificial intelligence",
        "question": "what is the best way to learn machine learning?", 
        "n_answers": 2
    }

    # start data generation job 
    response = requests.post("http://127.0.0.1:8000/api/v1/QA-generation", json=data_generation_payload)
    job_id = response.json().get("job_id")

    # wait until collection job is completed 
    status = None
    while not status == "completed": 
        time.sleep(10)
        status = requests.get(f"http://127.0.0.1:8000/jobs/{job_id}").json().get("status")

    response = requests.get(f"http://127.0.0.1:8000/jobs/{job_id}")

    # print response 
    print(response.json())