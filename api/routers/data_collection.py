from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.models.requests import DataCollectionRequest
from api.models.responses import JobIDResponse
from api.services.job_manager import job_manager
from api.config import config
from src.scrapers.reddit_scraper import RedditScraper
from src.kb_builder.semantic_filter import SemanticFilter
from src.kb_builder.chunker import Chunker
from datetime import datetime 


router = APIRouter(prefix='/api/v1', tags=["data-collection"])

def run_collection_pipeline(job_id: str, request: DataCollectionRequest): 
    print("Data collection started")
    try: 
        # add reddit api credentials 
        scraper_config = request.reddit_scraper.model_dump()
        if not scraper_config["credentials"]: 
            scraper_config["credentials"] = config["reddit_scraper"]["credentials"]
        
        # run data collection pipeline & updates 
        job_manager.update_job(job_id, {"current_stage": "scraping... (stage 1/3)"})
        scraper = RedditScraper(scraper_config=scraper_config)
        raw_data = scraper.scrape(save_results=False)

        job_manager.update_job(job_id, {"current_stage": "filtering... (stage 2/3)"})
        filter = SemanticFilter(
            data=raw_data, 
            model_dir=config["paths"]["model_dir"],
            semantic_filter_config=request.semantic_filter.model_dump()
        )
        filtered_data = filter.filter(save_results=False)

        job_manager.update_job(job_id, {"current_stage": "chunking... (stage 3/3)"})
        chunker = Chunker(data=filtered_data)
        chunks = chunker.chunk(save_results=False) 

        # job is completed 
        job_manager.update_job(job_id, {
            "status": "completed",
            "current_stage": "done", 
            "results": chunks,
            "document_info": {
                "documents_scraped": len(raw_data),
                "documents_after_filtering": len(filtered_data), 
                "chunks_created": len(chunks)
            },
            "end_timestamp": str(datetime.now())
        })
        print("job complete.")
    
    except Exception as e: 
        job_manager.update_job(job_id, {
            "status": "failed", 
            "error_message": str(e), 
            "end_timestamp": str(datetime.now())            
        })


@router.post("/data-collection", response_model=JobIDResponse)
def data_collection(request: DataCollectionRequest, background_tasks: BackgroundTasks):
    # create new job & update job status 
    job_id = job_manager.create_job(job_type="data-collection")
    job_manager.update_job(job_id, {
        "status": "running",
        "current_stage": "job started",
    })
    # start background task 
    background_tasks.add_task(run_collection_pipeline, job_id, request)

    return {"job_id": job_id}




