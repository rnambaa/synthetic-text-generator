from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.models.requests import QAGenerationRequest
from api.models.responses import JobIDResponse
from api.services.job_manager import job_manager
from api.config import config
from src.RAG.retriever import Retriever
from src.RAG.QA_generator import QAgenerator   
from datetime import datetime 


router = APIRouter(prefix='/api/v1', tags=["answer-generation"])

def run_QA_generation(job_id: str, request: QAGenerationRequest): 
    
    print("QA generation started")
    try: 
        # parse request 
        chunks = job_manager.get_job(request.job_id).get("results")
        generator = QAgenerator(
            retriever=Retriever, 
            data=chunks,
            model_dir=config["paths"]["model_dir"]
        )

        job_manager.update_job(job_id, {"current_stage": "generating..."})
        answers = generator.generate(
            query=request.question, 
            topic=request.topic, 
            n_documents=request.n_answers
        )

        job_manager.update_job(job_id, {
            "status": "completed",
            "current_stage": "completed", 
            "results": answers,
            "end_timestamp": str(datetime.now())
        })
        print("job complete.")

    except Exception as e: 
        job_manager.update_job(job_id, {
            "status": "failed", 
            "error_message": str(e), 
            "end_timestamp": str(datetime.now())
        })


@router.post("/QA-generation", response_model=JobIDResponse)
def QA_generation(request: QAGenerationRequest, background_tasks: BackgroundTasks):
    # check if previous data collection job exists and is completed
    if not job_manager.get_job(request.job_id):
        raise HTTPException(status_code=404, detail="Job ID not found") 

    if job_manager.get_job(request.job_id).get("status") == "completed": 
        # create new job & update status 
        gen_job_id = job_manager.create_job("QA-generation")
        job_manager.update_job(gen_job_id, {
            "status": "running",
            "current_stage": "job started", 
        }) 
        # start background task
        background_tasks.add_task(run_QA_generation, gen_job_id, request)
    
        return {"job_id": gen_job_id}

    else: 
        raise HTTPException(status_code=400, detail="Data collection job not completed.")



