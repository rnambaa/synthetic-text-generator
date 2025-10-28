from api.models.requests import DataCollectionRequest, QAGenerationRequest
from api.models.responses import JobResponse
from api.services.job_manager import job_manager
from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job: 
        raise HTTPException(status_code=404, detail="Job ID not found")
    else: 
        return job 
    

