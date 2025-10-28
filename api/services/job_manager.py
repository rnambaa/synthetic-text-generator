# imports 
from typing import Dict, Any
import uuid 
from datetime import datetime

# function to generate jobid 
def generate_job_id(prefix): 
    return f"{prefix}-{str(uuid.uuid1())}"

class JobManager(): 

    def __init__(self):
        self.jobs = {}


    # useful methods
    def create_job(self, job_type: str): 
        """Creates a new job """
        # generate a job id 
        job_id = generate_job_id(job_type)
        
        # add template dict to self.jobs
        self.jobs[job_id] = {
            "type": job_type,
            "status": "started",
            "current_stage": None,
            "results": None,
            "document_info": None, 
            "start_timestamp": str(datetime.now()),
            "end_timestamp": None,
            "error_message": None 
        }
        return job_id


    def get_job(self, job_id: str):
        """Returns job information from job ID"""
        return self.jobs.get(job_id) 
    
    # def get_job_status(self, job_id: str): 
    #     """Returns job status from job ID"""
    #     return self.jobs.get(job_id).get("status")

    def update_job(self, job_id: str, updates: Dict[str, Any]):
        """Updates a job with new information"""
        if not job_id in self.jobs: 
            raise ValueError(f"Job ID {job_id} does not exist.")
        else: 
            self.jobs[job_id].update(updates)
        

# create shared instance that can be imported (singleton pattern)
job_manager = JobManager()


# figure out type hinting 