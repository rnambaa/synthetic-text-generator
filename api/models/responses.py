from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

class JobIDResponse(BaseModel): 
    job_id: str

class JobResponse(BaseModel): 
    type: str
    status: str
    current_stage: Optional[str] = None 
    results: Optional[Any] = None 
    document_info: Optional[Dict[str, int]] = None 
    start_timestamp: str
    end_timestamp: Optional[str] = None 
    error_message: Optional[str] = None 





