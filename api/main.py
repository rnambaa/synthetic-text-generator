from pathlib import Path
import sys 
# add root dir 
root = str(Path(__file__).parent.parent)
sys.path.insert(0, root)

import yaml 
from fastapi import FastAPI
from api.routers import jobs, data_collection, generation

app = FastAPI()

# include routers
app.include_router(jobs.router)
app.include_router(data_collection.router)
app.include_router(generation.router)

@app.get("/health")
def health_check(): 
    return {"status": "ok"}