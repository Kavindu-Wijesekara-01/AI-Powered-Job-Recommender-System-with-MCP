import asyncio
from src.helper import *
from main import search_jobs, SearchJobsRequest

async def run_test():
    req = SearchJobsRequest(
        keywords=["accountant"],
        resume_text="I am an accountant with 5 years of experience.",
        cv_details={
            "primary_field": "Finance",
            "llm_resume_skills": ["Auditing", "Taxation", "Excel"]
        }
    )
    
    try:
        res = await search_jobs(req)
        print("Success:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(run_test())
