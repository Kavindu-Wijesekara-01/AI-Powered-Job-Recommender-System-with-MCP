import sys
import traceback
import os
from dotenv import load_dotenv

load_dotenv()
print("Apify key:", os.getenv("apify_api_key"))

try:
    from src.job_api import get_linkedin_jobs
    jobs = get_linkedin_jobs("accountant", max_jobs=1)
    print("Jobs fetched:", len(jobs) if jobs else "None")
except Exception as e:
    traceback.print_exc()
