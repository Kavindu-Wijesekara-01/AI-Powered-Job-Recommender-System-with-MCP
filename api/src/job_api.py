import os
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()
APIFY_API_KEY = os.getenv("apify_api_key")

def get_linkedin_jobs(keyword, location="Sri Lanka", max_jobs=10):
    """
    Apify හි 'codingfrontend/linkedin-jobs-scraper' හරහා LinkedIn Jobs සෙවීම.
    දැන් job description එකත් ගන්නවා ML matching සඳහා.
    """
    if not APIFY_API_KEY:
        print("Error: Apify API key is missing!")
        return None

    client = ApifyClient(APIFY_API_KEY)
    actor_id = "codingfrontend/linkedin-jobs-scraper" 

    run_input = {
        "keywords": keyword,
        "location": location,
        "maxJobs": max_jobs, 
    }

    try:
        print(f"Apify හරහා සොයමින් පවතී: {keyword}")
        run = client.actor(actor_id).call(run_input=run_input)
        
        # --- අලුත් වෙනස (Error එක හදපු තැන) ---
        # 'run' එකේ වර්ගය අනුව හරියටම dataset ID එක ගන්නවා
        if isinstance(run, dict):
            dataset_id = run.get("defaultDatasetId")
        else:
            # Object එකක් නම් මෙහෙමයි ගන්නේ
            dataset_id = getattr(run, "defaultDatasetId", getattr(run, "default_dataset_id", None))
        # ----------------------------------------

        if not dataset_id:
            print("Error: Dataset ID එක සොයාගත නොහැක.")
            return None

        jobs_list = []
        for item in client.dataset(dataset_id).iterate_items():
            # Job description එකත් ගන්නවා (ML matching සඳහා)
            description = item.get("description", "") or item.get("jobDescription", "") or ""
            
            jobs_list.append({
                "title": item.get("jobTitle", "Unknown Title"),
                "company": item.get("company", "Unknown Company"),
                "location": item.get("location", location),
                "url": item.get("jobUrl", "#"),
                "description": description,  # ML matching සඳහා අලුතෙන් එකතු කරපු field
            })
            
        return jobs_list
        
    except Exception as e:
        print(f"Apify Error: {e}")
        return None


def get_linkedin_jobs_multi(keywords_list, location="Sri Lanka", max_jobs_per_keyword=5):
    """
    Multiple keywords වලට jobs search කරනවා.
    Args:
        keywords_list: list of keyword strings
        location: search location
        max_jobs_per_keyword: max jobs per keyword
    Returns: Combined list of unique jobs
    """
    all_jobs = []
    seen_urls = set()
    
    for keyword in keywords_list:
        keyword = keyword.strip()
        if not keyword:
            continue
            
        jobs = get_linkedin_jobs(keyword, location, max_jobs_per_keyword)
        
        if jobs:
            for job in jobs:
                # Duplicate jobs avoid කරනවා URL එකෙන්
                if job["url"] not in seen_urls:
                    seen_urls.add(job["url"])
                    all_jobs.append(job)
    
    return all_jobs