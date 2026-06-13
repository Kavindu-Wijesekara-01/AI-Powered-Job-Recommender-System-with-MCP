import os
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()
APIFY_API_KEY = os.getenv("apify_api_key")

def get_linkedin_jobs(keyword):
    """
    Apify හි 'codingfrontend/linkedin-jobs-scraper' හරහා LinkedIn Jobs සෙවීම.
    """
    if not APIFY_API_KEY:
        print("Error: Apify API key is missing!")
        return None

    client = ApifyClient(APIFY_API_KEY)
    actor_id = "codingfrontend/linkedin-jobs-scraper" 

    run_input = {
        "keywords": keyword,
        "location": "Sri Lanka",
        "maxJobs": 10, 
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
            jobs_list.append({
                "title": item.get("jobTitle", "Unknown Title"),
                "company": item.get("company", "Unknown Company"),
                "location": item.get("location", "Sri Lanka"),
                "url": item.get("jobUrl", "#")
            })
            
        return jobs_list
        
    except Exception as e:
        print(f"Apify Error: {e}")
        return None