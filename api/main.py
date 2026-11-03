from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import io

# Import our ML/LLM logic
from src.helper import (
    extract_text_from_pdf,
    clean_resume_text,
    get_llm_response,
    analyze_cv_srilanka_context,
)
from src.ml_matcher import JobMatcher
from src.job_api import get_linkedin_jobs, get_linkedin_jobs_multi

app = FastAPI(title="Job Recommender API")

# Initialize matcher at startup to avoid loading overhead on each request
matcher = JobMatcher()

class SearchJobsRequest(BaseModel):
    keywords: list[str]
    resume_text: str
    cv_details: dict

@app.post("/api/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    """
    Accepts a PDF resume, parses text, calls Groq LLM for extraction,
    runs the XGBoost ML classification and Business Rules.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    content = await file.read()
    
    # Create a file-like object for extract_text_from_pdf
    class DummyFile:
        def read(self):
            return content
            
    pdf_file = DummyFile()
    
    resume_text = extract_text_from_pdf(pdf_file)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        
    cleaned_resume = clean_resume_text(resume_text)
    
    # 1. Summary
    summary = get_llm_response(
        f"Act as an expert HR recruiter. Provide a professional summary of the "
        f"following resume in 3-4 sentences. Highlight the main skills, "
        f"educational background, and experience without assuming any specific "
        f"industry. Resume text: {resume_text}"
    )
    
    # 2. Skill Gaps
    skill_gaps = get_llm_response(
        f"Analyse this resume against the relevant industry standards based "
        f"purely on the candidate's actual background. Identify specific skill "
        f"gaps and suggest practical ways to improve them for their specific "
        f"field. Resume text: {resume_text}"
    )
    
    # 3. Roadmap
    roadmap = get_llm_response(
        f"You are an expert career counselor. Based strictly on the candidate's "
        f"actual background in the resume, provide a realistic step-by-step "
        f"career roadmap. STRICT RULES: 1. DO NOT suggest completing "
        f"qualifications already mentioned. 2. Focus on relevant career "
        f"advancement and professional development specific to their actual "
        f"field. Resume text: {resume_text}"
    )
    
    # 4. Profile Analysis & ML Prediction
    cv_details = analyze_cv_srilanka_context(resume_text)
    
    # 5. Extract ML skills
    detected_skills = matcher.extract_skills(cleaned_resume)
    
    return {
        "resume_text": cleaned_resume,
        "summary": summary,
        "skill_gaps": skill_gaps,
        "roadmap": roadmap,
        "cv_details": cv_details,
        "detected_skills": detected_skills
    }

@app.post("/api/search-jobs")
async def search_jobs(request: SearchJobsRequest):
    """
    Accepts keywords, fetches jobs from LinkedIn via Apify,
    and runs the semantic ML job matching.
    """
    keywords_list = request.keywords
    if not keywords_list:
        raise HTTPException(status_code=400, detail="No keywords provided.")
        
    if len(keywords_list) > 1:
        jobs_list = get_linkedin_jobs_multi(keywords_list[:3], max_jobs_per_keyword=5)
    else:
        jobs_list = get_linkedin_jobs(keywords_list[0])
        
    if not jobs_list:
        return {"ranked_jobs": [], "filtered_jobs": [], "stats": {}}
        
    # Rank jobs using the semantic matcher
    ranked_jobs = matcher.rank_jobs(
        request.resume_text, 
        jobs_list, 
        cv_details=request.cv_details
    )
    
    MIN_JOB_SCORE = 35.0
    import numpy as np
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        return obj

    ranked_jobs = convert_numpy(ranked_jobs)
    filtered_jobs = [j for j in ranked_jobs if j["match_data"]["total_score"] >= MIN_JOB_SCORE]
    
    stats = {}
    if filtered_jobs:
        scores = [j["match_data"]["total_score"] for j in filtered_jobs]
        stats = {
            "avg_score": float(sum(scores) / len(scores)),
            "max_score": float(max(scores)),
            "min_score": float(min(scores)),
            "good_matches": int(len([s for s in scores if s >= 60])),
            "total_scores": int(len(scores))
        }
        
    return {
        "ranked_jobs": ranked_jobs,
        "filtered_jobs": filtered_jobs,
        "stats": convert_numpy(stats)
    }
