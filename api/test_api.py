import requests

payload = {
    "keywords": ["accountant"],
    "resume_text": "I am an accountant with 5 years of experience in auditing and taxation.",
    "cv_details": {
        "primary_field": "Finance",
        "llm_resume_skills": ["Auditing", "Taxation", "Excel"]
    }
}

try:
    res = requests.post("http://localhost:8000/api/search-jobs", json=payload)
    print("Status:", res.status_code)
    print("Response:", res.text)
except Exception as e:
    print("Error:", e)
