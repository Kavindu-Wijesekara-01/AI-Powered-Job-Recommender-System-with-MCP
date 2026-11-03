"""
helper.py — Hybrid LLM + ML Job Recommender Engine
====================================================
Architecture:
  Stage 1 (Groq LLM)  → Strict Data Extractor: reads CV, outputs structured JSON profile.
  Stage 2 (ML Model)   → XGBoost Classifier: maps the structured profile (28 features)
                          to the exact job category, enforcing all business rules.

Business Rules:
  Rule 1 (Experience Override):
      If experience_years > 1 AND experience_field is known → prioritize that field.
      Education background is bypassed.
  Rule 2 (Education-Based, Freshers / ≤ 1 yr):
      Pure Sciences   → science_analytical      (NEVER management)
      Commerce        → audit_management / finance_accounting
      General IT      → it_general
      Specialized IT  → software_engineering / data_science_ml
      Arts/Music      → music_arts
      Healthcare      → healthcare
      Teaching/Edu    → teaching_education
      Other           → marketing_sales
  Rule 3 (Strict Filter):
      Confidence threshold = 0.72. If below, return empty recommendation list.
"""

import fitz
import re
import os
import json
import numpy as np
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

# ── Optional: joblib for model loading ───────────────────────────────────────
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

load_dotenv()

GROQ_API_KEY = os.getenv("groq_api_key")
client = Groq(api_key=GROQ_API_KEY)

def _get_available_models():
    try:
        if GROQ_API_KEY:
            return {m.id for m in client.models.list().data}
    except Exception as e:
        print(f"Warning: Could not fetch Groq models: {e}")
    return set()

_AVAILABLE_GROQ_MODELS = _get_available_models()

def _choose_model(preferred: list[str], default: str) -> str:
    if _AVAILABLE_GROQ_MODELS:
        for m in preferred:
            if m in _AVAILABLE_GROQ_MODELS:
                return m
    return default

FAST_MODEL = os.getenv(
    "GROQ_FAST_MODEL",
    _choose_model(["llama-3.1-8b-instant", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"], "openai/gpt-oss-20b")
)

SMART_MODEL = os.getenv(
    "GROQ_SMART_MODEL",
    _choose_model(["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"], "openai/gpt-oss-120b")
)

# ── Model paths ───────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent  # project root
MODEL_PATH = _BASE_DIR / "models" / "job_recommender_model.pkl"
ENCODER_PATH = _BASE_DIR / "models" / "label_encoder.pkl"

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (must match model_training.ipynb / train_model.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

EDUCATION_CATEGORIES = [
    "Pure Sciences",
    "Commerce/Accounting",
    "General IT",
    "Specialized IT",
    "Arts/Music",
    "Healthcare",
    "Teaching/Education",
    "Other",
]

EXPERIENCE_FIELDS = [
    "Software Engineering",
    "Data Science",
    "Finance/Accounting",
    "Management",
    "IT General",
    "Music/Arts",
    "Healthcare",
    "Teaching",
    "Marketing/Sales",
    "None",
]

EDUCATION_LEVELS = ["Other", "Diploma", "Bachelor", "Master", "PhD"]

SKILL_CATEGORIES = ["technical", "analytical", "creative", "business", "medical"]

# Which experience fields are "natural" for a given education category
EDUCATION_EXPERIENCE_ALIGNMENT = {
    "Pure Sciences":        {"Data Science", "IT General", "None"},
    "Commerce/Accounting":  {"Finance/Accounting", "Management", "Marketing/Sales", "None"},
    "General IT":           {"IT General", "Software Engineering", "None"},
    "Specialized IT":       {"Software Engineering", "Data Science", "None"},
    "Arts/Music":           {"Music/Arts", "None"},
    "Healthcare":           {"Healthcare", "None"},
    "Teaching/Education":   {"Teaching", "None"},
    "Other":                set(EXPERIENCE_FIELDS),
}

CONFIDENCE_THRESHOLD = 0.72  # Rule 3: strict filter (raised from 0.65)

# ── Curated keywords per job category ────────────────────────────────────────
CATEGORY_TO_KEYWORDS: dict[str, list[str]] = {
    "software_engineering": [
        "Software Engineer",
        "Backend Developer",
        "Full Stack Developer",
        "Frontend Developer",
        "Software Developer",
    ],
    "data_science_ml": [
        "Data Scientist",
        "Machine Learning Engineer",
        "AI Engineer",
        "Data Analyst",
        "Data Engineer",
    ],
    "it_general": [
        "IT Support Engineer",
        "System Administrator",
        "Network Engineer",
        "IT Consultant",
        "IT Manager",
    ],
    "finance_accounting": [
        "Accountant",
        "Finance Executive",
        "Financial Analyst",
        "Accounts Executive",
        "Finance Manager",
    ],
    "audit_management": [
        "Audit Trainee",
        "Internal Auditor",
        "Management Trainee",
        "Business Analyst",
        "Operations Manager",
    ],
    "science_analytical": [
        "Research Analyst",
        "Statistical Analyst",
        "Quality Assurance Engineer",
        "Lab Analyst",
        "Data Analyst",
    ],
    "music_arts": [
        "Music Teacher",
        "Sound Engineer",
        "Music Producer",
        "Audio Engineer",
        "Music Director",
    ],
    "teaching_education": [
        "Teacher",
        "Lecturer",
        "Academic Coordinator",
        "Training Specialist",
        "Instructor",
    ],
    "healthcare": [
        "Healthcare Staff",
        "Clinical Officer",
        "Pharmacist",
        "Healthcare Administrator",
        "Medical Officer",
    ],
    "marketing_sales": [
        "Marketing Executive",
        "Sales Executive",
        "Digital Marketer",
        "Brand Manager",
        "Business Development Executive",
    ],
}

# ── Human-readable display names ─────────────────────────────────────────────
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "software_engineering": "Software Engineering",
    "data_science_ml": "Data Science / AI / ML",
    "it_general": "General IT",
    "finance_accounting": "Finance & Accounting",
    "audit_management": "Audit & Management",
    "science_analytical": "Science & Analytics",
    "music_arts": "Music & Performing Arts",
    "teaching_education": "Teaching & Education",
    "healthcare": "Healthcare & Medicine",
    "marketing_sales": "Marketing & Sales",
}

# ─────────────────────────────────────────────────────────────────────────────
# PDF & Text Utilities (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str | None:
    """Extract raw text from an uploaded PDF file object."""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def clean_resume_text(text: str) -> str:
    """Clean raw resume text for ML processing."""
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


def extract_resume_sections(text: str) -> dict:
    """
    Split resume into labelled sections (skills, experience, education, etc.).
    Returns a dict with section names as keys and text as values.
    """
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "summary": "",
        "other": "",
    }
    section_patterns = {
        "skills": r"(?i)(technical\s+skills|skills|technologies|competencies)",
        "experience": r"(?i)(work\s+experience|experience|employment|professional\s+experience)",
        "education": r"(?i)(education|academic|qualifications|degrees|certifications)",
        "projects": r"(?i)(projects|portfolio|personal\s+projects|academic\s+projects)",
        "summary": r"(?i)(summary|objective|profile|about\s+me|professional\s+summary)",
    }
    lines = text.split("\n")
    current_section = "other"
    for line in lines:
        stripped = line.strip()
        if not stripped:
            sections[current_section] += "\n"
            continue
        matched = False
        for section_name, pattern in section_patterns.items():
            if re.match(pattern, stripped):
                current_section = section_name
                matched = True
                break
        if not matched:
            sections[current_section] += stripped + "\n"
    for key in sections:
        sections[key] = sections[key].strip()
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Groq LLM — General Helpers (kept for summary / roadmap / gap tabs)
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_response(prompt: str) -> str | None:
    """Generic LLM call — used for summary, skill gaps, roadmap tabs."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=FAST_MODEL,
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error with Groq LLM: {e}")
        return None


def extract_skills_with_llm(text: str) -> list:
    """
    Extracts technical and professional skills from CV text via LLM.
    Returns a clean list of skill strings.
    Used by the ML ranking engine (ml_matcher.py) — unchanged role.
    """
    if not text or not text.strip():
        return []

    prompt = f"""You are an expert AI skill extractor. Extract all the technical and professional skills from the following text.
Return ONLY a JSON array of strings representing the skills. No other text, markdown, or explanations.
Example output: ["Python", "Machine Learning", "Communication", "React"]

Text:
{text[:2000]}
"""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You output only valid JSON arrays. No markdown, no explanations."},
                {"role": "user", "content": prompt},
            ],
            model=FAST_MODEL,
            temperature=0.1,
        )
        response_text = response.choices[0].message.content.strip()
        response_text = _strip_markdown_fences(response_text)
        skills = json.loads(response_text)
        if isinstance(skills, list):
            return [str(s) for s in skills]
        return []
    except Exception as e:
        print(f"Error extracting skills with LLM: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Groq LLM — Structured Profile Extractor  (v2 — enriched)
# ─────────────────────────────────────────────────────────────────────────────

def extract_profile_with_llm(resume_text: str) -> dict:
    """
    STAGE 1 — LLM as a Pure Data Extractor (v2).

    Reads the raw CV text and outputs a strictly structured JSON profile with:
      - experience_years       (float)
      - experience_field       (one of EXPERIENCE_FIELDS)
      - education_category     (one of EDUCATION_CATEGORIES)
      - education_level        (one of EDUCATION_LEVELS)          ← NEW
      - skill_category         (one of SKILL_CATEGORIES)          ← NEW
      - key_skills             (list of strings)
      - education_details      (string summary)

    The LLM does NOT choose jobs — it only extracts factual profile data.
    """
    edu_options = " | ".join(EDUCATION_CATEGORIES)
    exp_options = " | ".join(EXPERIENCE_FIELDS)
    edu_level_options = " | ".join(EDUCATION_LEVELS)
    skill_cat_options = " | ".join(SKILL_CATEGORIES)

    prompt = f"""You are a precise HR data extraction system. Your ONLY job is to extract factual data from the resume below and return it as a strict JSON object. Do NOT suggest jobs or careers.

EDUCATION_CATEGORIES (pick EXACTLY one):
  {edu_options}

  Mapping guidance:
  - "Pure Sciences"       → A/L Maths, Physics, BSc Physics, Statistics, Applied Maths, Chemistry, Biology (non-medical)
  - "Commerce/Accounting" → A/L Commerce, BCom, AAT, CIMA, ACCA, Accounting degree, Business Studies, BBA
  - "General IT"          → HNDIT, IT Diploma, basic CompSci certificate, IT certificate courses, basic computing
  - "Specialized IT"      → BSc CS (Data Science specialization), MSc Machine Learning, Software Engineering degree, AI/ML degree, BSc in Software Engineering, BSc CS with honours
  - "Arts/Music"          → Music diploma, Conservatory, Fine Arts, Drama, Performing Arts, Music degree, Art degree
  - "Healthcare"          → MBBS, Nursing, Pharmacy, Physiotherapy, Medical degree, BDS, Biomedical Sciences
  - "Teaching/Education"  → BEd, PGDE, Teaching Diploma, Education degree
  - "Other"               → Law, Social Sciences, Psychology, General Arts, Political Science, Unknown

EXPERIENCE_FIELDS (pick EXACTLY one — use "None" if no real work experience):
  {exp_options}

  Rules for experience_field:
  - Use the ACTUAL FIELD they worked in (not what they studied).
  - Use "None" if experience_years <= 0 or if no real work/job/internship history is found.
  - If they did music gigs professionally → "Music/Arts"
  - If they worked as a software intern for 6 months → "Software Engineering" (with 0.5 years)
  - If they tutored students → "Teaching"
  - If they did freelance web development → "Software Engineering"

EDUCATION_LEVELS (pick EXACTLY one):
  {edu_level_options}

  - "Other"    → No formal higher education, or unknown
  - "Diploma"  → HND, HNDIT, professional diploma, certificate courses, AAT
  - "Bachelor" → BSc, BA, BCom, BEd, BBA, MBBS, any bachelor's degree
  - "Master"   → MSc, MBA, MA, MEd, any master's degree
  - "PhD"      → PhD, DPhil, doctoral degree

SKILL_CATEGORIES (pick EXACTLY one — based on their DOMINANT skill type):
  {skill_cat_options}

  - "technical"   → Programming, software development, IT infrastructure, networking, engineering
  - "analytical"  → Data analysis, statistics, mathematics, research, scientific computation
  - "creative"    → Music, art, design, writing, content creation, performing arts
  - "business"    → Management, finance, accounting, marketing, sales, HR, administration
  - "medical"     → Healthcare, nursing, pharmacy, clinical, laboratory, patient care

Extract and return ONLY this JSON, no extra text:
{{
  "experience_years": <float, e.g. 2.5 — total professional work experience, 0 if none>,
  "experience_field": "<one of the EXPERIENCE_FIELDS above>",
  "education_category": "<one of the EDUCATION_CATEGORIES above>",
  "education_level": "<one of the EDUCATION_LEVELS above>",
  "skill_category": "<one of the SKILL_CATEGORIES above>",
  "education_details": "<concise string: highest degree, subjects, grades>",
  "key_skills": ["<skill1>", "<skill2>", "..."]
}}

RESUME TEXT:
{resume_text}
"""
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON generator. "
                        "Output ONLY valid JSON. No markdown, no explanations, no extra text. "
                        "Pick category values EXACTLY from the provided lists."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=SMART_MODEL,  # Stronger model for accurate field detection
            temperature=0.0,  # Zero temperature for deterministic extraction
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_markdown_fences(raw)
        profile = json.loads(raw)

        # Validate and sanitise extracted values
        profile["experience_years"] = float(profile.get("experience_years", 0) or 0)

        exp_field = profile.get("experience_field", "None")
        if exp_field not in EXPERIENCE_FIELDS:
            exp_field = _fuzzy_match_field(exp_field, EXPERIENCE_FIELDS, fallback="None")
        profile["experience_field"] = exp_field

        edu_cat = profile.get("education_category", "Other")
        if edu_cat not in EDUCATION_CATEGORIES:
            edu_cat = _fuzzy_match_field(edu_cat, EDUCATION_CATEGORIES, fallback="Other")
        profile["education_category"] = edu_cat

        edu_level = profile.get("education_level", "Other")
        if edu_level not in EDUCATION_LEVELS:
            edu_level = _fuzzy_match_field(edu_level, EDUCATION_LEVELS, fallback="Other")
        profile["education_level"] = edu_level

        skill_cat = profile.get("skill_category", "technical")
        if skill_cat not in SKILL_CATEGORIES:
            skill_cat = _fuzzy_match_field(skill_cat, SKILL_CATEGORIES, fallback="technical")
        profile["skill_category"] = skill_cat

        if not isinstance(profile.get("key_skills"), list):
            profile["key_skills"] = []
        if not isinstance(profile.get("education_details"), str):
            profile["education_details"] = ""

        return profile

    except Exception as e:
        print(f"LLM profile extraction error: {e}")
        return {
            "experience_years": 0.0,
            "experience_field": "None",
            "education_category": "Other",
            "education_level": "Other",
            "skill_category": "technical",
            "education_details": "",
            "key_skills": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: ML Model — Feature Engineering + Inference  (v2 — 28 features)
# ─────────────────────────────────────────────────────────────────────────────

def encode_profile_for_model(profile: dict) -> np.ndarray:
    """
    Converts a structured profile dict into a 28-feature float32 numpy vector.

    Feature Layout (28 features) — must match train_model.py exactly:
      [0]     experience_years_normalized    (0.0–1.0, capped at 20 yrs)
      [1]     has_significant_experience     (1.0 if exp > 1 yr)
      [2–9]   education_category one-hot     (8 categories)
      [10–19] experience_field one-hot       (10 fields)
      [20]    education_level ordinal        (0.0–1.0, normalised / 4)
      [21–25] skill_category one-hot         (5 categories)
      [26]    experience_education_match     (1.0 if experience aligns with edu)
      [27]    is_career_switcher             (1.0 if exp > 1 yr & misaligned field)
    """
    features: list[float] = []

    # ── [0–1] Experience features ────────────────────────────────────────────
    exp_years = float(profile.get("experience_years", 0))
    features.append(min(exp_years / 20.0, 1.0))           # [0]
    features.append(1.0 if exp_years > 1.0 else 0.0)      # [1]

    # ── [2–9] Education category one-hot ─────────────────────────────────────
    edu_cat = profile.get("education_category", "Other")
    for cat in EDUCATION_CATEGORIES:
        features.append(1.0 if edu_cat == cat else 0.0)

    # ── [10–19] Experience field one-hot ─────────────────────────────────────
    exp_field = profile.get("experience_field", "None")
    for field in EXPERIENCE_FIELDS:
        features.append(1.0 if exp_field == field else 0.0)

    # ── [20] Education level ordinal ─────────────────────────────────────────
    edu_level = profile.get("education_level", "Other")
    level_idx = EDUCATION_LEVELS.index(edu_level) if edu_level in EDUCATION_LEVELS else 0
    features.append(level_idx / 4.0)

    # ── [21–25] Skill category one-hot ───────────────────────────────────────
    skill_cat = profile.get("skill_category", "technical")
    for sc in SKILL_CATEGORIES:
        features.append(1.0 if skill_cat == sc else 0.0)

    # ── [26] Experience–education alignment ──────────────────────────────────
    aligned = EDUCATION_EXPERIENCE_ALIGNMENT.get(edu_cat, set())
    features.append(1.0 if exp_field in aligned else 0.0)

    # ── [27] Career switcher flag ────────────────────────────────────────────
    is_switcher = (exp_years > 1.0 and exp_field != "None" and exp_field not in aligned)
    features.append(1.0 if is_switcher else 0.0)

    return np.array(features, dtype=np.float32)


@lru_cache(maxsize=1)
def load_recommendation_model() -> tuple | None:
    """
    Loads the trained XGBoost model and LabelEncoder from disk.
    Uses lru_cache so the model is loaded only once across all calls.

    Returns:
        (model, label_encoder) tuple, or None if model files are missing.
    """
    if not JOBLIB_AVAILABLE:
        print("⚠️  joblib not installed. Run: pip install joblib")
        return None

    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        print(
            f"⚠️  Model files not found at {MODEL_PATH}.\n"
            "   Please run model_training.ipynb first to train and save the model."
        )
        return None

    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("✅ Job recommender ML model loaded successfully.")
        return model, label_encoder
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        return None


def predict_job_category_ml(
    profile: dict,
    model,
    label_encoder,
) -> tuple[str, float]:
    """
    STAGE 2 — Pure ML Inference.

    Predicts the best job category + confidence score from the encoded profile.
    Does NOT apply business rule overrides (those come in apply_business_rules()).

    Returns:
        (category_str, confidence_float 0.0–1.0)
    """
    feature_vec = encode_profile_for_model(profile).reshape(1, -1)
    probas = model.predict_proba(feature_vec)[0]
    best_idx = int(np.argmax(probas))
    confidence = float(probas[best_idx])
    category = label_encoder.inverse_transform([best_idx])[0]
    return category, confidence


def apply_business_rules(
    profile: dict,
    ml_category: str,
    ml_confidence: float,
) -> tuple[str, float, str]:
    """
    Enforces all 3 strict business rules on top of the ML output.

    Rule 1 — Experience Override (hard override):
        If exp > 1 yr AND field is known → map directly to the experience field category.
        Confidence is forced to 1.0 (certainty), bypassing the ML output.

    Rule 2 — Education Guard (soft correction):
        For freshers, prevent illogical cross-category predictions
        (e.g., Pure Sciences → audit_management is BLOCKED).

    Rule 3 — Confidence Threshold:
        If final confidence < CONFIDENCE_THRESHOLD → return ("__below_threshold__", ...).

    Returns:
        (final_category, final_confidence, rule_applied)
    """
    exp_years = float(profile.get("experience_years", 0))
    exp_field = profile.get("experience_field", "None")
    edu_cat = profile.get("education_category", "Other")

    # ── Rule 1: Experience Override ──────────────────────────────────────────
    EXP_FIELD_TO_CATEGORY: dict[str, str] = {
        "Software Engineering": "software_engineering",
        "Data Science":          "data_science_ml",
        "Finance/Accounting":    "finance_accounting",
        "Management":            "audit_management",
        "IT General":            "it_general",
        "Music/Arts":            "music_arts",
        "Healthcare":            "healthcare",
        "Teaching":              "teaching_education",
        "Marketing/Sales":       "marketing_sales",
    }

    if exp_years > 1.0 and exp_field in EXP_FIELD_TO_CATEGORY:
        category = EXP_FIELD_TO_CATEGORY[exp_field]
        return category, 1.0, "rule_1_experience_override"

    # ── Rule 2: Education Guard (fresher path) ───────────────────────────────
    # Define which categories are FORBIDDEN for each education type
    FORBIDDEN_CATEGORIES: dict[str, set[str]] = {
        "Pure Sciences": {
            "audit_management",       # STRICTLY no generic management for scientists
            "finance_accounting",     # Not finance either
            "music_arts",
            "teaching_education",
            "healthcare",
            "marketing_sales",
        },
        "Arts/Music": {
            "software_engineering",
            "data_science_ml",
            "it_general",
            "finance_accounting",
            "audit_management",
            "science_analytical",
            "healthcare",
            "marketing_sales",
        },
        "Healthcare": {
            "software_engineering",
            "data_science_ml",
            "it_general",
            "music_arts",
            "finance_accounting",
            "audit_management",
            "marketing_sales",
            "science_analytical",
        },
        "Teaching/Education": {
            "software_engineering",
            "data_science_ml",
            "finance_accounting",
            "audit_management",
            "healthcare",
            "music_arts",
            "it_general",
            "marketing_sales",
        },
        "Specialized IT": {
            "music_arts",
            "teaching_education",
            "healthcare",
            "finance_accounting",
            "audit_management",
            "marketing_sales",
        },
        "General IT": {
            "music_arts",
            "teaching_education",
            "healthcare",
            "finance_accounting",
            "audit_management",
            "data_science_ml",
        },
        "Commerce/Accounting": {
            "software_engineering",
            "data_science_ml",
            "music_arts",
            "healthcare",
            "science_analytical",
            "it_general",
            "teaching_education",
        },
        "Other": {
            "software_engineering",
            "data_science_ml",
            "music_arts",
            "healthcare",
            "science_analytical",
            "teaching_education",
        },
    }

    # Correction map: if ML predicts a forbidden category, what to use instead
    FALLBACK_CATEGORY: dict[str, str] = {
        "Pure Sciences":        "science_analytical",
        "Arts/Music":           "music_arts",
        "Healthcare":           "healthcare",
        "Teaching/Education":   "teaching_education",
        "General IT":           "it_general",
        "Specialized IT":       "software_engineering",
        "Commerce/Accounting":  "audit_management",
        "Other":                "marketing_sales",
    }

    forbidden = FORBIDDEN_CATEGORIES.get(edu_cat, set())
    if ml_category in forbidden:
        corrected_category = FALLBACK_CATEGORY.get(edu_cat, ml_category)
        # Correcting confidence slightly downward since we overrode ML
        corrected_confidence = max(ml_confidence * 0.85, CONFIDENCE_THRESHOLD)
        return corrected_category, corrected_confidence, "rule_2_education_guard"

    # ── Rule 3: Confidence threshold ─────────────────────────────────────────
    if ml_confidence < CONFIDENCE_THRESHOLD:
        return "__below_threshold__", ml_confidence, "rule_3_low_confidence"

    return ml_category, ml_confidence, "ml_prediction"


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cv_srilanka_context(resume_text: str) -> dict:
    """
    Main orchestrator for the Hybrid LLM + ML recommendation pipeline.

    Flow:
      1. Groq LLM extracts structured profile JSON from raw CV text.
      2. ML Model predicts job category + confidence.
      3. Business Rule Engine (apply_business_rules) enforces hard rules.
      4. Category is mapped to search keywords.

    Returns:
        dict with keys:
          - experience_years     (float)
          - experience_field     (str)
          - education_category   (str)
          - education            (str)   ← human-readable summary (for UI)
          - primary_field        (str)   ← display name of predicted category
          - job_category         (str)   ← raw category key
          - job_keywords         (list)  ← LinkedIn search keywords
          - confidence           (float) ← 0.0–1.0 match confidence
          - rule_applied         (str)   ← which rule fired
          - key_skills           (list)
          - recommendation_ok    (bool)  ← False if below confidence threshold
    """
    # ── Step 1: LLM Profile Extraction ───────────────────────────────────────
    profile = extract_profile_with_llm(resume_text)

    exp_years = profile.get("experience_years", 0.0)
    exp_field = profile.get("experience_field", "None")
    edu_cat = profile.get("education_category", "Other")
    edu_details = profile.get("education_details", "")
    key_skills = profile.get("key_skills", [])

    # ── Step 2: ML Model Inference ────────────────────────────────────────────
    model_result = load_recommendation_model()

    if model_result is not None:
        model, label_encoder = model_result
        ml_category, ml_confidence = predict_job_category_ml(profile, model, label_encoder)
    else:
        # Fallback: use a simple rule-based category without ML
        ml_category, ml_confidence = _rule_based_fallback(profile)

    # ── Step 3: Business Rule Engine ─────────────────────────────────────────
    final_category, final_confidence, rule_applied = apply_business_rules(
        profile, ml_category, ml_confidence
    )

    # ── Step 4: Build output dict ─────────────────────────────────────────────
    if final_category == "__below_threshold__":
        return {
            "experience_years": exp_years,
            "experience_field": exp_field,
            "education_category": edu_cat,
            "education": edu_details,
            "primary_field": "Unknown",
            "job_category": "__below_threshold__",
            "job_keywords": [],
            "confidence": final_confidence,
            "rule_applied": rule_applied,
            "key_skills": key_skills,
            "recommendation_ok": False,
        }

    keywords = CATEGORY_TO_KEYWORDS.get(final_category, [])
    display_name = CATEGORY_DISPLAY_NAMES.get(final_category, final_category.replace("_", " ").title())

    return {
        "experience_years": exp_years,
        "experience_field": exp_field,
        "education_category": edu_cat,
        "education": edu_details,
        "primary_field": display_name,
        "job_category": final_category,
        "job_keywords": keywords,
        "confidence": round(final_confidence, 4),
        "rule_applied": rule_applied,
        "key_skills": key_skills,
        "recommendation_ok": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json … ``` fences that some LLMs add despite instructions."""
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _fuzzy_match_field(value: str, valid_options: list[str], fallback: str) -> str:
    """
    Simple case-insensitive partial match for LLM outputs that almost match
    a valid option (e.g. "Finance" → "Finance/Accounting").
    """
    value_lower = value.lower().strip()
    for option in valid_options:
        if value_lower in option.lower() or option.lower() in value_lower:
            return option
    return fallback


def _rule_based_fallback(profile: dict) -> tuple[str, float]:
    """
    Pure rule-based fallback used when the ML model file is not found.
    Implements the business rules directly in Python — less nuanced than
    the trained XGBoost, but guarantees the hard rules are always enforced.
    """
    exp_years = float(profile.get("experience_years", 0))
    exp_field = profile.get("experience_field", "None")
    edu_cat = profile.get("education_category", "Other")

    EXP_FIELD_TO_CATEGORY = {
        "Software Engineering": "software_engineering",
        "Data Science": "data_science_ml",
        "Finance/Accounting": "finance_accounting",
        "Management": "audit_management",
        "IT General": "it_general",
        "Music/Arts": "music_arts",
        "Healthcare": "healthcare",
        "Teaching": "teaching_education",
        "Marketing/Sales": "marketing_sales",
    }

    EDU_TO_CATEGORY = {
        "Pure Sciences": "science_analytical",
        "Commerce/Accounting": "audit_management",
        "General IT": "it_general",
        "Specialized IT": "software_engineering",
        "Arts/Music": "music_arts",
        "Healthcare": "healthcare",
        "Teaching/Education": "teaching_education",
        "Other": "marketing_sales",
    }

    if exp_years > 1.0 and exp_field in EXP_FIELD_TO_CATEGORY:
        return EXP_FIELD_TO_CATEGORY[exp_field], 0.90

    category = EDU_TO_CATEGORY.get(edu_cat, "marketing_sales")
    return category, 0.80