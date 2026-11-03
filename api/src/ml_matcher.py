"""
ML-Powered Job Matching Engine
Sentence Transformers + TF-IDF + Skill Matching hybrid system
for accurate Resume ↔ Job matching with percentage scores.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process

from src.skills_db import (
    normalize_skill,
    get_all_known_skills,
    detect_experience_level,
    KNOWN_SKILLS,
)
from src.helper import extract_skills_with_llm


class JobMatcher:
    """
    ML-based Resume ↔ Job matching engine.

    Scoring Components:
    - Semantic Similarity (40%): Sentence Transformer embeddings
    - Skill Match (35%): Exact + fuzzy skill matching
    - TF-IDF Similarity (15%): Term frequency based matching
    - Experience Match (10%): Experience level comparison
    """

    # Score weights
    WEIGHT_SEMANTIC = 0.30
    WEIGHT_SKILLS = 0.25
    WEIGHT_EXPERIENCE = 0.25
    WEIGHT_EDUCATION = 0.20

    # Fuzzy match threshold
    FUZZY_THRESHOLD = 80  # 80% similarity for fuzzy skill matching

    def __init__(self):
        """Load ML models (first time download ~80MB, then cached)."""
        print("Loading Sentence Transformer model...")
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),  # unigrams + bigrams
        )
        print("ML models loaded successfully!")

    # =========================================================
    # CORE: Embedding & Similarity
    # =========================================================

    def get_embedding(self, text: str) -> np.ndarray:
        """Text එක 384-dimensional vector embedding බවට convert කරනවා."""
        clean_text = self._preprocess_text(text)
        embedding = self.model.encode(clean_text, show_progress_bar=False)
        return embedding

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Sentence Transformer cosine similarity.
        Returns: 0.0 to 100.0
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Reshape for sklearn cosine_similarity
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)

        similarity = cosine_similarity(emb1, emb2)[0][0]

        # Convert from [-1, 1] range to [0, 100] range
        score = max(0.0, similarity) * 100.0
        return round(score, 1)

    def tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        TF-IDF based cosine similarity.
        Returns: 0.0 to 100.0
        """
        clean1 = self._preprocess_text(text1)
        clean2 = self._preprocess_text(text2)

        try:
            tfidf_matrix = self.tfidf.fit_transform([clean1, clean2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            score = max(0.0, similarity) * 100.0
            return round(score, 1)
        except Exception:
            return 0.0

    # =========================================================
    # SKILL EXTRACTION & MATCHING
    # =========================================================

    def extract_skills(self, text: str) -> list:
        """
        Text එකෙන් skills extract කරනවා.
        Known skills list එක්ක fuzzy matching use කරනවා.
        Returns: list of normalized skill strings
        """
        text_lower = text.lower()
        found_skills = set()

        # 1. Direct keyword matching (exact)
        for skill in KNOWN_SKILLS:
            skill_lower = skill.lower()
            # Word boundary matching for better accuracy
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            if re.search(pattern, text_lower):
                normalized = normalize_skill(skill_lower)
                found_skills.add(normalized)

        # 2. Fuzzy matching for skills that might be slightly different
        # Extract potential skill-like words (2-3 word phrases)
        words = re.findall(r'[a-zA-Z][a-zA-Z0-9+#.]*(?:\s+[a-zA-Z][a-zA-Z0-9+#.]*){0,2}', text)
        for word_group in words:
            word_clean = word_group.strip().lower()
            if len(word_clean) < 2:
                continue

            # Fuzzy match against known skills
            match = process.extractOne(
                word_clean,
                KNOWN_SKILLS,
                scorer=fuzz.ratio,
                score_cutoff=self.FUZZY_THRESHOLD,
            )
            if match:
                matched_skill = match[0]
                normalized = normalize_skill(matched_skill)
                found_skills.add(normalized)

        return sorted(list(found_skills))

    def skill_match_score(self, resume_skills: list, job_skills: list) -> float:
        """
        Resume skills සහ Job skills අතර match score calculate කරනවා.
        Uses semantic vector similarity instead of exact/fuzzy string matching.
        Returns: 0.0 to 100.0
        """
        if not job_skills:
            return 50.0  # Job skills නැත්තම් neutral score

        if not resume_skills:
            return 0.0

        total_matches = 0
        matched_job_skills = set()

        # For each job skill, find the best matching resume skill using semantic similarity
        for job_skill in job_skills:
            best_sim = 0.0
            for resume_skill in resume_skills:
                # We use semantic similarity which returns 0-100
                sim = self.semantic_similarity(job_skill, resume_skill)
                if sim > best_sim:
                    best_sim = sim
            
            # If similarity is > 75%, consider it a match
            if best_sim > 75.0:
                total_matches += 1
                matched_job_skills.add(job_skill)

        score = (total_matches / len(job_skills)) * 100.0
        return round(min(score, 100.0), 1)

    # =========================================================
    # EXPERIENCE MATCHING
    # =========================================================

    def experience_match_score(self, resume_text: str, job_text: str) -> float:
        """
        Experience level comparison.
        Returns: 0.0 to 100.0
        """
        resume_level = detect_experience_level(resume_text)
        job_level = detect_experience_level(job_text)

        if resume_level == -1 or job_level == -1:
            return 50.0  # Can't determine, give neutral score

        # Calculate how close the levels are
        diff = abs(resume_level - job_level)
        max_diff = 7  # Max possible difference

        # Closer levels = higher score
        score = max(0, (1 - (diff / max_diff))) * 100.0

        # Bonus: if resume level >= job level (overqualified is okay)
        if resume_level >= job_level:
            score = min(100.0, score + 10)

        return round(score, 1)

    # =========================================================
    # EDUCATION MATCHING
    # =========================================================

    def education_match_score(self, cv_education_text: str, job_description: str) -> float:
        """
        Calculates how well the CV's specific education matches the job description.
        Uses semantic similarity between extracted education and job description.
        Returns: 0.0 to 100.0
        """
        if not cv_education_text or len(cv_education_text.strip()) == 0:
            return 50.0 # Neutral if no education found
            
        # We check semantic similarity specifically against education requirements.
        # Simple heuristic: semantic similarity between CV education and full job description.
        # A high score means the job requires things mentioned in the CV's education.
        return self.semantic_similarity(cv_education_text, job_description)

    # =========================================================
    # HYBRID SCORING (MAIN METHOD)
    # =========================================================

    def calculate_match_score(self, resume_text: str, job_description: str, cv_details: dict = None) -> dict:
        """
        Hybrid match score calculate කරනවා.

        Returns: dict with:
        - total_score: float (0-100)
        - semantic_score: float (0-100)
        - skill_score: float (0-100)
        - experience_score: float (0-100)
        - education_score: float (0-100)
        - resume_skills: list
        - job_skills: list
        - matched_skills: list
        - missing_skills: list
        """
        # 1. Semantic Similarity
        semantic = self.semantic_similarity(resume_text, job_description)

        # 2. Skill Extraction & Matching
        # Check if LLM skills were pre-extracted in cv_details
        if cv_details and 'llm_resume_skills' in cv_details and cv_details['llm_resume_skills']:
            resume_skills = cv_details['llm_resume_skills']
        else:
            resume_skills = self.extract_skills(resume_text)
            
        # We don't use LLM for job description here as it would take too long for N jobs.
        # We use the fast regex/fuzzy extractor for the job description.
        job_skills = self.extract_skills(job_description)
        skill_score = self.skill_match_score(resume_skills, job_skills)

        # Find matched and missing skills using semantic matching
        matched = []
        missing = []
        for job_skill in job_skills:
            is_match = False
            for resume_skill in resume_skills:
                if self.semantic_similarity(job_skill, resume_skill) > 75.0:
                    is_match = True
                    break
            if is_match:
                matched.append(job_skill)
            else:
                missing.append(job_skill)

        # 3. Experience Match
        experience = self.experience_match_score(resume_text, job_description)
        job_level = detect_experience_level(job_description)
        
        # Override experience match if structured cv_details is provided
        if cv_details and "experience_years" in cv_details:
            cv_years = cv_details["experience_years"]
            if job_level != -1:
                # Map job level roughly to years: intern=0, junior=1, mid=3, senior=5, lead=7+
                level_to_years = {0: 0, 1: 1, 2: 3, 3: 5, 4: 7, 5: 10, 6: 12, 7: 15}
                req_years = level_to_years.get(job_level, 0)
                
                diff = abs(cv_years - req_years)
                score = max(0, (1 - (diff / 5))) * 100.0 # 5 years diff = 0 score
                if cv_years >= req_years:
                    score = min(100.0, score + 15) # Bonus for having more experience
                experience = round(score, 1)

        # 4. Education Match
        education = 50.0
        if cv_details and "education" in cv_details:
            education = self.education_match_score(cv_details["education"], job_description)

        # 5. Dynamic Weighting based on Job Level
        w_sem, w_skill, w_exp, w_edu = 0.30, 0.25, 0.25, 0.20
        if job_level >= 3: # Senior and above (mid=3, senior=5, lead=7+)
            w_sem, w_skill, w_exp, w_edu = 0.25, 0.25, 0.35, 0.15 # Experience is very important
        elif job_level == 0: # Intern/Fresher
            w_sem, w_skill, w_exp, w_edu = 0.20, 0.30, 0.10, 0.40 # Education and skills are very important

        total = (
            semantic * w_sem
            + skill_score * w_skill
            + experience * w_exp
            + education * w_edu
        )

        return {
            "total_score": round(total, 1),
            "semantic_score": semantic,
            "skill_score": skill_score,
            "experience_score": experience,
            "education_score": education,
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "matched_skills": matched,
            "missing_skills": missing,
        }

    def rank_jobs(self, resume_text: str, jobs_list: list, cv_details: dict = None) -> list:
        """
        Job list එක match score අනුව rank කරනවා.
        """
        ranked = []
        
        # Optimize: Extract resume skills ONLY ONCE using LLM for extreme accuracy
        llm_skills = extract_skills_with_llm(resume_text)
        if llm_skills:
            # Overwrite the extract_skills function temporarily for this instance
            # or just pass it in. Since calculate_match_score calls self.extract_skills(resume_text),
            # we should dynamically patch it or pass it. Let's pass it.
            pass
            
        # We need to modify calculate_match_score to accept pre-extracted skills.
        # But to avoid breaking other things, we will just use the LLM skills for cv_details and use them in calculate_match_score.
        if not cv_details:
            cv_details = {}
        cv_details['llm_resume_skills'] = llm_skills

        for job in jobs_list:
            job_desc = job.get("description", "")

            # Description නැත්තම් title + company එකෙන් match කරනවා
            if not job_desc or job_desc.strip() == "":
                job_desc = f"{job.get('title', '')} {job.get('company', '')} {job.get('location', '')}"

            match_data = self.calculate_match_score(resume_text, job_desc, cv_details)

            job_with_score = {**job, "match_data": match_data}
            ranked.append(job_with_score)

        # Sort by total_score descending
        ranked.sort(key=lambda x: x["match_data"]["total_score"], reverse=True)

        return ranked

    # =========================================================
    # UTILITIES
    # =========================================================

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Text clean කරනවා for better ML processing."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s+#./\-@]', ' ', text)

        # Normalize whitespace
        text = text.strip()

        return text

    @staticmethod
    def get_score_color(score: float) -> str:
        """Score එකට අනුව color return කරනවා."""
        if score >= 80:
            return "🟢"
        elif score >= 60:
            return "🟡"
        elif score >= 40:
            return "🟠"
        else:
            return "🔴"

    @staticmethod
    def get_score_label(score: float) -> str:
        """Score එකට අනුව label return කරනවා."""
        if score >= 85:
            return "Excellent Match"
        elif score >= 70:
            return "Good Match"
        elif score >= 55:
            return "Fair Match"
        elif score >= 40:
            return "Weak Match"
        else:
            return "Poor Match"
