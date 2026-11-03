import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Executive Career Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Bilingual Dictionary (Formal Enterprise Tone - Zero Emojis) ───────────────
TRANSLATIONS = {
    "en": {
        "nav_brand": "CAREER INTELLIGENCE",
        "app_title": "Career Intelligence & Market Alignment Dashboard",
        "app_subtitle": "Autonomous resume ingestion, statistical domain classification, and live labor market alignment.",
        "lang_select": "Interface Language / භාෂාව",
        "upload_label": "Upload Curriculum Vitae (PDF)",
        "upload_help": "Upload a searchable PDF document for automated parsing.",
        "upload_success": "Document staged: ",
        "btn_analyze": "Run Analysis & Market Match",
        "analyzing_spinner": "Running multi-stage inference pipeline...",
        "fetching_jobs_spinner": "Querying live labor registry and ranking requisitions...",
        "tab_summary": "Executive Summary",
        "tab_gaps": "Competency Deficits",
        "tab_roadmap": "Strategic Roadmap",
        "tab_profile": "Profile Parameters",
        "tab_skills": "Verified Proficiencies",
        "summary_title": "Executive Candidate Summary",
        "gaps_title": "Identified Competency Deficits",
        "roadmap_title": "Strategic Development Trajectory",
        "profile_title": "Candidate Classification Parameters",
        "profile_caption": "Stage 1: Groq LLM Ingestion | Stage 2: XGBoost Classifier | Stage 3: Deterministic Rules",
        "exp_years": "Verified Tenure",
        "years": "Years",
        "exp_field": "Primary Specialization",
        "edu_category": "Academic Stream",
        "edu_details": "Highest Qualification",
        "ml_result_title": "Statistical Domain Prediction",
        "pred_domain": "Predicted Professional Domain",
        "high_conf": "High Confidence",
        "med_conf": "Moderate Confidence",
        "low_conf": "Low Confidence",
        "job_kw_title": "Recommended Ingestion Keywords",
        "llm_skills_title": "Extracted Key Attributes",
        "ml_skills_title": "Corpus-Identified Technical Attributes",
        "skills_detected_count": "competencies verified against taxonomy",
        "no_skills_detected": "Zero distinct attributes verified from document.",
        "low_conf_warning": "Inference confidence fell below the operational threshold (72.0%). Recommendations are suppressed to ensure precision.",
        "jobs_title": "Active Market Requisitions (LinkedIn Sri Lanka)",
        "jobs_ranked_desc": "Live opportunities retrieved and ranked via multi-criteria evaluation.",
        "no_jobs_qualified": "No active requisitions met the composite qualification threshold.",
        "no_jobs_found": "Zero matching requisitions recorded in the registry for these parameters.",
        "found_matches": "Qualified Requisitions",
        "out_of": "out of",
        "fetched": "evaluated listings",
        "company": "Organization",
        "location": "Location",
        "match_score": "Composite Index",
        "breakdown": "Evaluation Index Breakdown",
        "semantic_metric": "Semantic Distance",
        "skills_metric": "Skill Overlap",
        "edu_metric": "Academic Parity",
        "exp_metric": "Tenure Parity",
        "matched_skills": "Verified Competencies (Present)",
        "missing_skills": "Required Competencies (Deficit)",
        "view_job_btn": "View Primary Requisition [External]",
        "overall_stats": "Corpus Aggregate Metrics",
        "avg_score": "Mean Index",
        "high_score": "Peak Index",
        "low_score": "Baseline Index",
        "good_matches": "Qualified Volume",
        "rule_labels": {
            "rule_1_experience_override": "Rule 1: Tenure Priority Override Enacted",
            "rule_2_education_guard": "Rule 2: Academic Domain Guard Enacted",
            "rule_3_low_confidence": "Rule 3: Confidence Suppression Threshold Active",
            "ml_prediction": "Unmodified Statistical Model Output"
        },
        "score_labels": {
            "excellent": "Superior Fit",
            "good": "Standard Fit",
            "fair": "Marginal Fit",
            "weak": "Insufficient Fit"
        }
    },
    "si": {
        "nav_brand": "වෘත්තීය බුද්ධි පුවරුව",
        "app_title": "වෘත්තීය බුද්ධි හා ශ්‍රම වෙළඳපල පෙළගැස්ම",
        "app_subtitle": "ස්වයංක්‍රීය CV දත්ත කියවීම, සංඛ්‍යානමය ක්ෂේත්‍ර වර්ගීකරණය සහ LinkedIn සජීවී රැකියා විග්‍රහය.",
        "lang_select": "භාෂාව / Language",
        "upload_label": "CV (PDF ආකෘතියෙන්) ඇතුළත් කරන්න",
        "upload_help": "ස්වයංක්‍රීය කියවීම සඳහා text කියවිය හැකි PDF ගොනුවක් ඇතුළත් කරන්න.",
        "upload_success": "විශ්ලේෂණය සඳහා ලේඛනය සූදානම්: ",
        "btn_analyze": "විශ්ලේෂණය සහ වෙළඳපල සැසඳීම ආරම්භ කරන්න",
        "analyzing_spinner": "බහු-අදියර පද්ධතිය මඟින් ලේඛනය විශ්ලේෂණය කෙරේ...",
        "fetching_jobs_spinner": "ශ්‍රම වෙළඳපල දත්ත පරීක්ෂා කර රැකියා ශ්‍රේණිගත කෙරේ...",
        "tab_summary": "විධායක සාරාංශය",
        "tab_gaps": "කුසලතා පරතරය",
        "tab_roadmap": "සංවර්ධන සැලැස්ම",
        "tab_profile": "පැතිකඩ පරාමිතීන්",
        "tab_skills": "තහවුරු කළ කුසලතා",
        "summary_title": "විධායක අපේක්ෂක සාරාංශය",
        "gaps_title": "හඳුනාගත් කුසලතා පරතරයන්",
        "roadmap_title": "උපායමාර්ගික වෘත්තීය සංවර්ධන සැලැස්ම",
        "profile_title": "අපේක්ෂක වර්ගීකරණ පරාමිතීන්",
        "profile_caption": "අදියර 1: Groq LLM දත්ත ලබාගැනීම | අදියර 2: XGBoost ක්ෂේත්‍ර වර්ගීකරණය | අදියර 3: ව්‍යාපාරික නීති",
        "exp_years": "සේවා පළපුරුද්ද",
        "years": "වසර",
        "exp_field": "ප්‍රධාන විශේෂඥතාව",
        "edu_category": "අධ්‍යාපන ක්ෂේත්‍රය",
        "edu_details": "ඉහළම සුදුසුකම",
        "ml_result_title": "සංඛ්‍යානමය ක්ෂේත්‍ර පුරෝකථනය",
        "pred_domain": "අනුමාන කළ වෘත්තීය ක්ෂේත්‍රය",
        "high_conf": "ඉහළ නිරවද්‍යතාව",
        "med_conf": "මධ්‍යම නිරවද්‍යතාව",
        "low_conf": "අඩු නිරවද්‍යතාව",
        "job_kw_title": "නිර්දේශිත සෙවුම් මූලපද",
        "llm_skills_title": "හඳුනාගත් ප්‍රධාන ලක්ෂණ",
        "ml_skills_title": "ලේඛනයෙන් තහවුරු කළ තාක්ෂණික කුසලතා",
        "skills_detected_count": "කුසලතා ප්‍රමාණයක් සාර්ථකව තහවුරු විය",
        "no_skills_detected": "කිසිදු පැහැදිලි කුසලතාවක් ලේඛනයෙන් තහවුරු නොවීය.",
        "low_conf_warning": "පුරෝකථන නිරවද්‍යතාව අවම සීමාවට (72.0%) වඩා අඩුය. තීරණවල නිරවද්‍යතාව තහවුරු කරගැනීම සඳහා නිර්දේශ පෙන්වීම සීමා කර ඇත.",
        "jobs_title": "සක්‍රීය රැකියා අවස්ථා (LinkedIn ශ්‍රී ලංකා)",
        "jobs_ranked_desc": "සජීවීව ලබාගත් අවස්ථා බහු-සාධක විශ්ලේෂණයක් හරහා ශ්‍රේණිගත කර ඇත.",
        "no_jobs_qualified": "සම්මත සුදුසුකම් සීමාව සපුරාලන රැකියා වාර්තා නොවීය.",
        "no_jobs_found": "මෙම පරාමිතීන් සඳහා මේ මොහොතේ ලියාපදිංචි රැකියා හමු නොවීය.",
        "found_matches": "සුදුසු රැකියා අවස්ථා",
        "out_of": "ප්‍රමාණයෙන්",
        "fetched": "සොයාගත් වෙළඳපල දැන්වීම්",
        "company": "ආයතනය",
        "location": "ස්ථානය",
        "match_score": "සංයුක්ත ගැළපීමේ දර්ශකය",
        "breakdown": "දර්ශක ලකුණු විග්‍රහය",
        "semantic_metric": "අර්ථමය ගැළපීම",
        "skills_metric": "කුසලතා ගැළපීම",
        "edu_metric": "අධ්‍යාපන ගැළපීම",
        "exp_metric": "සේවා කාල ගැළපීම",
        "matched_skills": "තහවුරු කළ කුසලතා (පවතින)",
        "missing_skills": "අවශ්‍ය අතිරේක කුසලතා (අඩුපාඩු)",
        "view_job_btn": "මූලාශ්‍ර රැකියා දැන්වීම බලන්න [External]",
        "overall_stats": "සමස්ත වෙළඳපල සංඛ්‍යාලේඛන",
        "avg_score": "මධ්‍යන්‍ය දර්ශකය",
        "high_score": "උපරිම දර්ශකය",
        "low_score": "මූලික දර්ශකය",
        "good_matches": "සුදුසු පරිමාව",
        "rule_labels": {
            "rule_1_experience_override": "නීතිය 1: සේවා කාල ප්‍රමුඛතා නීතිය ක්‍රියාත්මකයි",
            "rule_2_education_guard": "නීතිය 2: අධ්‍යාපන පදනම් නීතිය ක්‍රියාත්මකයි",
            "rule_3_low_confidence": "නීතිය 3: අඩු නිරවද්‍යතා පෙරහන් නීතිය ක්‍රියාත්මකයි",
            "ml_prediction": "සම්මත සංඛ්‍යානමය ආකෘති ප්‍රතිදානය"
        },
        "score_labels": {
            "excellent": "විශිෂ්ට ගැළපීමක්",
            "good": "ප්‍රමාණවත් ගැළපීමක්",
            "fair": "සීමිත ගැළපීමක්",
            "weak": "ප්‍රමාණවත් නොවන ගැළපීමක්"
        }
    }
}

# ── Session State Management ──────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

# ── Sleek Minimalist Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.25rem;">
        <div style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin-bottom: 0.2rem;">Workspace</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">Job Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 0.4rem;">Language / භාෂාව</div>""", unsafe_allow_html=True)
    selected_lang = st.radio(
        "Language",
        options=["English", "සිංහල"],
        index=0 if st.session_state["lang"] == "en" else 1,
        label_visibility="collapsed"
    )
    st.session_state["lang"] = "en" if selected_lang == "English" else "si"
    T = TRANSLATIONS[st.session_state["lang"]]

# ── Two-Tone Professional Dashboard CSS (Zero Emojis, Ultra-Responsive) ───────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Sinhala:wght@400;500;600;700&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Noto Sans Sinhala', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }

    .stApp {
        background-color: #f8fafc;
    }

    /* Two-Tone Header Hero Card */
    .two-tone-header {
        background: #0f172a;
        color: #ffffff;
        border-radius: 8px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #1e293b;
    }

    .two-tone-brand {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }

    .two-tone-title {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        color: #ffffff;
        margin: 0 0 0.4rem 0;
    }

    .two-tone-subtitle {
        font-size: 0.94rem;
        color: #cbd5e1;
        line-height: 1.55;
        margin: 0;
        max-width: 900px;
    }

    /* Structured White Metric Box */
    .dashboard-kpi {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.15s ease;
    }
    .dashboard-kpi:hover {
        border-color: #cbd5e1;
    }

    .dashboard-kpi-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }

    .dashboard-kpi-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
    }

    /* White Content Container */
    .panel-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        color: #1e293b;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Job Card in Two-Tone Tone */
    .job-card-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .job-card-panel:hover {
        border-color: #0f172a;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05);
    }

    .job-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        text-decoration: none;
        letter-spacing: -0.015em;
    }
    .job-card-title:hover {
        color: #2563eb;
        text-decoration: underline;
    }

    .job-meta-flex {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        color: #64748b;
        font-size: 0.84rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }

    /* Two-Tone Badges */
    .badge-solid-dark {
        display: inline-block;
        background: #0f172a;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .badge-subtle-light {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #0f172a;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Attributes and Skill Tags */
    .tag-item {
        display: inline-block;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #334155;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }

    .tag-match {
        display: inline-block;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 3px 2px 0;
    }

    .tag-deficit {
        display: inline-block;
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 3px 2px 0;
    }

    /* Aggregate Stats Box */
    .stats-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        text-align: center;
    }
    .stats-card-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .stats-card-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
    }

    /* Streamlit Primary Button in Two-Tone Navy */
    div.stButton > button[kind="primary"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.25rem !important;
        letter-spacing: 0.01em !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1e293b !important;
        border-color: #1e293b !important;
    }

    /* Streamlit Tabs Customization */
    div[data-baseweb="tab-list"] {
        border-bottom: 1px solid #e2e8f0;
        gap: 2px;
    }
    button[data-baseweb="tab"] {
        padding: 9px 16px;
        font-size: 0.84rem;
        font-weight: 600;
        color: #64748b;
        border-radius: 4px 4px 0 0;
        background-color: transparent;
    }
    button[aria-selected="true"] {
        color: #0f172a !important;
        border-bottom: 2px solid #0f172a !important;
    }

    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .two-tone-header {
            padding: 1.25rem 1.25rem;
            margin-bottom: 1rem;
        }
        .two-tone-title {
            font-size: 1.45rem;
        }
        .two-tone-subtitle {
            font-size: 0.85rem;
        }
        .dashboard-kpi {
            padding: 0.85rem 1rem;
        }
        .job-card-panel {
            padding: 1rem;
        }
        .job-card-title {
            font-size: 1.05rem;
        }
        .job-meta-flex {
            gap: 8px;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Header Two-Tone Hero Card ─────────────────────────────────────────────────
st.markdown(f"""
<div class="two-tone-header">
    <div class="two-tone-brand">{T["nav_brand"]}</div>
    <h1 class="two-tone-title">{T["app_title"]}</h1>
    <p class="two-tone-subtitle">{T["app_subtitle"]}</p>
</div>
""", unsafe_allow_html=True)

# ── Document Submission Section ───────────────────────────────────────────────
with st.container():
    uploaded_file = st.file_uploader(
        T["upload_label"],
        type=["pdf"],
        help=T["upload_help"]
    )

if uploaded_file is not None:
    st.info(f"{T['upload_success']} **{uploaded_file.name}**")

    col_btn, _ = st.columns([2, 5])
    with col_btn:
        run_analysis = st.button(T["btn_analyze"], type="primary", use_container_width=True)

    if run_analysis:
        with st.spinner(T["analyzing_spinner"]):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{API_URL}/api/analyze-resume", files=files)
                res.raise_for_status()
                st.session_state["analysis_data"] = res.json()
            except Exception as e:
                st.error(f"Inference pipeline communication failure: {e}")
                st.stop()

        cv_details = st.session_state["analysis_data"].get("cv_details", {})
        keywords_list = cv_details.get("job_keywords", [])
        recommendation_ok = cv_details.get("recommendation_ok", False)

        if recommendation_ok and keywords_list:
            with st.spinner(T["fetching_jobs_spinner"]):
                payload = {
                    "keywords": keywords_list,
                    "resume_text": st.session_state["analysis_data"].get("resume_text", ""),
                    "cv_details": cv_details
                }
                try:
                    jobs_res = requests.post(f"{API_URL}/api/search-jobs", json=payload)
                    jobs_res.raise_for_status()
                    st.session_state["jobs_data"] = jobs_res.json()
                except Exception as e:
                    st.warning(f"Market data query incomplete: {e}")
                    st.session_state["jobs_data"] = {}
        else:
            st.session_state["jobs_data"] = {}

# ── Analysis Display from Session State ───────────────────────────────────────
if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    resume_text = data.get("resume_text", "")
    summary = data.get("summary", "")
    skill_gaps = data.get("skill_gaps", "")
    roadmap = data.get("roadmap", "")
    cv_details = data.get("cv_details", {})
    detected_skills = data.get("detected_skills", [])
    recommendation_ok = cv_details.get("recommendation_ok", False)
    confidence = cv_details.get("confidence", 0.0)
    rule_applied = cv_details.get("rule_applied", "ml_prediction")
    primary_field = cv_details.get("primary_field", "Unassigned")
    keywords_list = cv_details.get("job_keywords", [])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Row (Two-Tone Responsive Cards) ───────────────────────────────────
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(f"""
        <div class="dashboard-kpi">
            <div class="dashboard-kpi-label">{T['exp_years']}</div>
            <div class="dashboard-kpi-value">{cv_details.get('experience_years', 0):.1f} {T['years']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""
        <div class="dashboard-kpi">
            <div class="dashboard-kpi-label">{T['exp_field']}</div>
            <div class="dashboard-kpi-value">{cv_details.get('experience_field', 'Unspecified')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"""
        <div class="dashboard-kpi">
            <div class="dashboard-kpi-label">{T['pred_domain']}</div>
            <div class="dashboard-kpi-value" style="color:#0f172a;">{primary_field}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_k4:
        st.markdown(f"""
        <div class="dashboard-kpi">
            <div class="dashboard-kpi-label">{T['match_score']}</div>
            <div class="dashboard-kpi-value">{confidence * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs Navigation ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        T["tab_summary"],
        T["tab_gaps"],
        T["tab_roadmap"],
        T["tab_profile"],
        T["tab_skills"]
    ])

    with tab1:
        st.markdown(f"#### {T['summary_title']}")
        st.markdown(f'<div class="panel-box">{summary}</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown(f"#### {T['gaps_title']}")
        st.markdown(f'<div class="panel-box">{skill_gaps}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown(f"#### {T['roadmap_title']}")
        st.markdown(f'<div class="panel-box">{roadmap}</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown(f"#### {T['profile_title']}")
        st.caption(T['profile_caption'])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="dashboard-kpi">
                <div class="dashboard-kpi-label">{T['edu_category']}</div>
                <div class="dashboard-kpi-value">{cv_details.get('education_category', 'Other')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="dashboard-kpi">
                <div class="dashboard-kpi-label">{T['edu_details']}</div>
                <div class="dashboard-kpi-value">{cv_details.get('education', 'Unspecified')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"##### {T['ml_result_title']}")

        conf_badge = f'<span class="badge-solid-dark">{T["high_conf"]} [{confidence * 100:.1f}%]</span>' if confidence >= 0.85 else f'<span class="badge-subtle-light">{T["med_conf"] if confidence >= 0.72 else T["low_conf"]} [{confidence * 100:.1f}%]</span>'
        rule_label = T["rule_labels"].get(rule_applied, rule_applied)

        col_c, col_d = st.columns([2, 1])
        with col_c:
            st.markdown(f"**{T['pred_domain']}:** &nbsp; <span style='font-size:1.05rem; font-weight:700; color:#0f172a;'>{primary_field}</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top:6px;'>{conf_badge}</div>", unsafe_allow_html=True)
        with col_d:
            st.markdown(f"<div style='font-size:0.8rem; color:#475569; padding:8px 12px; background:#f1f5f9; border:1px solid #cbd5e1; border-radius:4px;'>{rule_label}</div>", unsafe_allow_html=True)

        if recommendation_ok and keywords_list:
            st.markdown("---")
            st.markdown(f"##### {T['job_kw_title']}")
            kw_html = " ".join([f'<span class="tag-item" style="font-weight:700; color:#0f172a;">{kw}</span>' for kw in keywords_list])
            st.markdown(kw_html, unsafe_allow_html=True)
        elif not recommendation_ok:
            st.warning(T["low_conf_warning"])

        key_skills = cv_details.get("key_skills", [])
        if key_skills:
            st.markdown("---")
            st.markdown(f"##### {T['llm_skills_title']}")
            skills_html = " ".join([f'<span class="tag-item">{s}</span>' for s in key_skills])
            st.markdown(skills_html, unsafe_allow_html=True)

    with tab5:
        st.markdown(f"#### {T['ml_skills_title']}")
        if detected_skills:
            st.markdown(f"<div style='margin-bottom:12px; font-size:0.88rem; color:#0f172a;'><strong>{len(detected_skills)}</strong> {T['skills_detected_count']}</div>", unsafe_allow_html=True)
            skills_html = " ".join([f'<span class="tag-item">{skill}</span>' for skill in detected_skills])
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.warning(T["no_skills_detected"])

    # ── Market Requisitions Section ───────────────────────────────────────────
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(f"### {T['jobs_title']}")
    st.markdown(f"<p style='color:#64748b; font-size:0.9rem; margin-top:-6px; margin-bottom:1.2rem;'>{T['jobs_ranked_desc']}</p>", unsafe_allow_html=True)

    if not recommendation_ok:
        st.error(T["low_conf_warning"])
    else:
        jobs_data = st.session_state.get("jobs_data", {})
        ranked_jobs = jobs_data.get("ranked_jobs", [])
        filtered_jobs = jobs_data.get("filtered_jobs", [])
        stats = jobs_data.get("stats", {})

        if filtered_jobs:
            st.markdown(f"""
            <div style="background:#f8fafc; border:1px solid #cbd5e1; color:#0f172a; padding:8px 14px; border-radius:4px; font-size:0.84rem; font-weight:600; margin-bottom:1.2rem;">
                {T['found_matches']}: {len(filtered_jobs)} {T['out_of']} {len(ranked_jobs)} {T['fetched']}
            </div>
            """, unsafe_allow_html=True)

            for idx, job in enumerate(filtered_jobs):
                match_data = job["match_data"]
                total_score = match_data["total_score"]

                if total_score >= 85:
                    label = T["score_labels"]["excellent"]
                    badge_markup = f'<span class="badge-solid-dark">{total_score:.1f}% — {label}</span>'
                elif total_score >= 65:
                    label = T["score_labels"]["good"]
                    badge_markup = f'<span class="badge-subtle-light">{total_score:.1f}% — {label}</span>'
                elif total_score >= 45:
                    label = T["score_labels"]["fair"]
                    badge_markup = f'<span class="badge-subtle-light">{total_score:.1f}% — {label}</span>'
                else:
                    label = T["score_labels"]["weak"]
                    badge_markup = f'<span class="badge-subtle-light">{total_score:.1f}% — {label}</span>'

                job_title = job["title"]
                company = job["company"]
                location = job["location"]
                job_url = job["url"]

                st.markdown(f"""
                <div class="job-card-panel">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                        <div>
                            <a href="{job_url}" target="_blank" class="job-card-title">{job_title}</a>
                            <div class="job-meta-flex">
                                <span>{T['company']}: <strong>{company}</strong></span>
                                <span>{T['location']}: <strong>{location}</strong></span>
                            </div>
                        </div>
                        <div>
                            {badge_markup}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(total_score) / 100.0)

                with st.expander(f"{T['breakdown']} — {job_title}"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(T["semantic_metric"], f"{match_data['semantic_score']:.1f}%")
                    with col2:
                        st.metric(T["skills_metric"], f"{match_data['skill_score']:.1f}%")
                    with col3:
                        st.metric(T["edu_metric"], f"{match_data.get('education_score', 50.0):.1f}%")
                    with col4:
                        st.metric(T["exp_metric"], f"{match_data['experience_score']:.1f}%")

                    if match_data.get("matched_skills"):
                        st.markdown(f"**{T['matched_skills']}:**")
                        matched_html = " ".join([f'<span class="tag-match">{s}</span>' for s in match_data["matched_skills"]])
                        st.markdown(matched_html, unsafe_allow_html=True)

                    if match_data.get("missing_skills"):
                        st.markdown(f"**{T['missing_skills']}:**")
                        missing_html = " ".join([f'<span class="tag-deficit">{s}</span>' for s in match_data["missing_skills"]])
                        st.markdown(missing_html, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="margin-top:12px;">
                        <a href="{job_url}" target="_blank" style="display:inline-block; background:#0f172a; color:#ffffff; padding:6px 14px; border-radius:4px; font-size:0.8rem; font-weight:600; text-decoration:none;">
                            {T['view_job_btn']}
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

            # ── Aggregate Statistics ──────────────────────────────────────────
            if stats:
                st.markdown("---")
                st.markdown(f"#### {T['overall_stats']}")

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.markdown(f"""
                    <div class="stats-card">
                        <div class="stats-card-label">{T['avg_score']}</div>
                        <div class="stats-card-value">{stats.get('avg_score', 0):.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_s2:
                    st.markdown(f"""
                    <div class="stats-card">
                        <div class="stats-card-label">{T['high_score']}</div>
                        <div class="stats-card-value">{stats.get('max_score', 0):.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_s3:
                    st.markdown(f"""
                    <div class="stats-card">
                        <div class="stats-card-label">{T['low_score']}</div>
                        <div class="stats-card-value">{stats.get('min_score', 0):.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_s4:
                    st.markdown(f"""
                    <div class="stats-card">
                        <div class="stats-card-label">{T['good_matches']}</div>
                        <div class="stats-card-value">{stats.get('good_matches', 0)} / {stats.get('total_scores', 0)}</div>
                    </div>
                    """, unsafe_allow_html=True)

        elif ranked_jobs:
            st.warning(T["no_jobs_qualified"])
        else:
            st.info(T["no_jobs_found"])
