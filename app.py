import streamlit as st
from src.helper import extract_text_from_pdf, get_llm_response
from src.job_api import get_linkedin_jobs # අලුතෙන් එකතු කරපු API එක

# Mobile responsiveness සහ page config එක හදාගැනීම (Phone එකේ ලස්සනට පේන්න)
st.set_page_config(page_title="Job Recommender", page_icon="🎓", layout="centered")

st.title("🎓 Job Recommender System")
st.write("Upload your Resume (PDF) here. We will analyze it to find matching jobs and identify your skill gaps!")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file is not None:
    st.success("Resume uploaded successfully!")
    
    if st.button("Analyze Resume"):
        with st.spinner("Analyzing resume... Please wait."):
            
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if resume_text:
                st.success("Text extraction successful! Starting AI Analysis...")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🎯 Skill Gaps", "🗺️ Roadmap", "🔑 Job Keywords"])
                
                with tab1:
                    st.subheader("Resume Summary")
                    summary_prompt = f"Act as an expert HR recruiter. Provide a highly professional summary of the following resume in 3-4 sentences. Highlight the main skills, educational background, and experience without assuming any specific industry. Resume text: {resume_text}"
                    summary = get_llm_response(summary_prompt)
                    st.write(summary)
                
                with tab2:
                    st.subheader("Skill Gap & Missing Areas")
                    skill_gap_prompt = f"Analyze this resume against the relevant industry standards based purely on the candidate's actual background. Identify specific skill gaps and suggest practical ways to improve them for their specific field. Resume text: {resume_text}"
                    skill_gaps = get_llm_response(skill_gap_prompt)
                    st.write(skill_gaps)
                
                with tab3:
                    st.subheader("Future Roadmap & Preparation Strategy")
                    roadmap_prompt = f"""You are an expert career counselor. Based strictly on the candidate's actual background in the resume, provide a realistic, step-by-step career roadmap. 
                    STRICT RULES: 1. DO NOT suggest completing educational qualifications that are already mentioned. 2. Focus on relevant career advancement, upskilling, and professional development specific to their actual field. Resume text: {resume_text}"""
                    roadmap = get_llm_response(roadmap_prompt)
                    st.write(roadmap)
                    
                with tab4:
                    st.subheader("Extracted Job Keywords")
                    keyword_prompt = f"""You are a career expert. Carefully read the resume below and identify the top 3 to 5 most suitable job titles for this specific candidate.
                    Base your answer STRICTLY on their actual degree, subjects studied, skills, and any work experience mentioned in the resume.
                    DO NOT suggest generic or unrelated job titles. For example, do NOT suggest accounting or finance roles unless the resume explicitly mentions finance qualifications.
                    STRICT OUTPUT RULE: Return ONLY a comma-separated list of job titles. No sentences, no explanations, no bullet points.
                    Resume text: {resume_text}"""
                    keywords = get_llm_response(keyword_prompt)
                    st.info(f"Keywords: {keywords}")

                # --- අලුතෙන් එකතු කරපු LinkedIn Jobs හොයන කොටස ---
                st.divider()
                st.subheader("💼 Recommended Job Opportunities (LinkedIn)")
                
                if keywords and keywords.strip() != "":
                    # Keywords ලිස්ට් එකෙන් පළවෙනි Job Title එක ගන්නවා
                    first_keyword = keywords.split(',')[0].strip()
                    st.write(f"Searching for **'{first_keyword}'** jobs in Sri Lanka...")
                    
                    with st.spinner("Fetching data from LinkedIn via Apify... (This may take a few seconds)"):
                        jobs_list = get_linkedin_jobs(first_keyword)
                        
                        if jobs_list and len(jobs_list) > 0:
                            st.success(f"Found {len(jobs_list)} job opportunities!")
                            
                            for job in jobs_list:
                                job_title = job['title']
                                company = job['company']
                                location = job['location']
                                job_url = job['url']
                                
                                st.markdown(f"### 🔹 [{job_title}]({job_url})")
                                st.write(f"🏢 **Company:** {company}")
                                st.write(f"📍 **Location:** {location}")
                                st.write("---")
                        else:
                            st.warning("No jobs found for this role at the moment. Please try another Resume.")
            else:
                # PDF එකෙන් text එක ගන්න බැරි වුණොත් පෙන්වන error එක
                st.error("Failed to extract text from the PDF. Please try a different file.")