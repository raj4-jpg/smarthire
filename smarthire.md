# smarthire
import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. TEXT CLEANING & RESUME PARSING
# ==========================================
def clean_text(text: str) -> str:
    """Removes URLs, mentions, symbols, and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def extract_text_from_upload(uploaded_file) -> str:
    """Extracts raw text from uploaded PDF, DOCX, or TXT file."""
    fname = uploaded_file.name.lower()
    
    if fname.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
        
    elif fname.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            extracted_pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            return "\n".join(extracted_pages)
        except Exception as e:
            st.error(f"Error parsing PDF: {e}")
            return ""
            
    elif fname.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"Error parsing DOCX: {e}")
            return ""
            
    return ""

# ==========================================
# 2. IN-MEMORY TRAINING DATA & MODELS
# ==========================================
TECH_SKILLS = [
    "python", "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "sql", "tableau", "powerbi", "r", "docker",
    "kubernetes", "aws", "azure", "jenkins", "terraform", "ci/cd", "linux", "ansible",
    "java", "spring boot", "microservices", "hibernate", "kafka", "mysql", "postgresql",
    "html", "css", "javascript", "react", "figma", "ui/ux", "solidworks", "catia",
    "autocad", "fea", "ansys", "thermodynamics", "recruitment", "talent acquisition",
    "hris", "payroll", "supply chain", "logistics", "six sigma", "operations management"
]

@st.cache_resource
def setup_models_and_corpus():
    """Initializes the dataset, trains the classifier, and prepares the recommender."""
    # Labeled Resumes for Supervised Classifier Training
    training_resumes = [
        ("Data Science", "Experienced Data Scientist skilled in Python, Machine Learning, Deep Learning, Pandas, Scikit-Learn, SQL, and NLP predictive modeling."),
        ("Data Science", "Machine Learning Engineer with expertise in PyTorch, TensorFlow, Computer Vision, MLOps, Data Visualization, and Statistics."),
        ("Java Developer", "Backend Java Developer experienced in Core Java, Spring Boot, Microservices, Hibernate, REST APIs, PostgreSQL, and Docker."),
        ("Java Developer", "Enterprise Software Engineer building high scale Java applications with Spring Cloud, Maven, Kafka, and Kubernetes."),
        ("Web Designing", "Creative Web Designer & Frontend Developer skilled in HTML5, CSS3, JavaScript, React, Figma, UI/UX, and Responsive Design."),
        ("Web Designing", "UI/UX Designer creating wireframes, design systems, and frontend prototypes using Figma, Adobe XD, HTML, CSS, and Tailwind."),
        ("DevOps Engineer", "Cloud DevOps Specialist experienced with AWS, Azure, Docker, Kubernetes, CI/CD pipelines, Terraform, Jenkins, and Linux."),
        ("DevOps Engineer", "Site Reliability Engineer focused on Infrastructure as Code, Prometheus, Grafana, Ansible, Docker, and automation scripting."),
        ("Mechanical Engineer", "Mechanical Design Engineer skilled in SolidWorks, CATIA, AutoCAD, FEA stress simulation, ANSYS, GD&T, and thermodynamics."),
        ("Mechanical Engineer", "Thermal and Manufacturing Engineer experienced in 3D CAD modeling, heat transfer, prototype fabrication, and CNC machining."),
        ("HR", "Talent Acquisition Specialist with expertise in recruitment lifecycle, sourcing, candidate screening, onboarding, and HR policies."),
        ("Operations Manager", "Operations Manager leading supply chain optimization, Lean Six Sigma, logistics, vendor management, and workflow automation.")
    ]
    
    # 1. Train Supervised Category Classifier
    df_train = pd.DataFrame(training_resumes, columns=["Category", "Text"])
    df_train["Clean"] = df_train["Text"].apply(clean_text)
    
    vec_classifier = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train = vec_classifier.fit_transform(df_train["Clean"])
    y_train = df_train["Category"]
    
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, y_train)
    
    # 2. Unified Job Corpus for Unsupervised Recommendation
    job_postings = [
        {"title": "Data Scientist", "company": "TechCorp Analytics", "location": "Bengaluru", "exp": "1-3 yrs", "skills": "Python, Machine Learning, SQL, Pandas, Scikit-Learn, NLP", "desc": "Develop statistical algorithms and predictive pipelines for enterprise data intelligence."},
        {"title": "Machine Learning Engineer", "company": "AI Labs Global", "location": "Hyderabad", "exp": "2-4 yrs", "skills": "Python, PyTorch, Deep Learning, MLOps, Docker, AWS", "desc": "Deploy machine learning models into production environments with low latency."},
        {"title": "Java Backend Engineer", "company": "FinTech Systems", "location": "Pune", "exp": "2-5 yrs", "skills": "Java, Spring Boot, Microservices, PostgreSQL, Kafka, Docker", "desc": "Design and build scalable financial payment gateways using modern microservices."},
        {"title": "Frontend Web Designer", "company": "Creative Media", "location": "Remote", "exp": "1-3 yrs", "skills": "HTML, CSS, JavaScript, React, Figma, UI/UX", "desc": "Translate product wireframes into engaging, responsive web user interfaces."},
        {"title": "Cloud DevOps Engineer", "company": "InnoCloud Services", "location": "Gurugram", "exp": "3-5 yrs", "skills": "AWS, Docker, Kubernetes, Terraform, Jenkins, CI/CD", "desc": "Manage cloud infrastructure automation and build continuous integration delivery pipelines."},
        {"title": "Mechanical Design Engineer", "company": "Precision Engineering", "location": "Chennai", "exp": "1-4 yrs", "skills": "SolidWorks, CAD, AutoCAD, FEA, ANSYS, GD&T", "desc": "Model 3D components, perform thermal analysis, and prepare production mechanical blueprints."},
        {"title": "HR Talent Acquisition Partner", "company": "Apex Global", "location": "Mumbai", "exp": "2-4 yrs", "skills": "Recruitment, Sourcing, Screening, Onboarding, HRIS", "desc": "Oversee recruitment lifecycle and coordinate cross-functional campus hiring drives."},
        {"title": "Supply Chain & Operations Lead", "company": "LogiTech Solutions", "location": "Delhi NCR", "exp": "3-6 yrs", "skills": "Supply Chain, Logistics, Operations Management, Six Sigma", "desc": "Streamline warehouse fulfillment pipelines, track operational SLAs, and reduce overhead costs."}
    ]
    df_jobs = pd.DataFrame(job_postings)
    df_jobs["clean_text"] = (df_jobs["title"] + " " + df_jobs["skills"] + " " + df_jobs["desc"]).apply(clean_text)
    
    # 3. Vectorize Job Corpus
    vec_recommender = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    corpus_vectors = vec_recommender.fit_transform(df_jobs["clean_text"])
    
    return classifier, vec_classifier, df_jobs, vec_recommender, corpus_vectors

# ==========================================
# 3. SKILL GAP ENGINE
# ==========================================
def compute_skill_gap(resume_text: str, target_skills_str: str):
    """Calculates matched skills and missing skills via set difference."""
    resume_lower = resume_text.lower()
    candidate_skills = {
        s.title() for s in TECH_SKILLS 
        if re.search(r"\b" + re.escape(s) + r"\b", resume_lower)
    }
    
    job_skills_lower = target_skills_str.lower()
    required_skills = {
        s.title() for s in TECH_SKILLS 
        if re.search(r"\b" + re.escape(s) + r"\b", job_skills_lower)
    }
    
    if not required_skills:
        required_skills = {s.strip().title() for s in target_skills_str.split(",") if s.strip()}
        
    matched = sorted(list(required_skills.intersection(candidate_skills)))
    missing = sorted(list(required_skills - candidate_skills))
    
    return matched, missing

# ==========================================
# 4. STREAMLIT INTERACTIVE DASHBOARD
# ==========================================
st.set_page_config(page_title="SmartHire - ML Resume Matcher", page_icon="💼", layout="wide")

st.title("💼 SmartHire: Resume Matching & Career Guidance")
st.markdown("A classical Machine Learning portal providing resume classification, job matching, and skill-gap recommendations.")

classifier, vec_classifier, df_jobs, vec_recommender, corpus_vectors = setup_models_and_corpus()

uploaded_file = st.file_uploader("Upload Candidate Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    raw_resume = extract_text_from_upload(uploaded_file)
    cleaned_resume = clean_text(raw_resume)
    
    if not cleaned_resume.strip():
        st.warning("Could not extract readable text from the uploaded document. Please check the file.")
    else:
        st.success("Resume parsed successfully!")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Domain Classification", "🔍 Top Matching Jobs", "📊 Skill-Gap Report"])
        
        # TAB 1: Supervised Classifier
        with tab1:
            st.subheader("Predicted Career Domain (Supervised ML)")
            res_vec_clf = vec_classifier.transform([cleaned_resume])
            pred_domain = classifier.predict(res_vec_clf)[0]
            confidence = max(classifier.predict_proba(res_vec_clf)[0]) * 100
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Predicted Domain", pred_domain)
                st.metric("Model Confidence", f"{confidence:.1f}%")
            with c2:
                st.info(f"The multi-class Logistic Regression model mapped your profile to **{pred_domain}**.")
                
        # TAB 2: Unsupervised Recommender Engine
        with tab2:
            st.subheader("Top-N Job Recommendations (TF-IDF & Cosine Similarity)")
            res_vec_rec = vec_recommender.transform([cleaned_resume])
            sim_scores = cosine_similarity(res_vec_rec, corpus_vectors)[0]
            
            top_indices = sim_scores.argsort()[::-1][:5]
            
            for rank, idx in enumerate(top_indices):
                job = df_jobs.iloc[idx]
                score = round(float(sim_scores[idx]) * 100, 1)
                
                with st.expander(f"#{rank+1}: {job['title']} — {job['company']} (Match: {score}%)", expanded=(rank==0)):
                    st.write(f"📍 **Location:** {job['location']} | ⏳ **Experience:** {job['exp']}")
                    st.write(f"🛠️ **Required Skills:** `{job['skills']}`")
                    st.write(f"📝 **Overview:** {job['desc']}")
                    st.progress(min(score / 100.0, 1.0))
                    
        # TAB 3: Skill Gap Analysis
        with tab3:
            st.subheader("Skill-Gap & Upskilling Guidance")
            best_job = df_jobs.iloc[top_indices[0]]
            matched_skills, missing_skills = compute_skill_gap(raw_resume, best_job["skills"])
            
            st.markdown(f"#### Benchmark Target: **{best_job['title']}** at {best_job['company']}")
            
            col_match, col_miss = st.columns(2)
            with col_match:
                st.markdown("##### ✅ Skills Present in Resume")
                if matched_skills:
                    for s in matched_skills:
                        st.success(f"✓ {s}")
                else:
                    st.warning("No explicit target skills detected.")
                    
            with col_miss:
                st.markdown("##### ⚠️ Missing In-Demand Skills")
                if missing_skills:
                    for s in missing_skills:
                        st.error(f"✗ {s}")
                else:
                    st.balloons()
                    st.success("Great match! Your resume covers all key skills.")
                    
            st.markdown("---")
            if missing_skills:
                st.info(f"💡 **Recommendation:** Prioritize projects or certifications in: **{', '.join(missing_skills)}**.")
