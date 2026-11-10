import os
import re
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="SmartHire AI - Advanced Career Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern corporate UI
st.markdown("""
<style>
    .metric-card {
        background-color: #1E2229;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #30363D;
    }
    .skill-badge-found {
        background-color: #1B4D3E;
        color: #A3E635;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px;
        font-weight: 500;
    }
    .skill-badge-missing {
        background-color: #4C1D24;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 3px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. PREPROCESSING & PARSING MODULE
# =========================================================
def clean_text(text: str) -> str:
    """Sanitizes text by stripping links, handles, special characters, and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+\s*", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"""[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]""", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def extract_text_from_upload(uploaded_file) -> str:
    """Extracts text stream from PDF, DOCX, or plain text formats."""
    fname = uploaded_file.name.lower()
    if fname.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif fname.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
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

# =========================================================
# 2. ADVANCED DATASET, MODELS & CLUSTERING INITIALIZATION
# =========================================================
TECHNICAL_LEXICON = [
    "python", "r", "sql", "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "tableau", "powerbi", "docker", "kubernetes", "aws",
    "azure", "gcp", "jenkins", "terraform", "ci/cd", "linux", "ansible", "java", "spring boot",
    "microservices", "hibernate", "kafka", "mysql", "postgresql", "mongodb", "html", "css",
    "javascript", "typescript", "react", "next.js", "figma", "ui/ux", "solidworks", "catia",
    "autocad", "ansys", "fea", "thermodynamics", "gd&t", "recruitment", "talent acquisition",
    "hris", "payroll", "supply chain", "logistics", "six sigma", "operations management",
    "cybersecurity", "penetration testing", "siem", "firewall", "flutter", "swift", "kotlin"
]

@st.cache_resource
def build_engine():
    # 1. Supervised Training Data for Category Classifier (Model A)
    labeled_corpus = [
        ("Data Science", "Data Scientist proficient in Python, machine learning, deep learning, NLP, TensorFlow, Scikit-Learn, Pandas, SQL, Tableau, data visualization, predictive modeling."),
        ("Data Science", "Machine Learning Specialist skilled in PyTorch, algorithms, predictive analytics, statistics, data pipelines, big data, MLOps, classification, regression."),
        ("Java Developer", "Senior Java Developer skilled in Java 17, Spring Boot, Microservices, Hibernate, REST APIs, PostgreSQL, Kafka, multithreading, Docker, Maven."),
        ("Java Developer", "Backend Software Engineer experienced with Core Java, J2EE, Spring Cloud, relational databases, unit testing, JUnit, CI/CD, Kubernetes."),
        ("Web Designing", "Creative Web Designer & Frontend Specialist in HTML5, CSS3, JavaScript, React, Figma, Adobe XD, UI/UX design, wireframing, responsive mobile-first layouts."),
        ("Web Designing", "Frontend UI/UX Developer experienced in Tailwind CSS, Bootstrap, Figma prototypes, usability testing, TypeScript, modern web styling."),
        ("DevOps Engineer", "Cloud DevOps Engineer with strong hands-on expertise in AWS, Docker, Kubernetes, Terraform, Jenkins, CI/CD pipelines, Linux administration, Ansible, monitoring."),
        ("DevOps Engineer", "Site Reliability Engineer skilled in infrastructure automation, Prometheus, Grafana, cloud architecture, shell scripting, Docker containers."),
        ("Mechanical Engineer", "Mechanical Design Engineer with expertise in CAD, SolidWorks, CATIA, AutoCAD, finite element analysis (FEA), ANSYS, GD&T, 3D modeling, thermodynamics."),
        ("Mechanical Engineer", "Thermal & Simulation Engineer proficient in SolidWorks, heat transfer, prototype stress analysis, fluid mechanics, CNC manufacturing."),
        ("HR", "Human Resources Specialist experienced in talent acquisition, recruitment lifecycle, employee relations, onboarding, HR policies, HRIS, payroll."),
        ("Operations Manager", "Operations & Supply Chain Manager experienced in logistics, Lean Six Sigma, process improvement, vendor management, ERP systems, KPI monitoring."),
        ("Cybersecurity", "Information Security Analyst proficient in cybersecurity, penetration testing, SIEM, network defense, firewalls, vulnerability assessment, Linux, incident response."),
        ("Mobile Developer", "Mobile App Developer skilled in Flutter, Swift, Kotlin, iOS and Android architecture, REST API integration, state management, mobile UI design.")
    ]
    
    df_train = pd.DataFrame(labeled_corpus, columns=["Category", "Text"])
    df_train["Clean"] = df_train["Text"].apply(clean_text)
    
    vec_clf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train = vec_clf.fit_transform(df_train["Clean"])
    classifier = LogisticRegression(max_iter=1000, C=2.0)
    classifier.fit(X_train, df_train["Category"])
    
    # 2. Comprehensive Merged Job Corpus
    jobs_data = [
        {"id": 1, "title": "Senior Data Scientist", "company": "Kaggle Labs", "location": "Bengaluru", "exp": "2-5 yrs", "skills": "Python, Machine Learning, Scikit-Learn, Deep Learning, SQL, NLP, Tableau", "desc": "Design end-to-end predictive pipelines, train classification models, and communicate insights to leadership.", "domain": "Data Science"},
        {"id": 2, "title": "MLOps & AI Engineer", "company": "NeuralScale Systems", "location": "Hyderabad", "exp": "3-6 yrs", "skills": "Python, PyTorch, Docker, Kubernetes, MLOps, AWS, CI/CD", "desc": "Containerize, deploy, and scale deep learning algorithms in production with automated pipelines.", "domain": "Data Science"},
        {"id": 3, "title": "Java Microservices Lead", "company": "Enterprise FinTech", "location": "Pune", "exp": "3-6 yrs", "skills": "Java, Spring Boot, Microservices, Kafka, PostgreSQL, Docker, Hibernate", "desc": "Architect resilient transactional microservices supporting high concurrency and low latency operations.", "domain": "Java Developer"},
        {"id": 4, "title": "Fullstack Java Engineer", "company": "CloudPlatform Inc", "location": "Bengaluru", "exp": "1-3 yrs", "skills": "Java, Spring Boot, React, MySQL, REST APIs, Git", "desc": "Develop and maintain robust web backends and integrate with responsive web frontends.", "domain": "Java Developer"},
        {"id": 5, "title": "UI/UX & Web Designer", "company": "Studio Pixelcraft", "location": "Remote", "exp": "1-4 yrs", "skills": "Figma, HTML, CSS, JavaScript, React, UI/UX, Wireframing", "desc": "Produce intuitive user flows, maintain scalable component libraries, and build interactive client interfaces.", "domain": "Web Designing"},
        {"id": 6, "title": "Cloud DevOps Engineer", "company": "InnoCloud Tech", "location": "Hyderabad", "exp": "2-5 yrs", "skills": "AWS, Docker, Kubernetes, Terraform, Jenkins, Linux, CI/CD", "desc": "Automate multi-region cloud infrastructures using Infrastructure as Code (IaC) and maintain CI/CD pipelines.", "domain": "DevOps Engineer"},
        {"id": 7, "title": "CAD Mechanical Design Specialist", "company": "Apex Mobility Ltd", "location": "Chennai", "exp": "2-4 yrs", "skills": "SolidWorks, AutoCAD, FEA, ANSYS, GD&T, CAD, Thermodynamics", "desc": "Engineer structural chassis elements, evaluate thermal and structural stresses, and build prototyping blueprints.", "domain": "Mechanical Engineer"},
        {"id": 8, "title": "Talent Acquisition Partner", "company": "PeopleEdge HR", "location": "Mumbai", "exp": "1-4 yrs", "skills": "Recruitment, Sourcing, Talent Acquisition, Screening, HRIS, Onboarding", "desc": "Lead full lifecycle technical recruiting, build candidate pipelines, and coordinate hiring initiatives.", "domain": "HR"},
        {"id": 9, "title": "Supply Chain Operations Lead", "company": "SwiftCargo Global", "location": "Delhi NCR", "exp": "3-7 yrs", "skills": "Operations Management, Supply Chain, Logistics, Six Sigma, ERP", "desc": "Optimize warehouse inventory turnover, reduce fulfillment turnaround, and enforce Six Sigma standards.", "domain": "Operations Manager"},
        {"id": 10, "title": "Cybersecurity & Threat Analyst", "company": "CyberShield Defense", "location": "Bengaluru", "exp": "2-5 yrs", "skills": "Cybersecurity, Penetration Testing, Linux, SIEM, Firewalls, Python", "desc": "Monitor network perimeters, execute vulnerability audits, and remediate intrusion vectors.", "domain": "Cybersecurity"}
    ]
    df_corpus = pd.DataFrame(jobs_data)
    df_corpus["clean_text"] = (df_corpus["title"] + " " + df_corpus["skills"] + " " + df_corpus["desc"]).apply(clean_text)
    
    # 3. TF-IDF & K-Means Unsupervised Clustering
    vec_rec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    corpus_vectors = vec_rec.fit_transform(df_corpus["clean_text"])
    
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_corpus["cluster"] = kmeans.fit_predict(corpus_vectors)
    
    # PCA 2D coordinates for visual mapping
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(corpus_vectors.toarray())
    df_corpus["pca_x"] = pca_coords[:, 0]
    df_corpus["pca_y"] = pca_coords[:, 1]
    
    # 4. Supervised Fit / Shortlisting Predictor (Model B)
    # Feature inputs: [Jaccard Overlap, Cosine Similarity, Length Delta]
    np.random.seed(42)
    synth_X, synth_y = [], []
    for _ in range(250):
        jaccard = np.random.uniform(0.05, 0.70)
        cos_sim = np.random.uniform(0.10, 0.85)
        len_diff = np.random.randint(10, 200)
        # Ground truth rule: higher similarity and overlap predict interview shortlisting
        shortlisted = 1 if (cos_sim > 0.38 and jaccard > 0.18) else 0
        synth_X.append([jaccard, cos_sim, len_diff])
        synth_y.append(shortlisted)
        
    fit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
    fit_predictor.fit(synth_X, synth_y)
    
    return classifier, vec_clf, df_corpus, vec_rec, corpus_vectors, kmeans, pca, fit_predictor

classifier, vec_clf, df_corpus, vec_rec, corpus_vectors, kmeans, pca, fit_predictor = build_engine()

# =========================================================
# 3. FEATURE EXTRACTION & GAP COMPUTATION
# =========================================================
def analyze_skill_gap(candidate_text: str, target_skills_str: str):
    """Computes exact lexical matching and missing competencies."""
    cand_lower = candidate_text.lower()
    found_skills = {s.title() for s in TECHNICAL_LEXICON if re.search(r"\b" + re.escape(s) + r"\b", cand_lower)}
    
    req_skills = {s.strip().title() for s in target_skills_str.split(",") if s.strip()}
    
    matched = sorted(list(req_skills.intersection(found_skills)))
    missing = sorted(list(req_skills - found_skills))
    return matched, missing, sorted(list(found_skills))

def calculate_fit_score(resume_clean: str, job_clean: str, job_vec, candidate_vec):
    """Calculates Model B feature vector and shortlisting probability."""
    r_words = set(resume_clean.split())
    j_words = set(job_clean.split())
    
    jaccard = len(r_words.intersection(j_words)) / max(len(r_words.union(j_words)), 1)
    cos_sim = float(cosine_similarity(candidate_vec, job_vec)[0][0])
    len_diff = abs(len(r_words) - len(j_words))
    
    prob = fit_predictor.predict_proba([[jaccard, cos_sim, len_diff]])[0][1]
    return round(prob * 100, 1), round(cos_sim * 100, 1)

# =========================================================
# 4. SIDEBAR CONTROLS & RESUME UPLOAD
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("SmartHire Pro")
    st.caption("Classical ML-Powered Matching & Shortlisting Engine")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Upload Candidate CV", type=["pdf", "docx", "txt"])
    
    st.markdown("### 🎛️ Matching Filters")
    selected_locations = st.multiselect(
        "Preferred Locations", 
        options=sorted(df_corpus["location"].unique()), 
        default=sorted(df_corpus["location"].unique())
    )
    top_k = st.slider("Max Job Recommendations", min_value=3, max_value=8, value=5)
    st.markdown("---")
    st.caption("⚡ Model A: TF-IDF + Logistic Regression\n\n🔍 Unsupervised: Cosine Vector Space\n\n📊 Clustering: K-Means & PCA\n\n🎯 Model B: Fit Predictor (Random Forest)")

# =========================================================
# 5. DASHBOARD MAIN VIEW
# =========================================================
st.title("🚀 SmartHire: Resume Intelligence & Shortlisting Portal")
st.markdown("Automated candidate domain classification, Top-N role retrieval, and predictive hiring analysis without generative AI.")

if uploaded_file is None:
    st.info("👋 Upload a resume in PDF, DOCX, or TXT format using the sidebar to begin analysis.")
    
    # Preview current job database
    st.markdown("### 📋 Active Openings in System Database")
    st.dataframe(
        df_corpus[["title", "company", "location", "exp", "skills"]].rename(
            columns={"title": "Role Title", "company": "Company", "location": "Location", "exp": "Experience", "skills": "Key Skills"}
        ),
        use_container_width=True
    )
else:
    raw_resume = extract_text_from_upload(uploaded_file)
    cleaned_resume = clean_text(raw_resume)
    
    if not cleaned_resume.strip():
        st.error("Could not extract readable text from the file. Please check file formatting.")
    else:
        # Compute Resume Vectors
        res_clf_vec = vec_clf.transform([cleaned_resume])
        res_rec_vec = vec_rec.transform([cleaned_resume])
        
        # 1. Classification
        pred_category = classifier.predict(res_clf_vec)[0]
        cat_probs = classifier.predict_proba(res_clf_vec)[0]
        confidence = float(max(cat_probs)) * 100
        
        # 2. Recommender & Shortlisting
        sim_scores = cosine_similarity(res_rec_vec, corpus_vectors)[0]
        
        # Filter by selected locations
        filtered_indices = df_corpus[df_corpus["location"].isin(selected_locations)].index
        ranked_indices = [idx for idx in sim_scores.argsort()[::-1] if idx in filtered_indices][:top_k]
        
        # Header Metrics Banner
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Identified Domain", pred_category)
        with m2:
            st.metric("Model A Confidence", f"{confidence:.1f}%")
        with m3:
            top_match_pct = round(sim_scores[ranked_indices[0]] * 100, 1) if ranked_indices else 0
            st.metric("Top Role Match", f"{top_match_pct}%")
        with m4:
            st.metric("Active Postings Scanned", f"{len(df_corpus)}")
            
        st.markdown("---")
        
        # Tabs Layout
        tab_rec, tab_gap, tab_cluster, tab_cv = st.tabs([
            "🎯 Top Job Recommendations", 
            "📈 Skill Gap & Mentorship", 
            "🌐 K-Means & PCA Visualizer", 
            "📄 Candidate Text Extraction"
        ])
        
        # =========================================================
        # TAB 1: RECOMMENDATIONS & SHORTLISTING PREDICTOR
        # =========================================================
        with tab_rec:
            st.subheader("Ranked Job Openings & Predictive Fit Evaluation")
            
            if not ranked_indices:
                st.warning("No openings match your current location filters. Adjust filters in the sidebar.")
            else:
                for rank, idx in enumerate(ranked_indices):
                    job = df_corpus.iloc[idx]
                    job_vec = corpus_vectors[idx]
                    
                    fit_score, match_score = calculate_fit_score(cleaned_resume, job["clean_text"], job_vec, res_rec_vec)
                    
                    badge_color = "🟢 High Probability" if fit_score >= 60 else ("🟡 Moderate" if fit_score >= 35 else "🔴 Low Probability")
                    
                    with st.expander(f"#{rank+1}: {job['title']} — {job['company']} (Match: {match_score}%)", expanded=(rank==0)):
                        c_left, c_right = st.columns([2, 1])
                        with c_left:
                            st.markdown(f"📍 **Location:** `{job['location']}` | ⏳ **Experience Required:** `{job['exp']}`")
                            st.markdown(f"💼 **Domain Family:** `{job['domain']}`")
                            st.markdown(f"🛠️ **Mandatory Skills:** `{job['skills']}`")
                            st.write(f"📝 **Role Overview:** {job['desc']}")
                        with c_right:
                            st.markdown("##### Model B Fit Evaluation")
                            st.metric("Shortlist Probability", f"{fit_score}%")
                            st.progress(fit_score / 100.0)
                            st.caption(f"Estimated Outcome: **{badge_color}**")
                            
        # =========================================================
        # TAB 2: SKILL GAP REPORT & MENTORSHIP
        # =========================================================
        with tab_gap:
            st.subheader("Skill Gap Audit & Career Growth Recommendations")
            
            benchmark_job = df_corpus.iloc[ranked_indices[0]] if ranked_indices else df_corpus.iloc[0]
            st.markdown(f"Targeting Benchmark: **{benchmark_job['title']}** at **{benchmark_job['company']}**")
            
            matched, missing, candidate_all = analyze_skill_gap(raw_resume, benchmark_job["skills"])
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("#### ✅ Validated Competencies")
                if matched:
                    badges = "".join([f'<span class="skill-badge-found">✓ {s}</span>' for s in matched])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.warning("No explicit target skills detected in resume.")
                    
            with col_g2:
                st.markdown("#### ⚠️ High-Priority Skill Deficits")
                if missing:
                    badges = "".join([f'<span class="skill-badge-missing">✗ {s}</span>' for s in missing])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.success("🎉 Comprehensive alignment! Your resume covers all required skills.")
                    
            st.markdown("---")
            st.markdown("#### 💡 Rule-Based Career Guidance")
            if missing:
                st.write(
                    f"To optimize your profile for **{benchmark_job['title']}**, focus on hands-on project implementations demonstrating proficiency in: **{', '.join(missing)}**."
                )
            else:
                st.write("Your technical qualifications show strong role coverage. Highlight quantifiable performance outcomes and project impacts in your resume.")
                
            # Audit Report Download
            report_text = f"""SMARTHIRE CANDIDATE AUDIT REPORT
=====================================
Predicted Role: {pred_category} (Confidence: {confidence:.1f}%)
Benchmark Job: {benchmark_job['title']} - {benchmark_job['company']}
Match Score: {round(sim_scores[benchmark_job.name]*100, 1)}%

VALIDATED SKILLS:
{', '.join(matched) if matched else 'None identified'}

MISSING SKILLS TO ACQUIRE:
{', '.join(missing) if missing else 'None! Profile matches role criteria.'}
=====================================
"""
            st.download_button(
                label="📥 Download Skill-Gap Audit (TXT)",
                data=report_text,
                file_name="SmartHire_Skill_Report.txt",
                mime="text/plain"
            )

        # =========================================================
        # TAB 3: K-MEANS & PCA CLUSTERING
        # =========================================================
        with tab_cluster:
            st.subheader("Unsupervised Role Clustering & Candidate Vector Projection")
            st.caption("Visualizing 2D PCA projection of TF-IDF vectors across job postings, highlighting your resume's spatial position.")
            
            # Project candidate resume onto PCA coordinate space
            cand_pca = pca.transform(res_rec_vec.toarray())
            
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.scatterplot(
                data=df_corpus, 
                x="pca_x", 
                y="pca_y", 
                hue="domain", 
                style="domain", 
                s=120, 
                alpha=0.85, 
                ax=ax
            )
            
            # Mark candidate location
            ax.scatter(
                cand_pca[0, 0], 
                cand_pca[0, 1], 
                color="red", 
                s=280, 
                marker="*", 
                edgecolors="black", 
                linewidth=1.5, 
                label="⭐ Your Resume Position"
            )
            
            ax.set_title("Job Space Embedding (PCA Dimensions)", fontsize=13)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            st.pyplot(fig)

        # =========================================================
        # TAB 4: EXTRACTED TEXT
        # =========================================================
        with tab_cv:
            st.subheader("Normalized Resume Text")
            st.text_area("Extracted Stream Preview", raw_resume, height=350)
