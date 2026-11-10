import os
import re
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

# Set Wide Page Configuration
st.set_page_config(
    page_title="SmartHire AI - Career Matching Engine",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 1. LUXURY CREAM & BLUSH PINK DESIGN SYSTEM (CSS)
# =========================================================
st.markdown("""
<style>
    /* Global Background: Warm Cream */
    .stApp {
        background-color: #FAF6F0;
        color: #2D2424;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Top Header Banner */
    .main-header {
        background: linear-gradient(135deg, #FFE4E8 0%, #FFD1DC 50%, #FFF3EE 100%);
        border: 1px solid #FBCFE8;
        padding: 26px 32px;
        border-radius: 18px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(244, 114, 182, 0.08);
    }
    .main-header h1 {
        color: #881337;
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #9F1239;
        margin-top: 6px;
        font-size: 1.05rem;
    }

    /* Metric & Information Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFDF9;
        border: 1px solid #FCE7F3;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(244, 63, 94, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #9F1239 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #4C0519 !important;
        font-weight: 800;
    }

    /* Skill Badges */
    .badge-matched {
        background-color: #FCE7F3;
        color: #9D174D;
        border: 1px solid #F472B6;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }
    .badge-missing {
        background-color: #FFF1F2;
        color: #BE123C;
        border: 1px dashed #FB7185;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px;
    }

    /* Expander Container Styling */
    .streamlit-expanderHeader {
        background-color: #FFFDF9 !important;
        border-radius: 12px !important;
        color: #881337 !important;
        font-weight: 600 !important;
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #FFF9F3;
        border-right: 1px solid #FCE7F3;
    }

    /* Tab Selected Styling */
    button[data-baseweb="tab"] {
        color: #881337 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. TEXT PROCESSING & FILE EXTRACTOR
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
    """Extracts plain text stream from PDF, DOCX, or plain text formats."""
    fname = uploaded_file.name.lower()
    if fname.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif fname.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    elif fname.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    return ""

# =========================================================
# 3. LEXICON, MODELS & UNIFIED JOB CORPUS
# =========================================================
SKILL_DIMENSIONS = {
    "Programming": ["python", "r", "java", "javascript", "typescript", "c++", "sql"],
    "Machine Learning": ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "scikit-learn", "statistics"],
    "Data Engineering": ["pandas", "numpy", "kafka", "mysql", "postgresql", "mongodb", "etl"],
    "Cloud & DevOps": ["docker", "kubernetes", "aws", "azure", "jenkins", "terraform", "ci/cd", "linux"],
    "Frontend & Design": ["react", "html", "css", "figma", "ui/ux", "wireframing", "tailwind"],
    "Mechanical & Systems": ["solidworks", "catia", "autocad", "fea", "ansys", "thermodynamics", "gd&t"]
}

FLAT_LEXICON = [skill for skills in SKILL_DIMENSIONS.values() for skill in skills]

@st.cache_resource
def build_engine():
    # 1. Supervised Training Data for Category Classifier (Model A)
    labeled_corpus = [
        ("Data Science", "Data Scientist skilled in Python, machine learning, deep learning, NLP, TensorFlow, Scikit-Learn, Pandas, SQL, Tableau, data visualization, predictive modeling."),
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
        ("Operations Manager", "Operations & Supply Chain Manager experienced in logistics, Lean Six Sigma, process improvement, vendor management, ERP systems, KPI monitoring.")
    ]
    
    df_train = pd.DataFrame(labeled_corpus, columns=["Category", "Text"])
    df_train["Clean"] = df_train["Text"].apply(clean_text)
    
    vec_clf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train = vec_clf.fit_transform(df_train["Clean"])
    classifier = LogisticRegression(max_iter=1000, C=2.0)
    classifier.fit(X_train, df_train["Category"])
    
    # 2. Unified Job Corpus with Salary Bands
    jobs_data = [
        {"id": 1, "title": "Lead Data Scientist", "company": "Kaggle Labs", "location": "Bengaluru", "exp": "3-5 yrs", "skills": "Python, Machine Learning, Scikit-Learn, Deep Learning, SQL, NLP", "desc": "Train and validate machine learning architectures, statistical hypothesis testing, and business impact modeling.", "domain": "Data Science", "base_salary": "₹16 - 24 LPA"},
        {"id": 2, "title": "AI & MLOps Specialist", "company": "NeuralScale Systems", "location": "Hyderabad", "exp": "2-4 yrs", "skills": "Python, PyTorch, Docker, Kubernetes, AWS, CI/CD, Linux", "desc": "Design automated model monitoring systems, scale inference pipelines, and manage cloud endpoints.", "domain": "Data Science", "base_salary": "₹14 - 20 LPA"},
        {"id": 3, "title": "Senior Java Architect", "company": "FinTech Core", "location": "Pune", "exp": "4-7 yrs", "skills": "Java, Spring Boot, Microservices, Kafka, PostgreSQL, Docker, Hibernate", "desc": "Architect high-throughput transactional backends and lead enterprise integration.", "domain": "Java Developer", "base_salary": "₹18 - 26 LPA"},
        {"id": 4, "title": "Fullstack Web Designer", "company": "Studio Pixelcraft", "location": "Remote", "exp": "1-3 yrs", "skills": "Figma, HTML, CSS, JavaScript, React, UI/UX, Wireframing", "desc": "Craft component design systems, build interactive responsive web prototypes, and test usability.", "domain": "Web Designing", "base_salary": "₹7 - 12 LPA"},
        {"id": 5, "title": "Cloud DevOps Engineer", "company": "InnoCloud Tech", "location": "Hyderabad", "exp": "2-5 yrs", "skills": "AWS, Docker, Kubernetes, Terraform, Jenkins, Linux, CI/CD", "desc": "Maintain containerized infrastructures, automate staging deployments, and oversee zero-downtime releases.", "domain": "DevOps Engineer", "base_salary": "₹12 - 18 LPA"},
        {"id": 6, "title": "CAD Mechanical Design Engineer", "company": "Apex Mobility Ltd", "location": "Chennai", "exp": "2-4 yrs", "skills": "SolidWorks, AutoCAD, FEA, ANSYS, GD&T, CAD, Thermodynamics", "desc": "Perform thermal analysis, model mechanical enclosures in SolidWorks, and coordinate manufacturing tolerancing.", "domain": "Mechanical Engineer", "base_salary": "₹8 - 14 LPA"},
        {"id": 7, "title": "Technical Sourcing Specialist", "company": "PeopleEdge HR", "location": "Mumbai", "exp": "1-3 yrs", "skills": "Recruitment, Sourcing, Talent Acquisition, Screening, HRIS", "desc": "Manage end-to-end recruitment funnels, pipeline top software engineers, and conduct interviews.", "domain": "HR", "base_salary": "₹6 - 10 LPA"},
        {"id": 8, "title": "Operations & Logistics Manager", "company": "SwiftCargo Global", "location": "Delhi NCR", "exp": "3-6 yrs", "skills": "Operations Management, Supply Chain, Logistics, Six Sigma", "desc": "Streamline warehouse delivery throughput, manage partner SLAs, and optimize operational margins.", "domain": "Operations Manager", "base_salary": "₹11 - 17 LPA"}
    ]
    df_corpus = pd.DataFrame(jobs_data)
    df_corpus["clean_text"] = (df_corpus["title"] + " " + df_corpus["skills"] + " " + df_corpus["desc"]).apply(clean_text)
    
    # 3. Vectorization & K-Means Unsupervised Clustering
    vec_rec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    corpus_vectors = vec_rec.fit_transform(df_corpus["clean_text"])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_corpus["cluster"] = kmeans.fit_predict(corpus_vectors)
    
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(corpus_vectors.toarray())
    df_corpus["pca_x"] = pca_coords[:, 0]
    df_corpus["pca_y"] = pca_coords[:, 1]
    
    # 4. Supervised Fit / Shortlisting Model (Model B)
    np.random.seed(42)
    synth_X, synth_y = [], []
    for _ in range(250):
        jaccard = np.random.uniform(0.05, 0.75)
        cos_sim = np.random.uniform(0.10, 0.85)
        len_diff = np.random.randint(10, 200)
        shortlisted = 1 if (cos_sim > 0.35 and jaccard > 0.16) else 0
        synth_X.append([jaccard, cos_sim, len_diff])
        synth_y.append(shortlisted)
        
    fit_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
    fit_predictor.fit(synth_X, synth_y)
    
    return classifier, vec_clf, df_corpus, vec_rec, corpus_vectors, kmeans, pca, fit_predictor

classifier, vec_clf, df_corpus, vec_rec, corpus_vectors, kmeans, pca, fit_predictor = build_engine()

# =========================================================
# 4. SKILL RADAR MATRIX & AUDIT LOGIC
# =========================================================
def calculate_radar_scores(text: str):
    """Calculates coverage scores across skill dimensions."""
    text_lower = text.lower()
    scores = {}
    for dim, skill_list in SKILL_DIMENSIONS.items():
        found = sum(1 for s in skill_list if re.search(r"\b" + re.escape(s) + r"\b", text_lower))
        scores[dim] = min((found / max(len(skill_list) // 2, 1)) * 100, 100)
    return scores

def analyze_skill_gap(candidate_text: str, target_skills_str: str):
    cand_lower = candidate_text.lower()
    found_skills = {s.title() for s in FLAT_LEXICON if re.search(r"\b" + re.escape(s) + r"\b", cand_lower)}
    req_skills = {s.strip().title() for s in target_skills_str.split(",") if s.strip()}
    matched = sorted(list(req_skills.intersection(found_skills)))
    missing = sorted(list(req_skills - found_skills))
    return matched, missing

def calculate_fit_score(resume_clean: str, job_clean: str, job_vec, candidate_vec):
    r_words = set(resume_clean.split())
    j_words = set(job_clean.split())
    jaccard = len(r_words.intersection(j_words)) / max(len(r_words.union(j_words)), 1)
    cos_sim = float(cosine_similarity(candidate_vec, job_vec)[0][0])
    len_diff = abs(len(r_words) - len(j_words))
    prob = fit_predictor.predict_proba([[jaccard, cos_sim, len_diff]])[0][1]
    return round(prob * 100, 1), round(cos_sim * 100, 1)

# =========================================================
# 5. SIDEBAR CONFIGURATION
# =========================================================
with st.sidebar:
    st.markdown("<h2 style='color: #881337; margin-bottom: 0;'>🌸 SmartHire Pro</h2>", unsafe_allow_html=True)
    st.caption("Cream & Rose AI Recruitment Workspace")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📂 Upload Candidate Resume", type=["pdf", "docx", "txt"])
    
    st.markdown("### 🎛️ Matching Engine Tuning")
    skill_weight = st.slider("Skill Overlap Priority", min_value=0.2, max_value=1.0, value=0.6, step=0.1)
    
    selected_locations = st.multiselect(
        "Location Filters", 
        options=sorted(df_corpus["location"].unique()), 
        default=sorted(df_corpus["location"].unique())
    )
    max_results = st.slider("Top Recommendations Count", min_value=3, max_value=8, value=5)
    
    st.markdown("---")
    st.caption("✨ Pipeline: TF-IDF Vectorization • Logistic Regression Classifier • Cosine Distance • Random Forest Fit Engine")

# =========================================================
# 6. MAIN WORKSPACE
# =========================================================
st.markdown("""
<div class="main-header">
    <h1>SmartHire: Talent Intelligence & Match Portal</h1>
    <p>Predictive resume classification, Top-N role matching, and skill gap audits built entirely with classical machine learning.</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👋 Upload a resume in PDF, DOCX, or TXT format via the sidebar to initiate analysis.")
    st.markdown("### 📋 Active Openings Catalog")
    st.dataframe(
        df_corpus[["title", "company", "location", "exp", "skills", "base_salary"]].rename(
            columns={"title": "Role Title", "company": "Company", "location": "Location", "exp": "Experience", "skills": "Required Stack", "base_salary": "Salary Band"}
        ),
        use_container_width=True
    )
else:
    raw_resume = extract_text_from_upload(uploaded_file)
    cleaned_resume = clean_text(raw_resume)
    
    if not cleaned_resume.strip():
        st.error("Unable to extract text from the uploaded file. Please confirm the document is not an image-only scan.")
    else:
        # Compute Vectors
        res_clf_vec = vec_clf.transform([cleaned_resume])
        res_rec_vec = vec_rec.transform([cleaned_resume])
        
        # Supervised Classification (Model A)
        pred_category = classifier.predict(res_clf_vec)[0]
        confidence = float(max(classifier.predict_proba(res_clf_vec)[0])) * 100
        
        # Unsupervised Recommender Engine
        sim_scores = cosine_similarity(res_rec_vec, corpus_vectors)[0]
        
        # Filter indices by chosen location
        filtered_indices = df_corpus[df_corpus["location"].isin(selected_locations)].index
        ranked_indices = [idx for idx in sim_scores.argsort()[::-1] if idx in filtered_indices][:max_results]
        
        # Summary Metrics Header
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Predicted Domain", pred_category)
        with c2:
            st.metric("Model A Confidence", f"{confidence:.1f}%")
        with c3:
            top_match = round(sim_scores[ranked_indices[0]] * 100, 1) if ranked_indices else 0
            st.metric("Top Role Match", f"{top_match}%")
        with c4:
            st.metric("Corpus Postings", f"{len(df_corpus)}")
            
        st.write("")
        
        # Tabbed Panes
        tab_rec, tab_radar, tab_gap, tab_cluster, tab_cv = st.tabs([
            "🎯 Top Role Matches",
            "📊 Skill Radar Matrix", 
            "📈 Skill Gap & Mentorship",
            "🌐 PCA Vector Clustering", 
            "📄 Parsed Content"
        ])
        
        # =========================================================
        # TAB 1: RECOMMENDATIONS & MODEL B SHORTLISTING
        # =========================================================
        with tab_rec:
            st.subheader("Ranked Role Matches & Estimated Shortlisting Odds")
            
            if not ranked_indices:
                st.warning("No listings match your selected location filters.")
            else:
                for rank, idx in enumerate(ranked_indices):
                    job = df_corpus.iloc[idx]
                    job_vec = corpus_vectors[idx]
                    
                    fit_score, match_score = calculate_fit_score(cleaned_resume, job["clean_text"], job_vec, res_rec_vec)
                    
                    with st.expander(f"#{rank+1}: {job['title']} — {job['company']} (Match: {match_score}%)", expanded=(rank==0)):
                        col_l, col_r = st.columns([2, 1])
                        with col_l:
                            st.markdown(f"📍 **Location:** `{job['location']}` | ⏳ **Experience:** `{job['exp']}`")
                            st.markdown(f"💰 **Estimated Salary Band:** `{job['base_salary']}`")
                            st.markdown(f"🛠️ **Target Skills:** `{job['skills']}`")
                            st.write(f"📝 **Overview:** {job['desc']}")
                        with col_r:
                            st.markdown("##### Model B Fit Prediction")
                            st.metric("Shortlisting Probability", f"{fit_score}%")
                            st.progress(fit_score / 100.0)
                            status_text = "🟢 Strong Candidate" if fit_score >= 60 else ("🟡 Competitive" if fit_score >= 35 else "🔴 Significant Gaps")
                            st.caption(f"Screening Status: **{status_text}**")
                            
        # =========================================================
        # TAB 2: SKILL RADAR MATRIX
        # =========================================================
        with tab_radar:
            st.subheader("Competency Distribution Across Key Domains")
            st.caption("Aggregated technical presence across six core engineering dimensions.")
            
            radar_data = calculate_radar_scores(raw_resume)
            categories = list(radar_data.keys())
            values = list(radar_data.values())
            values += values[:1] # close polygon
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            # Matplotlib styling with cream & blush-pink palette
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#FAF6F0')
            ax.set_facecolor('#FFF9F3')
            
            ax.plot(angles, values, color='#BE123C', linewidth=2.5, linestyle='solid')
            ax.fill(angles, values, color='#FDA4AF', alpha=0.45)
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories, color='#881337', fontsize=10, weight='bold')
            ax.set_ylim(0, 100)
            ax.grid(color='#FBCFE8', linestyle='--')
            
            col_chart, col_data = st.columns([3, 2])
            with col_chart:
                st.pyplot(fig)
            with col_data:
                st.markdown("#### Dimension Breakdown")
                for dim, score in radar_data.items():
                    st.write(f"**{dim}:** `{score:.0f}% Coverage`")
                    st.progress(score / 100.0)

        # =========================================================
        # TAB 3: SKILL GAP & ACTIONABLE REPORT
        # =========================================================
        with tab_gap:
            st.subheader("Target Skill Gap & Personalized Upskilling Report")
            benchmark_job = df_corpus.iloc[ranked_indices[0]] if ranked_indices else df_corpus.iloc[0]
            st.markdown(f"Benchmark Role: **{benchmark_job['title']}** at **{benchmark_job['company']}**")
            
            matched, missing = analyze_skill_gap(raw_resume, benchmark_job["skills"])
            
            cg1, cg2 = st.columns(2)
            with cg1:
                st.markdown("#### ✅ Detected Competencies")
                if matched:
                    badges = "".join([f'<span class="badge-matched">✓ {s}</span>' for s in matched])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.warning("No explicit target competencies detected in CV.")
            with cg2:
                st.markdown("#### ⚠️ High-Priority Skill Deficits")
                if missing:
                    badges = "".join([f'<span class="badge-missing">✗ {s}</span>' for s in missing])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.success("🎉 Perfect match! Your profile addresses all core requirements.")
                    
            st.markdown("---")
            st.markdown("#### 💡 Targeted Career Growth Action Plan")
            if missing:
                st.write(f"To optimize interview shortlisting for **{benchmark_job['title']}**, focus on developing verified projects in: **{', '.join(missing)}**.")
            else:
                st.write("Your technical profile demonstrates comprehensive skill coverage. Emphasize quantifiable performance improvements, latency drops, or business metrics in your project descriptions.")
                
            report_text = f"""SMARTHIRE CANDIDATE AUDIT & CAREER GAP REPORT
===================================================
Predicted Domain: {pred_category} (Confidence: {confidence:.1f}%)
Benchmark Role: {benchmark_job['title']} ({benchmark_job['company']})
Match Score: {round(sim_scores[benchmark_job.name]*100, 1)}%
Estimated Salary Band: {benchmark_job['base_salary']}

MATCHED SKILLS:
{', '.join(matched) if matched else 'None'}

RECOMMENDED SKILLS TO LEARN:
{', '.join(missing) if missing else 'None! Profile fulfills role criteria.'}
===================================================
Generated with SmartHire Classical ML Engine.
"""
            st.download_button(
                label="📥 Download Audit Report (TXT)",
                data=report_text,
                file_name="SmartHire_Audit_Report.txt",
                mime="text/plain"
            )

        # =========================================================
        # TAB 4: K-MEANS & PCA 2D PROJECTION
        # =========================================================
        with tab_cluster:
            st.subheader("Job Space Embedding & Candidate Vector Projection")
            st.caption("2D PCA projection of TF-IDF vectors illustrating the spatial distribution of roles and your resume.")
            
            cand_pca = pca.transform(res_rec_vec.toarray())
            
            fig_pca, ax_pca = plt.subplots(figsize=(9, 5))
            fig_pca.patch.set_facecolor('#FAF6F0')
            ax_pca.set_facecolor('#FFF9F3')
            
            # Palette matching cream & rose
            palette = ["#FB7185", "#F43F5E", "#E11D48", "#BE123C", "#9F1239"]
            sns.scatterplot(
                data=df_corpus, 
                x="pca_x", 
                y="pca_y", 
                hue="domain", 
                palette=palette[:len(df_corpus["domain"].unique())],
                s=120, 
                alpha=0.9, 
                ax=ax_pca
            )
            
            # Project candidate marker
            ax_pca.scatter(
                cand_pca[0, 0], 
                cand_pca[0, 1], 
                color="#881337", 
                s=280, 
                marker="*", 
                edgecolors="#FFFFFF", 
                linewidth=1.8, 
                label="⭐ Your Resume Position"
            )
            
            ax_pca.set_title("PCA Projection of Postings vs. Candidate Profile", color="#881337", fontsize=12, weight="bold")
            ax_pca.set_xlabel("Principal Component 1", color="#881337")
            ax_pca.set_ylabel("Principal Component 2", color="#881337")
            ax_pca.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            st.pyplot(fig_pca)

        # =========================================================
        # TAB 5: PARSED CONTENT
        # =========================================================
        with tab_cv:
            st.subheader("Extracted Document Text")
            st.text_area("Normalized Stream Preview", raw_resume, height=350)
