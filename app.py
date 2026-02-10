import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Graph ----
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# ---- Resume Parsing ----
import pdfplumber
import docx
import re

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Career Analytics Dashboard", layout="wide")

# ================= LOAD DATA =================
@st.cache_data
def load_main_data():
    resumes = pd.read_csv("7000_students_resume_dataset.csv")
    jobs = pd.read_csv("1000_job_description_dataset.csv")
    syllabus = pd.read_csv("syllabus_subjects_skills.csv")
    return resumes, jobs, syllabus

@st.cache_data
def load_career_forecast():
    return pd.read_csv("career_forecast_trends.csv")

resumes, jobs, syllabus = load_main_data()
forecast_df = load_career_forecast()

# ================= PREPROCESS =================
resumes["profile"] = resumes["education"] + " " + resumes["skills"]
jobs["job_profile"] = jobs["job_role"] + " " + jobs["skills_required"] + " " + jobs["job_description"]

syllabus["skill_set"] = syllabus["skills_learned"].apply(
    lambda x: set(s.strip().lower() for s in x.split(","))
)
syllabus_skills_set = set().union(*syllabus["skill_set"].tolist())
all_job_roles = sorted(jobs["job_role"].unique())

# ================= TF-IDF =================
tfidf = TfidfVectorizer(stop_words="english")
resume_vecs = tfidf.fit_transform(resumes["profile"])
job_vecs = tfidf.transform(jobs["job_profile"])
similarity = cosine_similarity(resume_vecs, job_vecs)

# ================= SKILL VOCABULARY =================
resume_vocab = set(s.strip().lower() for skills in resumes["skills"] for s in skills.split(","))
job_vocab = set(s.strip().lower() for skills in jobs["skills_required"] for s in skills.split(","))
MASTER_SKILLS = resume_vocab | job_vocab | syllabus_skills_set

# ================= RESUME PARSING =================
def extract_text_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    return text.lower()

def extract_text_docx(file):
    doc = docx.Document(file)
    return " ".join(p.text for p in doc.paragraphs).lower()

def extract_skills(text):
    found = set()
    for skill in MASTER_SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", text):
            found.add(skill)
    return found

# ================= NAME EXTRACTION =================
def extract_candidate_name(text, file_name):
    """
    Heuristic-based name extraction:
    1. First non-empty line
    2. Title-case line (2–4 words)
    3. Fallback to file name
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Rule 1: First line
    if lines:
        first = lines[0]
        if 2 <= len(first.split()) <= 4:
            return first.title()

    # Rule 2: Title-case detection
    for line in lines[:10]:
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            return line.title()

    # Rule 3: Fallback to filename (without extension)
    return file_name.rsplit(".", 1)[0]

# ================= CORE FUNCTIONS =================
def recommend_unique_roles(student_id, top_n):
    idx = resumes[resumes["id"] == student_id].index[0]
    scores = similarity[idx]

    temp = jobs.copy()
    temp["score"] = scores * 100

    grouped = (
        temp.groupby("job_role")
        .agg({
            "score": "max",
            "skills_required": lambda x: ", ".join(
                set(", ".join(x).lower().split(","))
            )
        })
        .reset_index()
        .sort_values("score", ascending=False)
        .head(top_n)
    )

    grouped["score (%)"] = grouped["score"].round(2)
    return grouped

def combined_skill_coverage(resume_skills, job_skills):
    resume_set = set(s.strip().lower() for s in resume_skills.split(","))
    job_set = set(s.strip().lower() for s in job_skills.split(","))

    combined = resume_set | syllabus_skills_set
    obtained = combined & job_set
    missing = job_set - combined

    return obtained, missing

def render_skill_graph(student_name, role_name, obtained, missing):
    G = nx.Graph()

    G.add_node(
        student_name,
        label=student_name,
        color="#2E7D32",
        size=45,
        font={"size": 20, "color": "black", "bold": True}
    )

    G.add_node(
        role_name,
        label=role_name,
        color="#1565C0",
        size=40,
        font={"size": 18, "color": "black", "bold": True}
    )

    for s in obtained:
        G.add_node(s, label=s.title(), color="#A5D6A7", size=22, font={"size": 12})
        G.add_edge(student_name, s, label="has_skill", color="green")
        G.add_edge(role_name, s, label="requires", color="blue")

    for s in missing:
        G.add_node(s, label=s.title(), color="#EF9A9A", size=22, font={"size": 12})
        G.add_edge(role_name, s, label="requires", color="red", dashes=True)

    net = Network(height="550px", width="100%", bgcolor="#ffffff")
    net.from_nx(G)
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3500,
          "springLength": 240
        }
      }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        components.html(open(f.name).read(), height=580)

# ================= UI =================
st.title("🎓 Career Analytics Dashboard")
st.caption("Real-Time Skill Gap • Career Forecast • Skill Graph")

dashboard_view = st.radio(
    "Select Dashboard View",
    ["🎯 Skill Gap Analysis", "📈 Career Forecast", "🕸️ Skill Graph"],
    horizontal=True
)

# ================= SIDEBAR =================
with st.sidebar:
    st.subheader("Student Input Mode")
    mode = st.radio("Choose Input", ["Dataset Student", "Upload Resume"])

    if mode == "Dataset Student":
        student_id = st.selectbox("Student ID", resumes["id"])
        student = resumes[resumes["id"] == student_id].iloc[0]
        student_name = student["name"]
        active_skills = student["skills"]

    else:
        file = st.file_uploader("Upload Resume (PDF / DOCX)", type=["pdf", "docx"])
        if not file:
            st.stop()

        text = extract_text_pdf(file) if file.name.lower().endswith(".pdf") else extract_text_docx(file)
        skills = extract_skills(text)
        active_skills = ", ".join(sorted(skills))

        # ✅ Auto-detect candidate name
        student_name = extract_candidate_name(text, file.name)

# =================================================
# DASHBOARD 1: SKILL GAP
# =================================================
if dashboard_view == "🎯 Skill Gap Analysis":

    st.subheader("👤 Student Profile")
    st.write(f"**Name:** {student_name}")
    st.write(f"**Skills:** {active_skills}")

    role_name = st.selectbox("Select Career Role", all_job_roles)
    role = jobs[jobs["job_role"] == role_name].iloc[0]

    obtained, missing = combined_skill_coverage(active_skills, role["skills_required"])

    col1, col2 = st.columns([1, 3])
    with col1:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.pie([len(obtained), len(missing)], labels=["Matched", "Missing"], autopct="%1.0f%%")
        ax.axis("equal")
        st.pyplot(fig)

    with col2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ✅ Matched Skills")
            for s in sorted(obtained):
                st.write(f"• {s.title()}")
        with c2:
            st.markdown("### ❌ Missing Skills")
            for s in sorted(missing):
                st.write(f"• {s.title()}")

# =================================================
# DASHBOARD 2: FORECAST
# =================================================
if dashboard_view == "📈 Career Forecast":

    st.subheader("📈 Future Job Market Trends")
    st.dataframe(forecast_df, height=300)

    st.markdown("### 🎓 Learning Resources")
    for skill in forecast_df["Skill"].unique():
        with st.expander(f"📘 Learn {skill.title()}"):
            st.markdown(f"""
            - [Documentation](https://www.google.com/search?q={skill}+documentation)
            - [Coursera](https://www.coursera.org/search?query={skill})
            - [YouTube](https://www.youtube.com/results?search_query={skill}+tutorial)
            """)

# =================================================
# DASHBOARD 3: SKILL GRAPH
# =================================================
if dashboard_view == "🕸️ Student Graph":

    st.subheader("🕸️ Skill Relationship Graph")
    st.markdown("""
    **Legend**
    - 🟢 Student
    - 🔵 Job Role
    - 🟩 Obtained Skill
    - 🟥 Missing Skill
    """)

    role_name = st.selectbox("Select Career Role", all_job_roles)
    role = jobs[jobs["job_role"] == role_name].iloc[0]

    obtained, missing = combined_skill_coverage(active_skills, role["skills_required"])
    render_skill_graph(student_name, role_name, obtained, missing)

# ================= FOOTER =================
st.markdown("---")
st.caption("TF-IDF Skill Gap • Real-Time Resume Upload • Career Forecast • Graph Model")
