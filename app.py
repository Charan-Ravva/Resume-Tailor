import io
import json
import pandas as pd
import PyPDF2
import streamlit as st
from google import genai
from google.genai import types
from jobspy import scrape_jobs
from xhtml2pdf import pisa
from supabase import create_client, Client

# --- CONFIGURATION ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

# --- SUPABASE CONFIGURATION ---
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase_client() -> Client:
    """Initializes and caches the Supabase client connection."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase_client = get_supabase_client()


# --- SUPABASE DATABASE & STORAGE HELPERS ---
def save_application_supabase(job_id, company, title, location, job_url, ats_score, status, notes, pdf_bytes):
    """Saves application record to Supabase Postgres and uploads PDF to Storage."""
    if not supabase_client:
        return False

    pdf_filename = f"{job_id}.pdf"

    # 1. Upload PDF binary to 'resumes' bucket
    try:
        supabase_client.storage.from_("resumes").upload(
            path=pdf_filename,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
    except Exception as e:
        st.warning(f"Note on PDF Storage upload: {e}")

    # 2. Insert/Upsert record in 'applications' table
    data = {
        "job_id": job_id,
        "company": company,
        "title": title,
        "location": location,
        "job_url": job_url,
        "ats_score": str(ats_score),
        "status": status,
        "notes": notes,
        "pdf_path": pdf_filename
    }

    supabase_client.table("applications").upsert(data).execute()
    return True

def fetch_all_applications_supabase():
    """Fetches all applications from Supabase ordered by date created."""
    if not supabase_client:
        return pd.DataFrame()
    res = supabase_client.table("applications").select("*").order("created_at", desc=True).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

def download_pdf_from_supabase(pdf_filename):
    """Retrieves PDF binary from Supabase Storage."""
    if not supabase_client or not pdf_filename:
        return None
    try:
        return supabase_client.storage.from_("resumes").download(pdf_filename)
    except Exception as e:
        st.error(f"Error fetching PDF from storage: {e}")
        return None

def update_application_status_supabase(job_id, status, notes):
    """Updates status and notes for a specific job application."""
    if not supabase_client:
        return
    supabase_client.table("applications").update({"status": status, "notes": notes}).eq("job_id", job_id).execute()

def delete_application_supabase(job_id, pdf_filename):
    """Deletes an application record and its PDF from Supabase."""
    if not supabase_client:
        return
    supabase_client.table("applications").delete().eq("job_id", job_id).execute()
    if pdf_filename:
        try:
            supabase_client.storage.from_("resumes").remove([pdf_filename])
        except Exception:
            pass


# --- SESSION STATE INITIALIZATION ---
if "selected_jd" not in st.session_state:
    st.session_state.selected_jd = ""
if "selected_company" not in st.session_state:
    st.session_state.selected_company = ""
if "selected_title" not in st.session_state:
    st.session_state.selected_title = ""
if "selected_location" not in st.session_state:
    st.session_state.selected_location = ""
if "selected_job_url" not in st.session_state:
    st.session_state.selected_job_url = ""


# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from the uploaded PDF resume."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text


def tailor_resume(resume_text, job_description):
    """Sends the prompt to Gemini and enforces a rigid JSON schema response."""
    prompt = f"""
    You are an expert technical recruiter. 
    Analyze this candidate's resume:
    {resume_text}
    
    Optimize it against this Job Description:
    {job_description}
    
    CANDIDATE INFO:
    - Name: Sri Charan Ravva
    - Email: Sricharan.ravva07@gmail.com
    - Phone: 616-439-0213
    - Location: New York, USA
    - LinkedIn: www.linkedin.com/in/charanravva

    STEP 1: TARGET FOCUS IDENTIFICATION
    Identify the 3 to 5 core themes or highest-priority keywords emphasized most in the JD (e.g., Lead Generation, CRM Systems Architecture, Pipeline Velocity, Commercial Intelligence).

    STEP 2: MANDATORY BULLET RESTRUCTURING & REWRITING (STRICT)
    For EACH position in professional_experience:
    1. REORDER: Move or add bullet points that directly match the JD at the start.
    Generate 9 high-impact bullets for both roles.
    2. WRITE NEW BULLETS: Generate 1 to 2 BRAND NEW bullet points explicitly describing accomplishment-driven tasks built around missing JD keywords (e.g., event analysis, win/loss trend reporting, re-engaging dormant accounts).
    3. NO TACKING ON / REWRITE ENTIRELY: NEVER simply tack JD keywords onto the end or start of old sentences. Rewrite the entire sentence seamlessly around the accomplishment.
    4. STRUCTURE RULE: Every single bullet point MUST strictly follow: 
       [Strong Action Verb] + [Context & Business Task] + [Technical Tool Used] + [Quantifiable Business Outcome/Metric].
       
    STEP 3: 
    - DO NOT alter past or current employment job titles inside professional_experience. Only reflect the target position title inside the Professional Summary.

    STEP 4: KEYWORD GAP AUDIT
    Scan the JD against the master resume across 4 buckets:
    - Platforms & Tools (e.g., Salesforce, HubSpot, Marketo, Databricks)
    - Languages & Scripting (e.g., SQL, Python, R, AMPScript)
    - Methodologies & Processes (e.g., A/B Testing, Lead Scoring, CRM Hygiene)
    - Inject every missing tool and methodology into the appropriate category in "technical_skills". Reorder each category so tools mentioned in the JD appear FIRST. Keywords or methodology should not be more than 15 per category.

    STEP 5: PROFESSIONAL SUMMARY CUSTOMIZATION
    Rewrite the summary (4–5 sentences max) to directly reflect the target role's exact title and core responsibilities. Highlight tech stack, years of experience, and business impact. Strip out ALL LaTeX symbols (like '$') and convert to plain text.

    You must output a single JSON object matching this exact structural schema:
    {{
      "name": "Candidate Full Name",
      "contact": {{ "email": "...", "phone": "...", "location": "...", "linkedin": "..." }},
      "summary": "Optimized professional summary paragraph",
      "technical_skills": {{
         "Category Name (e.g., Data Querying & Programming)": ["Skill 1", "Skill 2"]
      }},
      "professional_experience": [
         {{
           "company": "...",
           "role": "...",
           "date": "...",
           "location": "...",
           "bullets": ["Optimized bullet 1", "Optimized bullet 2"]
         }}
      ],
      "education": [
         {{ "degree": "...", "school": "...", "date": "..." }}
      ],
      "estimated_ats_score": "95%",
      "added_keywords": ["keyword1", "keyword2"],
      "missing_keywords": ["keyword3"],
      "explanation": "Brief explanation of updates."
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0
        ),
    )
    return json.loads(response.text)


def create_pdf(data):
    """Compiles the JSON data structure into an elegant PDF file with strict orphan/page-break protection."""

    linkedin_raw = data["contact"].get(
        "linkedin", "www.linkedin.com/in/charanravva"
    )
    if linkedin_raw.startswith("http"):
        linkedin_url = linkedin_raw
    else:
        linkedin_url = f"https://{linkedin_raw}"

    linkedin_html = f'<a href="{linkedin_url}">{linkedin_raw}</a>'

    # 1. Technical Skills Block
    skills_html = ""
    for category, skills in data.get("technical_skills", {}).items():
        skills_html += f"""
        <p style="margin: 2px 0; font-size: 10pt;">
            <strong>{category}:</strong> {", ".join(skills)}
        </p>
        """

    # 2. Professional Experience Block
    experience_html = ""
    for job in data.get("professional_experience", []):
        bullets_html = "".join([
            f"<li style='margin-bottom: 2px; font-size: 9.5pt;'>{b}</li>"
            for b in job.get("bullets", [])
        ])
        experience_html += f"""
        <div class="job-block">
            <table style="width: 100%; margin-top: 4px; margin-bottom: 2px;" cellpadding="0" cellspacing="0">
                <tr style="-pdf-keep-with-next: true;">
                    <td style="font-weight: bold; font-size: 10pt; width: 60%;">{job.get('role')}, {job.get('company')}</td>
                    <td style="text-align: right; font-style: italic; font-size: 10pt; width: 40%;">{job.get('date')} | {job.get('location')}</td>
                </tr>
            </table>
            <ul style="margin-top: 2px; margin-bottom: 6px; padding-left: 18px;">
                {bullets_html}
            </ul>
        </div>
        """

    # 3. Education Block
    education_html = ""
    for edu in data.get("education", []):
        education_html += f"""
        <div style="page-break-inside: avoid;">
            <table style="width: 100%; margin-top: 4px;" cellpadding="0" cellspacing="0">
                <tr>
                    <td style="font-weight: bold; font-size: 10pt; width: 70%;">{edu.get('degree')}</td>
                    <td style="text-align: right; font-style: italic; font-size: 10pt; width: 30%;">{edu.get('date')}</td>
                </tr>
                <tr>
                    <td style="font-size: 10pt; font-style: italic;" colspan="2">{edu.get('school')}</td>
                </tr>
            </table>
        </div>
        """

    # 4. Global HTML & CSS Template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @page {{
            size: letter;
            margin: 0.4in;
        }}
        body {{
            font-family: calibri;
            color: #111111;
            text-align: justify;
            line-height: 1.25;
            font-size: 9pt;
        }}
        .name {{
            text-align: center;
            font-size: 16pt;
            font-weight: bold;
            margin-bottom: 2px;
        }}
        .contact {{
            text-align: center;
            font-size: 9.5pt;
            margin-bottom: 10px;
            color: #111111;
        }}
        .contact a {{
            color: #0066CC;
            text-decoration: underline;
        }}
        .section-title {{
            font-size: 11pt;
            font-weight: bold;
            text-transform: uppercase;
            border-bottom: 1px solid #222222;
            margin-top: 8px;
            margin-bottom: 4px;
            padding-bottom: 1px;
            page-break-after: avoid;
            -pdf-keep-with-next: true;
        }}
        .summary {{
            font-size: 9.5pt;
            text-align: justify;
            margin-bottom: 6px;
        }}
        .job-block {{
            page-break-inside: avoid;
            margin-bottom: 6px;
        }}
    </style>
    </head>
    <body>
        <div class="name">{data.get('name', '')}</div>
        <div class="contact">
            {data['contact'].get('email', '')} | {data['contact'].get('phone', '')} | {data['contact'].get('location', '')} | {linkedin_html}
        </div>
        
        <div class="section-title">Summary</div>
        <div class="summary">{data.get('summary', '')}</div>
        
        <div class="section-title">Technical Skills</div>
        {skills_html}
        
        <div class="section-title">Professional Experience</div>
        {experience_html}
        
        <div class="section-title">Education</div>
        {education_html}
    </body>
    </html>
    """

    result = io.BytesIO()
    pisa_status = pisa.CreatePDF(html_template, dest=result)

    if not pisa_status.err:
        result.seek(0)
        return result
    return None


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="AI Resume Tailor & Job Tracker", layout="wide")

tab1, tab2, tab3 = st.tabs([
    "🔍 Job Scraper (Last 24 Hours)", 
    "🎯 Resume Tailor", 
    "📊 Application Tracker Dashboard"
])

# ==========================================
# TAB 1: LIVE JOB SCRAPER
# ==========================================
with tab1:
    st.header("Find Marketing Analyst Jobs (Posted in Last 24 Hours)")

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        search_term = st.text_input("Job Title Query", value="Marketing Analyst")
    with col_b:
        location = st.text_input("Location", value="United States")
    with col_c:
        results_num = st.number_input(
            "Max Results", min_value=10, max_value=100, value=25
        )

    boards = st.multiselect(
        "Target Platforms",
        ["linkedin", "indeed", "zip_recruiter", "glassdoor"],
        default=["linkedin", "indeed"],
    )

    if st.button("Search Fresh Postings", type="primary"):
        with st.spinner("Scraping live listings from the past 24 hours..."):
            try:
                jobs_df = scrape_jobs(
                    site_name=boards,
                    search_term=search_term,
                    location=location,
                    results_wanted=results_num,
                    hours_old=24,
                    country_indeed="USA",
                    linkedin_fetch_description=True,
                )

                if not jobs_df.empty:
                    st.session_state.jobs_df = jobs_df
                    st.success(
                        f"Found {len(jobs_df)} jobs posted in the last 24 hours!"
                    )
                else:
                    st.warning(
                        "No jobs found matching your criteria in the past 24 hours."
                    )
            except Exception as err:
                st.error(f"Error executing scraper: {err}")

    # Render results table and import controls
    if "jobs_df" in st.session_state and not st.session_state.jobs_df.empty:
        df = st.session_state.jobs_df

        st.dataframe(
            df[["site", "title", "company", "location", "date_posted", "job_url"]],
            use_container_width=True,
        )

        st.subheader("Import Job to Resume Generator")
        job_options = [
            f"{row['company']} - {row['title']} ({row['site']})"
            for idx, row in df.iterrows()
        ]
        selected_index = st.selectbox(
            "Select a job listing:",
            range(len(job_options)),
            format_func=lambda x: job_options[x],
        )

        selected_row = df.iloc[selected_index]
        st.markdown(
            f"**Selected:** {selected_row['title']} at **{selected_row['company']}**"
        )
        st.markdown(f"🔗 [View Original Job Posting]({selected_row['job_url']})")

        with st.expander("Preview Full Job Description"):
            st.write(selected_row.get("description", "No description available."))

        if st.button("➡️ Import Description into Resume Tailor"):
            st.session_state.selected_jd = selected_row.get("description", "")
            st.session_state.selected_company = selected_row.get("company", "")
            st.session_state.selected_title = selected_row.get("title", "")
            st.session_state.selected_location = selected_row.get("location", "")
            st.session_state.selected_job_url = selected_row.get("job_url", "")
            st.success(
                "Successfully imported! Navigate to the 'Resume Tailor' tab to generate your PDF."
            )


# ==========================================
# TAB 2: RESUME TAILOR
# ==========================================
with tab2:
    st.header("🎯 AI Resume Tailor")
    st.write(
        "Upload a base resume, review or edit the target job description, and"
        " generate your optimized PDF."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Your Master Resume")
        uploaded_resume = st.file_uploader("Upload PDF", type=["pdf"])

    with col2:
        st.subheader("2. Target Job Details")
        company_name = st.text_input(
            "Company Name",
            value=st.session_state.selected_company,
            placeholder="e.g., Intuit, Google, PepsiCo",
        )
        target_title = st.text_input(
            "Job Title",
            value=st.session_state.selected_title,
            placeholder="e.g., Senior Data Analyst",
        )
        job_description = st.text_area(
            "Paste Job Description Here",
            value=st.session_state.selected_jd,
            height=200,
        )

    if st.button("Tailor My Resume", type="primary", use_container_width=True):
        if uploaded_resume and job_description:
            with st.spinner(
                "Analyzing data and generating your optimized resume document..."
            ):
                try:
                    base_text = extract_text_from_pdf(uploaded_resume)
                    result_data = tailor_resume(base_text, job_description)
                    pdf_buffer = create_pdf(result_data)

                    if pdf_buffer:
                        st.success("Resume Tailored and Formatted Successfully!")
                        st.metric(
                            label="Estimated ATS Match Score",
                            value=result_data.get("estimated_ats_score", "N/A"),
                        )
                        st.info(
                            "**Optimization Summary:**"
                            f" {result_data.get('explanation', '')}"
                        )

                        colA, colB = st.columns(2)
                        with colA:
                            with st.expander("✅ Keywords Integrated"):
                                st.write(
                                    ", ".join(
                                        result_data.get("added_keywords", [])
                                    )
                                )
                        with colB:
                            with st.expander(
                                "❌ Omitted Keywords (Couldn't fit naturally)"
                            ):
                                st.write(
                                    ", ".join(
                                        result_data.get("missing_keywords", [])
                                    )
                                )

                        clean_name = (
                            result_data.get("name", "Sri_Charan_Ravva")
                            .strip()
                            .lower()
                            .replace(" ", "_")
                        )
                        clean_company = (
                            company_name.strip().lower().replace(" ", "_")
                        )
                        clean_title = (
                            target_title.strip().lower().replace(" ", "_")
                        )

                        if not clean_company:
                            clean_company = "optimized"
                        if not clean_title:
                            clean_title = "role"

                        # Unique ID for Supabase Primary Key
                        job_id = f"{clean_company}_{clean_title}"

                        # --- AUTO-SAVE TO SUPABASE TRACKER ---
                        save_success = save_application_supabase(
                            job_id=job_id,
                            company=company_name or "Target Company",
                            title=target_title or "Marketing Analyst",
                            location=st.session_state.get("selected_location", "USA"),
                            job_url=st.session_state.get("selected_job_url", ""),
                            ats_score=result_data.get("estimated_ats_score", "N/A"),
                            status="Tailored",
                            notes=result_data.get("explanation", ""),
                            pdf_bytes=pdf_buffer.getvalue()
                        )
                        if save_success:
                            st.toast("Saved application record and PDF to Supabase!", icon="☁️")

                        st.download_button(
                            label="⬇️ Download Optimized Resume (.pdf)",
                            data=pdf_buffer,
                            file_name=f"{clean_name}_{clean_company}.pdf",
                            mime="application/pdf",
                            type="primary",
                        )
                    else:
                        st.error(
                            "The PDF rendering engine encountered a layout error"
                            " processing the generated text."
                        )

                except Exception as e:
                    st.error(
                        "An error occurred during calculation or runtime"
                        f" processing: {e}"
                    )
        else:
            st.warning(
                "Please make sure you have uploaded a resume file and provided a"
                " target job description."
            )


# ==========================================
# TAB 3: APPLICATION TRACKER DASHBOARD
# ==========================================
with tab3:
    st.header("📊 Application Tracker Dashboard")
    st.caption("Manage your job hunt, track application progress, download tailored resumes, and save interview notes.")

    apps_df = fetch_all_applications_supabase()

    if apps_df.empty:
        st.info("No applications tracked yet. Tailor a resume to populate this tracker!")
    else:
        # Metrics Row
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Total Tracked", len(apps_df))
        col_m2.metric("Applied", len(apps_df[apps_df["status"] == "Applied"]))
        col_m3.metric("Interviewing 🎯", len(apps_df[apps_df["status"] == "Interviewing"]))
        col_m4.metric("Offers 🍾", len(apps_df[apps_df["status"] == "Offer"]))
        
        # Calculate Average ATS score safely
        ats_numeric = pd.to_numeric(apps_df['ats_score'].astype(str).str.rstrip('%'), errors='coerce')
        avg_ats = f"{ats_numeric.mean():.1f}%" if not ats_numeric.isna().all() else "N/A"
        col_m5.metric("Avg ATS Match", avg_ats)

        st.divider()

        # Status Filter
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            status_filter = st.selectbox(
                "Filter by Status:",
                options=["All", "Tailored", "Applied", "Interviewing", "Offer", "Rejected"]
            )
        
        filtered_df = apps_df if status_filter == "All" else apps_df[apps_df["status"] == status_filter]

        # Applications Table
        st.dataframe(
            filtered_df[["company", "title", "status", "ats_score", "job_url", "created_at"]],
            column_config={
                "job_url": st.column_config.LinkColumn("Job Posting"),
                "created_at": st.column_config.DatetimeColumn("Date Tracked", format="D MMM YYYY, HH:mm"),
            },
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Application Detail & Status Updater
        st.subheader("📝 Application Manager")
        
        job_options = {
            f"{row['company']} - {row['title']} ({row['status']})": row['job_id']
            for _, row in apps_df.iterrows()
        }
        
        selected_label = st.selectbox("Select an application to manage:", list(job_options.keys()))
        selected_id = job_options[selected_label]
        
        app = apps_df[apps_df["job_id"] == selected_id].iloc[0]

        col_d1, col_d2 = st.columns([2, 1])

        with col_d1:
            st.markdown(f"### **{app['title']}** at **{app['company']}**")
            if app["job_url"]:
                st.link_button("🔗 Open Original Job Posting", app["job_url"])

            with st.form(f"update_form_{app['job_id']}"):
                current_status = app["status"] if app["status"] in ["Tailored", "Applied", "Interviewing", "Offer", "Rejected"] else "Tailored"
                new_status = st.selectbox(
                    "Update Status:",
                    options=["Tailored", "Applied", "Interviewing", "Offer", "Rejected"],
                    index=["Tailored", "Applied", "Interviewing", "Offer", "Rejected"].index(current_status)
                )
                
                new_notes = st.text_area("Notes (Recruiter contacts, call dates, follow-up actions):", value=app["notes"] if app["notes"] else "")
                
                if st.form_submit_button("💾 Save Status & Notes"):
                    update_application_status_supabase(app["job_id"], new_status, new_notes)
                    st.toast("Application status updated!", icon="✅")
                    st.rerun()

        with col_d2:
            st.markdown("#### **Resume File**")
            st.info(f"**ATS Match Score:** {app['ats_score']}")
            
            if app["pdf_path"]:
                pdf_bytes = download_pdf_from_supabase(app["pdf_path"])
                if pdf_bytes:
                    st.download_button(
                        label="⬇️ Retrieve Saved Resume PDF",
                        data=pdf_bytes,
                        file_name=f"{app['company']}_{app['title']}_Resume.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.write("No PDF associated with this record.")

            st.markdown("---")
            if st.button("🗑️ Delete Record", type="secondary"):
                delete_application_supabase(app["job_id"], app["pdf_path"])
                st.toast("Application record removed.")
                st.rerun()
