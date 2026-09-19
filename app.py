import os
import io
import re
import json
import hashlib
import tempfile
import zipfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


st.set_page_config(page_title="Prep AI V3", page_icon="🎓", layout="wide")

APP_ROOT = Path(__file__).resolve().parent
DATABASE_DIR = APP_ROOT / "faiss_index"
DATABASE_INDEX = DATABASE_DIR / "database.faiss"
DATABASE_METADATA = DATABASE_DIR / "metadata.json"
DATABASE_CONFIG = DATABASE_DIR / "config.json"
DATABASE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATABASE_SUBJECTS = ["Biology", "Chemistry", "Physics", "English"]

# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    records = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            records.append({"text": text, "filename": filename, "page": page_number})
    return records


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
    return [{"text": text, "filename": filename, "page": None}] if text else []


def extract_txt(file_bytes, filename):
    text = re.sub(r"\s+", " ", file_bytes.decode("utf-8", errors="ignore")).strip()
    return [{"text": text, "filename": filename, "page": None}] if text else []


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`>-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [{"text": text, "filename": filename, "page": None}] if text else []


def extract_document(file_bytes, filename):
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(file_bytes, filename)
    if ext == ".docx":
        return extract_docx(file_bytes, filename)
    if ext == ".txt":
        return extract_txt(file_bytes, filename)
    if ext == ".md":
        return extract_md(file_bytes, filename)
    raise ValueError(f"Unsupported file type: {ext}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text_records(records, chunk_size=900, overlap=150):
    chunks = []
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    for record in records:
        words = record["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end]).strip()
            if text:
                chunks.append({
                    "text": text,
                    "filename": record["filename"],
                    "page": record.get("page"),
                })
            if end >= len(words):
                break
            start = end - overlap
    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name=DATABASE_EMBEDDING_MODEL):
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Loading database FAISS index...")
def load_database():
    if not DATABASE_INDEX.exists() or not DATABASE_METADATA.exists():
        raise FileNotFoundError(
            "Database artifacts are missing. Put database.faiss, metadata.json "
            "and config.json inside the faiss_index folder beside app.py."
        )
    index = faiss.read_index(str(DATABASE_INDEX))
    with DATABASE_METADATA.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    config = {}
    if DATABASE_CONFIG.exists():
        with DATABASE_CONFIG.open("r", encoding="utf-8") as f:
            config = json.load(f)
    if index.ntotal != len(metadata):
        raise ValueError("FAISS vector count does not match metadata count.")
    return index, metadata, config


def build_vector_index(chunks):
    model = load_embedding_model()
    embeddings = model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def fingerprint_chunks(chunks):
    payload = json.dumps(chunks, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# -----------------------------
# Hybrid retrieval
# -----------------------------
def important_words(text):
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
        "is", "are", "was", "were", "with", "from", "by", "as", "at",
        "what", "which", "who", "how", "why", "when", "where", "that",
        "this", "these", "those", "chapter", "topic",
    }
    return {
        w for w in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(w) > 2 and w not in stopwords
    }


def keyword_scores(query, chunks):
    qwords = important_words(query)
    scores = []
    for chunk in chunks:
        cwords = important_words(chunk["text"])
        scores.append(len(qwords & cwords) / len(qwords) if qwords else 0.0)
    return np.array(scores, dtype="float32")


def hybrid_search(query, chunks, index, top_k=8, semantic_weight=0.7, subject=None):
    model = load_embedding_model()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    if index is None or not chunks:
        return []

    candidate_k = min(index.ntotal, max(top_k * 12, 50))
    semantic_scores, indices = index.search(q, candidate_k)

    qwords = important_words(query)
    candidates = []
    for semantic, idx in zip(semantic_scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[int(idx)]
        if subject and chunk.get("subject", "").lower() != subject.lower():
            continue
        cwords = important_words(chunk["text"])
        keyword = len(qwords & cwords) / len(qwords) if qwords else 0.0
        candidates.append((int(idx), float(semantic), float(keyword)))

    if not candidates:
        return []

    sem_values = [x[1] for x in candidates]
    key_values = [x[2] for x in candidates]
    sem_min, sem_max = min(sem_values), max(sem_values)
    key_min, key_max = min(key_values), max(key_values)

    def norm(value, low, high):
        return 0.0 if high - low < 1e-9 else (value - low) / (high - low)

    results = []
    for idx, semantic, keyword in candidates:
        sem_n = norm(semantic, sem_min, sem_max)
        key_n = norm(keyword, key_min, key_max)
        hybrid = semantic_weight * sem_n + (1 - semantic_weight) * key_n
        results.append((hybrid, semantic, keyword, idx))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def build_context(chunks, results):
    parts = []
    for n, (_, _, _, idx) in enumerate(results, start=1):
        c = chunks[idx]
        parts.append(
            f"[SOURCE {n}]\nSubject: {c.get('subject') or 'Personalized'}\nFile: {c['filename']}\nPage: {c.get('page') or 'N/A'}\nText: {c['text']}"
        )
    return "\n\n".join(parts)


def source_label(chunk):
    return f"{chunk['filename']} — page {chunk['page']}" if chunk.get("page") else chunk["filename"]


# -----------------------------
# Groq
# -----------------------------
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to Streamlit secrets or environment variables.")
    return Groq(api_key=api_key)


def ask_groq(topic, context, mode, difficulty, count, instruction, model_name):
    client = get_groq_client()

    if mode in {"MCQs", "Quiz"}:
        system_prompt = f"""
You are an expert MDCAT exam question writer.
Create up to {count} high-quality single-best-answer MCQs using ONLY the supplied SOURCE CONTEXT.
Do not use outside knowledge. If the context does not support enough questions, generate fewer.
Difficulty: {difficulty}
- Easy: direct recall and straightforward understanding.
- Medium: conceptual understanding, comparison, application, interpretation.
- Hard: multi-step reasoning and challenging but fair distractors.
- Mixed MDCAT Level: a mixture of easy, medium, and hard.
Each question must have exactly four options A, B, C, D and exactly one correct answer.
Return ONLY valid JSON:
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct_answer":"A","explanation":"...","source":1}}]}}
Rules:
- correct_answer must be A, B, C, or D.
- source must be the SOURCE number supporting the question.
- Do not reveal the answer in the question or options.
- Keep questions at MDCAT level and avoid duplicates.
- Do not invent facts outside the context.
"""
    else:
        system_prompt = """
You are an expert study assistant. Answer ONLY from the supplied SOURCE CONTEXT.
If the information is not available, clearly say it is not available in the provided material.
"""

    user_prompt = (
        f"Student chapter/topic: {topic}\nDifficulty: {difficulty}\n"
        f"Additional instruction: {instruction or 'None'}\n\nSOURCE CONTEXT:\n{context}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2 if mode in {"MCQs", "Quiz"} else 0.1,
        max_tokens=8000,
    )
    return response.choices[0].message.content.strip()


def parse_mcq_json(raw):
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    cleaned = []
    for i, item in enumerate(data.get("questions", []), start=1):
        if not isinstance(item, dict) or not item.get("question"):
            continue
        options = item.get("options", {})
        if not all(str(options.get(x, "")).strip() for x in ["A", "B", "C", "D"]):
            continue
        correct = str(item.get("correct_answer", "")).upper().strip()
        if correct not in {"A", "B", "C", "D"}:
            continue
        try:
            source = int(item.get("source"))
        except (TypeError, ValueError):
            source = None
        cleaned.append({
            "id": f"q{i}",
            "question": str(item["question"]).strip(),
            "options": {x: str(options[x]).strip() for x in ["A", "B", "C", "D"]},
            "correct_answer": correct,
            "explanation": str(item.get("explanation", "")).strip(),
            "source": source,
        })
    if not cleaned:
        raise ValueError("No valid MCQs were returned by Groq.")
    return cleaned


# -----------------------------
# Google Drive + document processing
# -----------------------------
def detect_drive_file_type(file_bytes, original_url=""):
    """Detect a supported document type when Google Drive does not provide a filename."""
    if file_bytes.startswith(b"%PDF"):
        return ".pdf"

    # DOCX is a ZIP container with [Content_Types].xml and word/document.xml.
    if file_bytes.startswith(b"PK"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                names = set(zf.namelist())
                if "word/document.xml" in names and "[Content_Types].xml" in names:
                    return ".docx"
        except zipfile.BadZipFile:
            pass

    # Fall back to the URL extension if one is explicitly present.
    url_path = original_url.split("?", 1)[0].split("#", 1)[0].lower()
    for ext in (".pdf", ".docx", ".txt", ".md"):
        if url_path.endswith(ext):
            return ext

    # Drive often does not expose the original extension. Treat readable
    # non-binary content as TXT so it can also cover Markdown files.
    try:
        decoded = file_bytes.decode("utf-8")
        if decoded.strip():
            return ".txt"
    except UnicodeDecodeError:
        pass

    return ""


def load_drive_file(url):
    """Download one public Google Drive file and give it a usable extension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "drive_download"
        downloaded = gdown.download(
            url=url,
            output=str(output_path),
            quiet=True,
        )

        if not downloaded:
            raise ValueError(
                "Google Drive download failed. Make sure the file is shared "
                "as Anyone with the link / Viewer and that the link is a file link."
            )

        path = Path(downloaded)
        file_bytes = path.read_bytes()

    extension = detect_drive_file_type(file_bytes, url)
    if not extension:
        raise ValueError(
            "Google Drive file type could not be detected. "
            "Please use a PDF, DOCX, TXT, or MD file."
        )

    names = {
        ".pdf": "Google Drive document.pdf",
        ".docx": "Google Drive document.docx",
        ".txt": "Google Drive document.txt",
        ".md": "Google Drive document.md",
    }

    return {"name": names[extension], "bytes": file_bytes}


def process_files(items, chunk_size, overlap):
    records, info = [], []
    for item in items:
        extracted = extract_document(item["bytes"], item["name"])
        records.extend(extracted)
        pages = {r["page"] for r in extracted if r.get("page") is not None}
        info.append({
            "filename": item["name"],
            "characters": sum(len(r["text"]) for r in extracted),
            "pages": len(pages) if pages else None,
        })
    chunks = chunk_text_records(records, chunk_size, overlap)
    return chunks, info


# -----------------------------
# PDF exports
# -----------------------------
def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def create_mcq_pdf(questions, title, include_answers=True, user_answers=None, score=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45, title=title)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=18)
    qstyle = ParagraphStyle("Q", parent=styles["Heading3"], spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("B", parent=styles["BodyText"], leading=14, spaceAfter=5)
    story = [Paragraph(esc(title), title_style)]
    if score:
        story.append(Paragraph(f"<b>Score:</b> {score['correct']} / {score['total']} &nbsp;&nbsp; <b>Percentage:</b> {score['percentage']:.1f}%", body))
        story.append(Spacer(1, 10))
    for n, q in enumerate(questions, 1):
        story.append(Paragraph(f"{n}. {esc(q['question'])}", qstyle))
        for letter in ["A", "B", "C", "D"]:
            story.append(Paragraph(f"<b>{letter}.</b> {esc(q['options'][letter])}", body))
        if user_answers is not None:
            story.append(Paragraph(f"<b>Your Answer:</b> {esc(user_answers.get(q['id'], 'Not attempted'))}", body))
        if include_answers:
            story.append(Paragraph(f"<b>Correct Answer:</b> {esc(q['correct_answer'])}", body))
            if q.get("explanation"):
                story.append(Paragraph(f"<b>Explanation:</b> {esc(q['explanation'])}", body))
        story.append(Spacer(1, 8))
    doc.build(story)
    return buffer.getvalue()


def create_explanation_pdf(topic, answer, chunks, title):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45, title=title)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=18)
    body = ParagraphStyle("B", parent=styles["BodyText"], leading=14, spaceAfter=9)
    heading = ParagraphStyle("H", parent=styles["Heading2"], spaceBefore=12, spaceAfter=8)
    story = [Paragraph(esc(title), title_style), Paragraph(f"<b>Topic:</b> {esc(topic)}", body)]
    story.append(Paragraph(esc(answer).replace("\n", "<br/>"), body))
    story.append(Paragraph("Retrieved Sources", heading))
    for i, c in enumerate(chunks, 1):
        story.append(Paragraph(f"<b>Source {i}:</b> {esc(c['filename'])} — Page {esc(c.get('page') or 'N/A')}", body))
        story.append(Paragraph(esc(c["text"]), body))
    doc.build(story)
    return buffer.getvalue()


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "learning_mode": "Personalized Learning",
    "chunks": [], "index": None, "embeddings": None, "fingerprint": None,
    "document_info": [], "last_results": [], "last_context": "", "last_answer": "",
    "generated_questions": [], "quiz_submitted": False, "quiz_answers": {},
    "quiz_score": None, "quiz_nonce": 0, "last_mode": "", "last_topic": "",
    "ui_color": "Blue", "selected_model": "openai/gpt-oss-120b",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# UI settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    color_options = {
        "Blue": "#2563EB", "Green": "#16A34A", "Purple": "#7C3AED",
        "Orange": "#EA580C", "Red": "#DC2626",
    }
    st.session_state["ui_color"] = st.selectbox(
        "Change UI color", list(color_options),
        index=list(color_options).index(st.session_state["ui_color"]),
    )
    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    st.session_state["selected_model"] = st.selectbox(
        "LLM Model", models,
        index=models.index(st.session_state["selected_model"]),
    )
    st.markdown(f"""
    <style>
    div.stButton > button[kind="primary"] {{background-color:{color_options[st.session_state['ui_color']]}; border-color:{color_options[st.session_state['ui_color']]};}}
    </style>
    """, unsafe_allow_html=True)

    st.divider()
    st.header("Retrieval Settings")
    chunk_size = st.slider("Chunk size (words)", 300, 1800, 900, 100)
    overlap = st.slider("Chunk overlap (words)", 0, 400, 150, 25)
    top_k = st.slider("Retrieved chunks", 2, 15, 8)
    semantic_weight = st.slider("Semantic search weight", 0.0, 1.0, 0.7, 0.05)
    st.info("Add GROQ_API_KEY to .streamlit/secrets.toml or the environment.")


st.title("🎓 Prep AI V3")
st.caption("Advanced RAG learning platform — Personalized Learning + Database Learning")

st.subheader("1. Choose Learning Mode")
learning_mode = st.radio(
    "Learning mode",
    ["Personalized Learning", "Database Learning"],
    horizontal=True,
)
st.session_state["learning_mode"] = learning_mode

# ------------------------------------------------------------
# Personalized learning
# ------------------------------------------------------------
if learning_mode == "Personalized Learning":
    st.info("Upload your own study material. These files are processed in the current session and are not stored in the database.")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    drive_url = st.text_input(
        "Optional Google Drive file link",
        placeholder="Paste a public Google Drive file link",
    )

    if uploaded or drive_url.strip():
        if st.button("Process Personalized Material", type="primary"):
            items=[]
            if uploaded:
                items.extend({"name": f.name, "bytes": f.getvalue()} for f in uploaded)
            if drive_url.strip():
                try:
                    with st.spinner("Downloading Google Drive file..."):
                        items.append(load_drive_file(drive_url.strip()))
                except Exception as exc:
                    st.error(f"Google Drive error: {exc}")

            if items:
                try:
                    with st.spinner("Extracting text, chunking, and building FAISS index..."):
                        chunks, info = process_files(items, chunk_size, overlap)
                        if not chunks:
                            raise ValueError("No text could be extracted from the supplied files.")
                        index, embeddings = build_vector_index(chunks)
                        st.session_state["chunks"] = chunks
                        st.session_state["index"] = index
                        st.session_state["embeddings"] = embeddings
                        st.session_state["fingerprint"] = fingerprint_chunks(chunks)
                        st.session_state["document_info"] = info
                        st.session_state["generated_questions"] = []
                        st.session_state["last_results"] = []
                        st.session_state["quiz_submitted"] = False
                        st.session_state["quiz_answers"] = {}
                        st.session_state["quiz_score"] = None
                    st.success(f"Processed {len(info)} document(s) and created {len(chunks)} chunks.")
                except Exception as exc:
                    st.error(f"Processing error: {exc}")

    if st.session_state["document_info"]:
        st.subheader("2. Extracted document information")
        st.dataframe(st.session_state["document_info"], use_container_width=True, hide_index=True)
        st.caption(f"Total chunks: {len(st.session_state['chunks'])}")

# ------------------------------------------------------------
# Database learning
# ------------------------------------------------------------
else:
    st.info("Database Learning uses only the pre-built FAISS index and metadata. The original PDFs are NOT required by the app.")
    try:
        db_index, db_chunks, db_config = load_database()
        st.success(f"Database loaded — {db_index.ntotal:,} vectors")
        subject = st.selectbox("Select Subject", DATABASE_SUBJECTS)
        subject_count = sum(1 for c in db_chunks if c.get("subject", "").lower() == subject.lower())
        st.caption(f"{subject}: {subject_count:,} indexed chunks")
    except Exception as exc:
        db_index, db_chunks, db_config = None, [], {}
        subject = DATABASE_SUBJECTS[0]
        st.error(f"Database loading error: {exc}")
        st.info("Put faiss_index/database.faiss, metadata.json, and config.json beside app.py.")

# ------------------------------------------------------------
# Study controls
# ------------------------------------------------------------
st.subheader("3. Study")
topic = st.text_input("Chapter / topic", placeholder="Example: Cell membrane, Genetics, Newton's laws, Tenses...")
mode = st.selectbox("Mode", ["MCQs", "Answer explanation", "Quiz"])
difficulty = st.selectbox("Difficulty level", ["Easy", "Medium", "Hard", "Mixed MDCAT Level"], index=3, disabled=(mode == "Answer explanation"))
count = st.number_input("Number of MCQs", 1, 100, 20, 1, disabled=(mode == "Answer explanation"))
instruction = st.text_area("Optional instruction", placeholder="Example: Focus on conceptual questions.")
button_label = {"MCQs": "Generate MCQs", "Answer explanation": "Get Explanation", "Quiz": "Start Quiz"}[mode]

if st.button(button_label, type="primary"):
    if not topic.strip():
        st.warning("Please enter a chapter or topic.")
    else:
        try:
            if learning_mode == "Personalized Learning":
                chunks = st.session_state["chunks"]
                index = st.session_state["index"]
                subject_filter = None
                if not chunks or index is None:
                    st.warning("Please process personalized material first.")
                    st.stop()
            else:
                chunks = db_chunks
                index = db_index
                subject_filter = subject
                if not chunks or index is None:
                    st.warning("Database is not available.")
                    st.stop()

            with st.spinner("Running hybrid RAG and generating response..."):
                results = hybrid_search(topic, chunks, index, top_k, semantic_weight, subject_filter)
                if not results:
                    st.warning("No relevant material was retrieved for this topic.")
                    st.stop()
                context = build_context(chunks, results)
                raw = ask_groq(
                    topic, context, mode, difficulty, int(count), instruction,
                    st.session_state["selected_model"],
                )

                st.session_state["last_results"] = results
                st.session_state["last_context"] = context
                st.session_state["last_topic"] = topic
                st.session_state["last_mode"] = mode
                st.session_state["last_answer"] = ""
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_score"] = None

                if mode in {"MCQs", "Quiz"}:
                    st.session_state["quiz_nonce"] += 1
                    questions = parse_mcq_json(raw)[:int(count)]
                    for i, q in enumerate(questions, 1):
                        q["id"] = f"q{st.session_state['quiz_nonce']}_{i}"
                    st.session_state["generated_questions"] = questions
                else:
                    st.session_state["generated_questions"] = []
                    st.session_state["last_answer"] = raw

            st.success("Response generated.")
        except Exception as exc:
            st.error(f"Generation error: {exc}")

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
if st.session_state["generated_questions"] and st.session_state["last_mode"] == "MCQs":
    st.divider()
    st.subheader("🧠 Generated MCQs with Answer Key")
    questions = st.session_state["generated_questions"]
    for n, q in enumerate(questions, 1):
        st.markdown(f"### {n}. {q['question']}")
        for letter in ["A", "B", "C", "D"]:
            st.write(f"**{letter}.** {q['options'][letter]}")
        st.success(f"**Answer:** {q['correct_answer']}")
        st.info(f"**Explanation:** {q['explanation']}")
        st.divider()
    pdf = create_mcq_pdf(questions, f"Prep AI MCQs - {st.session_state['last_topic']}", True)
    st.download_button("⬇️ Download MCQs PDF", pdf, "prep_ai_mcqs.pdf", "application/pdf")

if st.session_state["generated_questions"] and st.session_state["last_mode"] == "Quiz":
    st.divider()
    st.subheader("📝 Interactive Quiz")
    questions = st.session_state["generated_questions"]
    if not st.session_state["quiz_submitted"]:
        with st.form("quiz_form"):
            answers = {}
            for n, q in enumerate(questions, 1):
                st.markdown(f"### Question {n}")
                st.write(q["question"])
                opts = q["options"]
                selected = st.radio(
                    "Choose an answer:",
                    ["Not attempted", "A", "B", "C", "D"],
                    format_func=lambda x, opts=opts: "Not attempted" if x == "Not attempted" else f"{x}. {opts[x]}",
                    key=f"quiz_{q['id']}",
                )
                answers[q["id"]] = selected
            submit = st.form_submit_button("Submit Quiz", type="primary")
        if submit:
            correct = sum(answers.get(q["id"]) == q["correct_answer"] for q in questions)
            attempted = sum(answers.get(q["id"]) in {"A", "B", "C", "D"} for q in questions)
            total = len(questions)
            st.session_state["quiz_answers"] = answers
            st.session_state["quiz_score"] = {
                "correct": correct, "wrong": attempted-correct,
                "unattempted": total-attempted, "total": total,
                "percentage": (correct/total*100) if total else 0,
            }
            st.session_state["quiz_submitted"] = True
            st.rerun()
    else:
        score = st.session_state["quiz_score"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Score", f"{score['correct']} / {score['total']}")
        c2.metric("Percentage", f"{score['percentage']:.1f}%")
        c3.metric("Wrong", score["wrong"])
        c4.metric("Unattempted", score["unattempted"])
        for n,q in enumerate(questions,1):
            user = st.session_state["quiz_answers"].get(q["id"], "Not attempted")
            status = "✅ Correct" if user == q["correct_answer"] else ("⚪ Unattempted" if user == "Not attempted" else "❌ Incorrect")
            st.markdown(f"### {n}. {status}")
            st.write(q["question"])
            for letter in ["A","B","C","D"]:
                st.write(f"**{letter}.** {q['options'][letter]}")
            st.write(f"**Your answer:** {user}")
            st.write(f"**Correct answer:** {q['correct_answer']}")
            st.info(f"**Explanation:** {q['explanation']}")
            st.divider()
        pdf=create_mcq_pdf(questions, f"Prep AI Quiz Review - {st.session_state['last_topic']}", True, st.session_state["quiz_answers"], score)
        st.download_button("⬇️ Download Quiz PDF", pdf, "prep_ai_quiz.pdf", "application/pdf")
        if st.button("🔄 Generate New Quiz"):
            st.session_state["generated_questions"]=[]
            st.session_state["quiz_submitted"]=False
            st.session_state["quiz_answers"]={}
            st.session_state["quiz_score"]=None
            st.rerun()

if st.session_state["last_answer"] and st.session_state["last_mode"] == "Answer explanation":
    st.divider()
    st.subheader("📖 Answer Explanation")
    st.markdown(st.session_state["last_answer"])
    source_chunks = [
        (st.session_state["chunks"] if learning_mode == "Personalized Learning" else db_chunks)[idx]
        for _,_,_,idx in st.session_state["last_results"]
    ]
    pdf=create_explanation_pdf(st.session_state["last_topic"], st.session_state["last_answer"], source_chunks, f"Prep AI Explanation - {st.session_state['last_topic']}")
    st.download_button("⬇️ Download Explanation PDF", pdf, "prep_ai_explanation.pdf", "application/pdf")

if st.session_state["last_results"]:
    st.divider()
    st.subheader("🔎 Retrieved RAG Sources")
    source_chunks = st.session_state["chunks"] if learning_mode == "Personalized Learning" else db_chunks
    for n, (_, semantic, keyword, idx) in enumerate(st.session_state["last_results"],1):
        chunk=source_chunks[idx]
        page=chunk.get("page") or "N/A"
        subject_label=chunk.get("subject") or "Personalized"
        with st.expander(f"Source {n}: {chunk['filename']} | Page {page} | {subject_label}"):
            st.write(chunk["text"])
            st.caption(f"Semantic: {semantic:.3f} | Keyword: {keyword:.3f}")
