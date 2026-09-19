# Prep AI V3

Prep AI V3 is an advanced RAG learning platform with two learning modes: **Personalized Learning** and **Database Learning**.

## GitHub structure

```text
prep-ai-v3/
├── app.py
├── requirements.txt
├── readme.md
└── faiss_index/
    ├── database.faiss
    ├── metadata.json
    └── config.json
```

### Database documents are NOT uploaded

Do **not** put the original Biology/Chemistry/Physics/English PDFs in GitHub. The application only needs the pre-built retrieval artifacts:

- `database.faiss` — vector index
- `metadata.json` — chunk text plus filename/page/subject metadata needed to reconstruct RAG context
- `config.json` — ingestion configuration

`embeddings.npy` is optional and can be omitted because the vectors are already stored inside `database.faiss`.

The metadata contains derived chunk text, so it is not literally “embeddings only”; a RAG generator needs the text corresponding to retrieved vectors. The original PDFs remain outside the application.

## Architecture

```text
                       Prep AI V3
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
   Personalized Learning       Database Learning
             │                           │
   PDF/DOCX/TXT/MD               Subject selector
   or public Google Drive file/folder       Biology/Chemistry/Physics/English
             │                           │
             ▼                           ▼
       Extraction                  Pre-built FAISS
             │                    + metadata.json
             ▼                           │
         Chunking                         │
             │                           │
             ▼                           │
   Sentence Transformer                 │
       embeddings                       │
             │                           │
             ▼                           │
        FAISS index ◄───────────────────┘
             │
             └──────────────┬──────────────┘
                            ▼
                    Hybrid Retrieval
                  /                    \
          Semantic Search          Keyword Search
                  \                    /
                   └───────┬────────────┘
                           ▼
                    Ranked RAG Context
                           ▼
                         Groq
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
            MCQs     Explanation       Quiz
              │            │            │
              └────────────┼────────────┘
                           ▼
                 Sources + PDF export
```

## Database RAG

The Colab ingestion pipeline creates the database before deployment:

```text
Google Drive Database PDFs
        ↓
Colab + ingest.py
        ↓
Page extraction
        ↓
Overlapping chunks
        ↓
Sentence Transformer embeddings
        ↓
FAISS IndexFlatIP
        ↓
faiss_index/
```

At runtime, the Streamlit app never downloads the database PDFs. It loads the FAISS index and metadata directly. Normalized vectors with inner-product search are used as cosine similarity retrieval.

## Personalized RAG

Students can upload PDF, DOCX, TXT and MD material. A public Google Drive **file** link can also be supplied. The material follows the same extraction → chunking → embedding → FAISS → hybrid-search pipeline.

## Study modes

- **MCQs** — MDCAT-style questions with answer key and explanation.
- **Answer explanation** — answer generated only from retrieved context.
- **Quiz** — interactive MCQs with hidden answers, score, percentage, review and PDF export.

## Hybrid search

The app combines semantic similarity and keyword overlap:

```text
Hybrid = semantic_weight × semantic +
         (1 - semantic_weight) × keyword
```

Database mode searches a broad FAISS candidate set and then filters candidates by the selected subject before final hybrid ranking.

## Caching

`st.cache_resource` is used for the embedding model and database index so expensive resources are not recreated on normal Streamlit reruns.

## Groq

The app uses the official Groq Python SDK and supports model selection in Settings. MCQ generation uses JSON output parsing so the quiz structure is predictable.

Create `.streamlit/secrets.toml` locally or in your deployment settings:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Never hardcode the key in `app.py` and never commit `secrets.toml`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Current September 2026 package baseline

This build pins the versions available for the current 2026 environment: Streamlit 1.64.0, Groq 1.7.0, Sentence Transformers 6.1.0 and FAISS CPU 1.15.1.

## Important repository note

If the FAISS artifacts are too large for normal GitHub repository limits, keep the source code in GitHub and store the `faiss_index` artifacts in Git LFS or object storage, then add a small download/bootstrap mechanism later. Do not solve this by committing the original PDF library.

## Security

Do not commit API keys, private Drive credentials, original copyrighted/private PDFs, or `.streamlit/secrets.toml`.


## Google Drive Personalized Learning

The Personalized Learning input accepts either:

- a public Google Drive **file link**
- a public Google Drive **folder link**

For a folder link, Prep AI recursively downloads the folder contents and processes only:

- PDF
- DOCX
- TXT
- MD

The app preserves the original filename and relative folder path. It does not treat the Google Drive folder's HTML page as a TXT document.

The extracted-document table reports:

- filename
- source path
- file type
- character count
- pages for PDFs
- chunks created

For DOCX/TXT/MD, physical PDF-style page numbers are shown as `N/A` because those formats do not reliably contain page metadata.

Current gdown releases automatically parse Google Drive share URLs; the old `fuzzy=True` argument must not be used because it was removed from current gdown releases.
