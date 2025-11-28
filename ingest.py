# ingest.py
import os
import re
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

USE_GEMINI = os.getenv("USE_GEMINI", "0") == "1"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBED_ENDPOINT = os.getenv("GEMINI_EMBED_ENDPOINT", "https://api.gemini.example/v1/embeddings")  # replace if needed

DOCS_DIR = Path("docs")
INDEX_FILE = "vectorstore.faiss"
META_FILE = "docs_meta.npy"
CHUNK_SIZE_CHARS = 1200  # ~300 tokens (rough)
CHUNK_OVERLAP = 200

# ---------- simple text extraction helpers ----------
def extract_text_from_file(path: Path):
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    # very small pdf/docx fallback: attempt naive text extraction to keep deps small
    # If you want high quality extraction, install 'pdfplumber' and 'python-docx'
    try:
        import fitz  # PyMuPDF (optional)
    except Exception:
        fitz = None
    if path.suffix.lower() == ".pdf" and fitz:
        txt = ""
        doc = fitz.open(path)
        for pg in doc:
            txt += pg.get_text()
        return txt
    # fallback for docx
    try:
        from docx import Document
    except Exception:
        Document = None
    if path.suffix.lower() == ".docx" and Document:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    # last resort: return empty or file bytes decoded
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

# ---------- chunk text ----------
def chunk_text(text, chunk_size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------- embeddings ----------
def load_local_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model

def embed_texts_local(model, texts):
    # returns numpy array (n, d)
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embs

def embed_texts_gemini(texts):
    # Minimal HTTP-based example. You must set GEMINI_API_KEY and endpoint correctly.
    # The exact request/response fields depend on Gemini's embedding endpoint.
    import requests
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    # payload shape may differ for the real Gemini API; adapt as needed.
    payload = {"model": "gemini-embeddings-001", "input": texts}
    resp = requests.post(GEMINI_EMBED_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # Convert response to numpy array — adapt indexing to real response format.
    # Expecting: {"data":[{"embedding": [...]}, ...]}
    embs = []
    if "data" in data:
        for itm in data["data"]:
            embs.append(np.array(itm["embedding"], dtype=np.float32))
    else:
        raise RuntimeError("Unexpected Gemini response: " + str(data))
    return np.vstack(embs)

# ---------- build faiss ----------
def build_faiss(embs: np.ndarray):
    import faiss
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # inner-product; we will normalize embeddings
    faiss.normalize_L2(embs)
    index.add(embs)
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved FAISS index to {INDEX_FILE}")

def save_meta(meta_list):
    np.save(META_FILE, np.array(meta_list, dtype=object), allow_pickle=True)
    print(f"Saved metadata to {META_FILE}")

def main():
    files = [p for p in DOCS_DIR.glob("*") if p.is_file()]
    if not files:
        print("No files found in docs/ — put your .txt/.pdf/.docx there and re-run.")
        return
    all_texts = []
    meta = []
    for f in files:
        text = extract_text_from_file(f)
        if not text or len(text.strip()) < 20:
            continue
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_texts.append(c)
            meta.append({"source": str(f.name), "chunk_id": i, "text_start": None})
    print(f"Prepared {len(all_texts)} chunks from {len(files)} files.")

    # Embeddings
    if USE_GEMINI:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        print("Using Gemini embeddings (remote).")
        embs = embed_texts_gemini(all_texts)
    else:
        print("Using local sentence-transformers embedding model (offline).")
        model = load_local_embedder()
        embs = embed_texts_local(model, all_texts)

    # Normalize & store index
    # convert to float32
    embs = embs.astype(np.float32)
    # normalize for cosine similarity if using IndexFlatIP
    import faiss
    faiss.normalize_L2(embs)
    build_faiss(embs)
    save_meta(meta)
    # save raw texts too (for retrieval)
    with open("chunks_texts.json", "w", encoding="utf-8") as fh:
        json.dump(all_texts, fh)
    print("Done.")

if __name__ == "__main__":
    main()
