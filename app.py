import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# -------------------------------
#  CONFIG
# -------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL = SentenceTransformer(EMBEDDING_MODEL)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------
#  LOAD DOCUMENTS
# -------------------------------
def load_documents(folder="docs"):
    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if os.path.isfile(path) and path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append((file, content))
    return docs

# -------------------------------
#  CHUNKING
# -------------------------------
def chunk_text(text, size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

# -------------------------------
#  BUILD KNOWLEDGE BASE
# -------------------------------
documents = load_documents()
chunks = []
for doc_name, content in documents:
    for idx, chunk in enumerate(chunk_text(content)):
        chunks.append({
            "doc": doc_name,
            "chunk": chunk,
            "embedding": MODEL.encode(chunk)
        })

# -------------------------------
#  RETRIEVAL FUNCTION
# -------------------------------
def retrieve(query, top_k=1):
    q_emb = MODEL.encode(query)
    sims = [cosine_similarity([q_emb], [c["embedding"]])[0][0] for c in chunks]
    sorted_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    return [chunks[i] for i in sorted_idx[:top_k]]

# -------------------------------
#  GENERATE ANSWER USING GEMINI
# -------------------------------
def generate_answer(query, contexts):
    context_text = "\n\n".join([c["chunk"] for c in contexts])
    prompt = f"""
You are an AI assistant answering based on company documents.
Use ONLY the context given. Do not answer outside the provided content.

Context:
{context_text}

User Question: {query}
Answer clearly and professionally.
    """
    response = llm.generate_content(prompt)
    return response.text

# -------------------------------
#  STREAMLIT UI
# -------------------------------
st.set_page_config(
    page_title="Company Policy Assistant",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Container styling */
    .block-container {
        max-width: 900px;
        padding: 2rem 1rem;
    }
    
    /* Header card */
    .header-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 400;
    }
    
    /* Input section */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Answer card */
    .answer-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        margin: 1.5rem 0;
        border-left: 5px solid #10b981;
    }
    
    .answer-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .answer-text {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #374151;
    }
    
    /* Context card */
    .context-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        border: 1px solid #e2e8f0;
    }
    
    .context-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    
    .source-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .context-text {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #4b5563;
    }
    
    /* Stats section */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        flex: 1;
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-card">
    <div class="header-title">📚 Company Policy Assistant</div>
    <div class="header-subtitle">Get instant answers from your company documents using AI</div>
</div>
""", unsafe_allow_html=True)

# Stats section
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(documents)}</div>
        <div class="stat-label">Documents Loaded</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{len(chunks)}</div>
        <div class="stat-label">Knowledge Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">AI</div>
        <div class="stat-label">Powered by Gemini</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Query input
user_query = st.text_input(
    "Ask your question",
    placeholder="e.g., What is the vacation policy?",
    label_visibility="collapsed"
)

# Process query
if user_query:
    with st.spinner("🔍 Searching knowledge base..."):
        retrieved = retrieve(user_query, top_k=1)
        answer = generate_answer(user_query, retrieved)
    
    # Display answer
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-header">
            🤖 AI Response
        </div>
        <div class="answer-text">{answer}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display context
    st.markdown('<div class="context-header">📎 Source Information</div>', unsafe_allow_html=True)
    
    for ctx in retrieved:
        st.markdown(f"""
        <div class="context-card">
            <div class="source-badge">📄 {ctx['doc']}</div>
            <div class="context-text">{ctx['chunk']}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: white; font-size: 0.9rem; opacity: 0.8;'>
    Powered by Sentence Transformers & Google Gemini
</div>
""", unsafe_allow_html=True)