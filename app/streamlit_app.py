import streamlit as st
import sys, os, json
import pandas as pd
from PIL import Image
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.retriever import retrieve
from generation.qa_chain import answer
from retrieval.vector_store import build_index

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal RAG | DSAI 413",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg: #0e1117; --surface: #1a1c24; --surface2: #262730;
    --border: #3b3d45; --accent: #f97316; --text: #e2e8f0;
    --text-dim: #94a3b8; --success: #22c55e;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: var(--bg); }
.block-container { padding-top: 2rem !important; max-width: 1200px !important; }
h1, h2, h3 { color: #ffffff !important; font-weight: 600 !important; }
.subtitle { color: var(--text-dim); font-size: 1.1rem; margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }
[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; background-color: var(--surface2) !important; border-radius: 0 8px 8px 0 !important; display: flex !important; }
.stButton>button { width: 100%; border-radius: 8px; font-weight: 600; transition: all 0.2s; }
.stButton>button[kind="primary"] { background-color: var(--accent); color: white; border: none; }
.stButton>button[kind="primary"]:hover { opacity: 0.9; transform: translateY(-1px); }
[data-testid="stChatInput"] { background-color: var(--surface) !important; border-radius: 12px !important; border: 1px solid var(--border) !important; }
.answer-box { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; line-height: 1.7; font-size: 1rem; color: var(--text); box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
.evidence-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin-right: 0.5rem; }
.badge-image { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
.badge-table { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
.badge-text { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Helper Functions ──────────────────────────────────────────────────────────
# this is use to read the pdf and the metadata to count how many papers and pages are indexed
def get_kb_stats():
    stats = {"papers": 0, "pages": 0}
    try:
        if os.path.exists("data/pdfs"):
            stats["papers"] = len(list(Path("data/pdfs").glob("*.pdf")))
        if os.path.exists("data/page_metadata.json"):
            with open("data/page_metadata.json", encoding="utf-8", errors="replace") as f:
                stats["pages"] = len(json.load(f))
    except Exception:
        pass
    return stats


def _get_modality_badge(s):
    ct = s.get("content_type", "page")
    mod = s.get("modality", "image")
    if ct == "figure" or (mod == "image" and ct == "page"):
        return "<span class='evidence-badge badge-image'>🖼️ IMAGE</span>"
    if ct == "table":
        return "<span class='evidence-badge badge-table'>📊 TABLE</span>"
    return "<span class='evidence-badge badge-text'>📝 TEXT</span>"


def _load_text_metadata() -> dict:
    """
    FIX: text_metadata.json is a LIST of dicts, not a dict.
    Structure: [{"paper_id": "...", "page_number": 1, "text": "...", ...}, ...]
    Convert it to a lookup dict keyed by (paper_id, page_number) for fast access.
    Also handles the Windows cp1252 encoding issue by forcing utf-8.
    """
    path = "data/text_metadata.json"
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            # Build lookup: (paper_id, page_number) -> text
            return {
                (entry.get("paper_id", ""), int(entry.get("page_number", 0))): entry.get("text", "")
                for entry in raw
                if isinstance(entry, dict)
            }
        elif isinstance(raw, dict):
            return raw  # already a dict, use as-is
    except Exception:
        pass
    return {}


def _find_text_for_chunk(s: dict, text_lookup: dict) -> str:
    """
    Look up page text for a retrieved chunk.
    text_lookup is keyed by (paper_id, page_number) tuples (from _load_text_metadata).
    Falls back to inline text/caption on the metadata dict itself.
    """
    # 1. Inline text already on the metadata
    text = (s.get("text") or s.get("caption") or "").strip()
    if text:
        return text

    if not text_lookup:
        return ""

    pid  = s.get("paper_id", "")
    page = s.get("page_number", "")

    # 2. Tuple key lookup (primary path)
    try:
        key = (pid, int(page))
        if key in text_lookup:
            return text_lookup[key]
    except (ValueError, TypeError):
        pass

    # 3. Fallback: any entry whose paper_id matches (returns first page found)
    for (k_pid, k_page), v in text_lookup.items():
        if k_pid == pid:
            return v

    return ""


def render_chunks(snippets):
    if not snippets:
        return
    st.markdown('<h4 style="margin-top:1rem; color:#94a3b8; font-size:1rem;">📚 Contextual Evidence</h4>', unsafe_allow_html=True)

    # Load text metadata — converts list-of-dicts to (paper_id, page_number)->text
    text_lookup = _load_text_metadata()

    for i, s in enumerate(snippets):
        pid   = s.get("paper_id", "?")
        page  = s.get("page_number", "?")
        score = s.get("_score", 0.0)
        badge = _get_modality_badge(s)
        img   = s.get("image_path", "")

        text = _find_text_for_chunk(s, text_lookup)

        expander_title = f"Source {i+1} | 📄 {pid} (Page {page}) | 🎯 Score: {score:.3f}"

        with st.expander(expander_title, expanded=(i == 0)):
            st.markdown(f"{badge}", unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                if img and os.path.exists(img):
                    st.image(Image.open(img), use_container_width=True)
            with c2:
                if text:
                    st.markdown("**Extracted Content:**")
                    st.markdown(
                        f"<div style='background:#1a1c24; padding:1rem; border-radius:6px; "
                        f"border:1px solid #3b3d45; font-size:0.85rem; max-height:250px; "
                        f"overflow-y:auto;'>{text}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Show available keys to help debug
                    sample_keys = list(text_data.keys())[:5] if text_data else []
                    st.warning(
                        f"Text not found for paper `{pid}` page `{page}`. "
                        f"Sample keys in text_metadata.json: `{sample_keys}`. "
                        f"Update `_find_text_for_chunk()` to match your key format."
                    )


def do_query(q):
    results = retrieve(q, top_k=st.session_state.get('top_k', 3))
    r = answer(q, results)
    snippets = []
    for score, meta in results:
        meta["_score"] = score
        snippets.append(meta)
    return r.get("answer", "Error generating answer."), snippets


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/182px-Python-logo-notext.svg.png", width=50)
    st.markdown("## Multi-Modal RAG")
    st.markdown("<div style='color:#94a3b8; font-size:0.85rem; margin-bottom:2rem;'>DSAI 413 - Assignment 1</div>", unsafe_allow_html=True)

    stats = get_kb_stats()
    st.markdown("### 📊 Knowledge Base")
    col1, col2 = st.columns(2)
    col1.metric("Documents", stats["papers"])
    col2.metric("Total Pages", stats["pages"])
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    st.slider("Top K Results", 1, 10, 3, key="top_k")

    if st.button("📂 Re-Index Documents", type="primary"):
        with st.spinner("Embedding and indexing multi-modal vectors..."):
            try:
                build_index()
                st.success("Indexing complete!")
            except Exception as e:
                st.error(f"Indexing failed: {str(e)}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by ColPali v1.2 & Groq LLM")

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("<h1>🔬 Document Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Query text, tables, and complex figures across your knowledge base.</div>", unsafe_allow_html=True)

tab_chat, tab_eval = st.tabs(["💬 Interactive QA", "📊 Evaluation Suite"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.history:
        st.markdown("<br><br>", unsafe_allow_html=True)
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.info("👋 **Welcome!** Try asking a question to get started.")

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f'<div class="answer-box">{turn["answer"]}</div>', unsafe_allow_html=True)
            render_chunks(turn.get("snippets", []))

    q = st.chat_input("Ask about charts, tables, or text...")
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Retrieving context..."):
                ans, snips = do_query(q)
            st.markdown(f'<div class="answer-box">{ans}</div>', unsafe_allow_html=True)
            render_chunks(snips)
        st.session_state.history.append({"query": q, "answer": ans, "snippets": snips})

# ── Tab 2: Evaluation ─────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("### 🧪 Automated Benchmark")
    col1, col2 = st.columns([1, 3])
    with col1:
        k_val = st.slider("Select K for Recall@K", 1, 10, 5, key="eval_k")
        run_btn = st.button("▶ Run Evaluation", type="primary")

    with col2:
        if run_btn:
            from evaluation.benchmark import BENCHMARK, _is_hit, _load_category_map

            # FIX: Load the real category map so _is_hit can match paper categories
            category_map = _load_category_map()

            hits, total, rows = 0, 0, []
            prog_bar = st.progress(0)

            for i, (q2, expected_keywords, mod) in enumerate(BENCHMARK):
                try:
                    results = retrieve(q2, top_k=k_val)
                    pids    = [m.get("paper_id", "") for _, m in results]

                    # FIX: Pass the real category_map instead of {}
                    hit = _is_hit(pids, expected_keywords, category_map)

                    hits  += int(hit)
                    total += 1

                    rows.append({
                        "Modality": mod,
                        "Query":    q2[:60] + ("..." if len(q2) > 60 else ""),
                        "Expected Keywords": str(expected_keywords),
                        "Retrieved IDs":     str(pids[:3]),
                        "Status":            "✅ HIT" if hit else "❌ MISS",
                    })
                except Exception as e:
                    st.error(f"Error on query {i}: {e}")

                prog_bar.progress((i + 1) / len(BENCHMARK))

            recall = hits / total if total else 0

            st.markdown(f"""
            <div style="text-align:center; padding:2rem; border:1px solid var(--border);
                        border-radius:10px; background:var(--surface); margin-bottom:1rem;">
                <h4 style="margin:0; color:var(--text-dim);">Final Score: Recall@{k_val}</h4>
                <h1 style="font-size:4rem; margin:0.5rem 0;
                           color:{'#22c55e' if recall > 0.7 else '#f97316'};">
                    {recall:.0%}
                </h1>
                <p style="margin:0; color:var(--text-dim);">
                    {hits} successful hits out of {total} queries
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # ── Debug helper: show a sample of category_map entries ──────────
            if not category_map:
                st.warning(
                    "⚠️ `category_map` is empty — `data/pdfs/category_map.json` was not found "
                    "or is malformed. The evaluation is falling back to keyword matching against "
                    "paper IDs only. Make sure your downloader writes this file."
                )
            else:
                with st.expander("🗂️ Category map preview (first 10 entries)"):
                    sample = dict(list(category_map.items())[:10])
                    st.json(sample)