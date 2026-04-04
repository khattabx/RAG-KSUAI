"""
Streamlit Web UI for لائحة RAG System
Run: py -m streamlit run app.py
"""

import streamlit as st

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="مساعد لائحة الكلية",
    page_icon="🎓",
    layout="centered",
)

# ── RTL + styling ───────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { direction: rtl; font-family: Cairo; }
.answer-box {
    background: #1e2a3a;
    color: #e8f0fe;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.source-box {
    background: #1a2332;
    padding: 0.5rem;
    border-radius: 6px;
    margin: 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load RAG ───────────────────────────────────────────
@st.cache_resource
def load_rag():
    from rag_system import LaihaRAG
    rag = LaihaRAG("./index")
    rag.ensure_index("data.json")
    return rag

rag = load_rag()

# ── Ollama ─────────────────────────────────────────────
from rag_system import check_ollama, generate_answer
ollama_running = check_ollama()

# ── Header ─────────────────────────────────────────────
st.title("🎓 مساعد لائحة الكلية")
st.caption("كلية الذكاء الاصطناعي — جامعة كفر الشيخ")

# ── Input ──────────────────────────────────────────────
query = st.text_input("اسأل")

# ── History ────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Ask ────────────────────────────────────────────────
if st.button("🔍 اسأل") and query:

    retrieved = rag.search(query, top_k=5)

    def format_retrieved_answer(chunks):
        return "\n\n".join([
            "- " + (chunk.get("text") or chunk.get("text_ar") or chunk.get("description_en", ""))
            for chunk in chunks[:3]
            if (chunk.get("text") or chunk.get("text_ar") or chunk.get("description_en", "")).strip()
        ]) or "لا توجد نتيجة"

    if retrieved and retrieved[0].get("type") == "courses":
        answer = "\n".join([f"- {c}" for c in retrieved[0].get("courses", [])])
        mode = "retrieval"
    elif ollama_running:
        try:
            answer = generate_answer(query, retrieved)
            mode = "ollama"
        except Exception:
            answer = format_retrieved_answer(retrieved)
            mode = "retrieval"
    else:
        answer = format_retrieved_answer(retrieved)
        mode = "retrieval"

    st.session_state.history.append({
        "query": query,
        "answer": answer,
        "sources": retrieved,
        "mode": mode
    })

# ── Display ────────────────────────────────────────────
for item in reversed(st.session_state.history):

    st.markdown(f"### ❓ {item['query']}")

    if item["mode"] == "ollama":
        st.markdown("🤖 **AI Answer**")
    else:
        st.markdown("🔍 **Search Result**")

    st.markdown(f"<div class='answer-box'>{item['answer']}</div>", unsafe_allow_html=True)

    with st.expander("📚 المصادر"):
        for src in item["sources"]:
            text = src.get("text") or src.get("text_ar") or src.get("description_en", "")
            st.markdown(
                f"<div class='source-box'>{text[:200]}...</div>",
                unsafe_allow_html=True
            )