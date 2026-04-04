"""
RAG System للائحة كلية الذكاء الاصطناعي - جامعة كفر الشيخ
Hybrid Search: TF-IDF + Keyword matching
"""

import json, re, pickle, os, unicodedata
import numpy as np
from pathlib import Path

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b-instruct"
GEN_MAX_CONTEXT_CHUNKS = 2
GEN_MAX_CHARS_PER_CHUNK = 260
GEN_NUM_PREDICT = 160
GEN_TIMEOUT_SECONDS = 45


def load_json_data(path: str = "data.json") -> list:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


# ─── TEXT PROCESSING ────────────────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    result = [unicodedata.normalize('NFKC', c) for c in text]
    text = ''.join(result)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text


def extract_and_chunk(pdf_path: str, chunk_size: int = 600, overlap: int = 150) -> list:
    import fitz
    doc = fitz.open(pdf_path)
    chunks, current_chunk, current_page = [], "", "1"
    for i, page in enumerate(doc):
        text = normalize_arabic(page.get_text())
        if not text.strip():
            continue
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if len(current_chunk) + len(line) > chunk_size and current_chunk.strip():
                chunks.append({"id": len(chunks), "text": current_chunk.strip(), "page": current_page})
                current_chunk = current_chunk[-overlap:] + " " + line
            else:
                current_chunk += " " + line
        current_page = str(i + 2)
    if current_chunk.strip():
        chunks.append({"id": len(chunks), "text": current_chunk.strip(), "page": current_page})
    print(f"Extracted {len(chunks)} chunks from {len(doc)} pages")
    return chunks


# ─── INDEXING ────────────────────────────────────────────────────────────────

def build_index(chunks: list, index_dir: str = "./index") -> tuple:
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer
    os.makedirs(index_dir, exist_ok=True)
    texts = [prepare_text(c) for c in chunks]
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                                  max_features=10000, sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, f"{index_dir}/faiss.index")
    with open(f"{index_dir}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(f"{index_dir}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Index built: {dim} dims, {index.ntotal} vectors")
    return index, vectorizer, chunks


def load_index(index_dir: str = "./index") -> tuple:
    import faiss
    index = faiss.read_index(f"{index_dir}/faiss.index")
    with open(f"{index_dir}/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(f"{index_dir}/chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded index: {index.ntotal} vectors")
    return index, vectorizer, chunks


# ─── HYBRID RETRIEVAL ────────────────────────────────────────────────────────

def prepare_text(entry: dict) -> str:
    """Build searchable text from a data.json entry."""
    parts = []
    for field in ['summary', 'title', 'title_ar']:
        if entry.get(field): parts.append(str(entry[field]))
    if entry.get('keywords'): parts.append(' '.join(entry['keywords']))
    if entry.get('text_ar'): parts.append(entry['text_ar'])
    if entry.get('description_en'): parts.append(entry['description_en'])
    if entry.get('courses'): parts.append(' '.join(entry['courses']))
    if entry.get('level'): parts.append(f'مستوى {entry["level"]} level {entry["level"]}')
    if entry.get('semester'): parts.append(f'فصل {entry["semester"]} semester {entry["semester"]}')
    # fallback for old plain-text chunks
    if entry.get('text'): parts.append(entry['text'])
    return ' '.join(parts)


def keyword_score(query: str, text: str) -> float:
    """Score based on exact keyword matches — boosts precise answers."""
    words = [w for w in re.findall(r'[\u0600-\u06FF]+|\d+', query) if len(w) > 1]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in text)
    return hits / len(words)


def extract_level_semester(query: str):
    level_map = {
        "المستوى الأول": 1,
        "المستوى الاول": 1,
        "المستوى الثاني": 2,
        "المستوى التاني": 2,
        "المستوى الثالث": 3,
        "المستوى الرابع": 4,
    }
    semester_map = {
        "الفصل الأول": 1,
        "الفصل الاول": 1,
        "الفصل الثاني": 2,
        "الفصل التاني": 2,
    }

    level = None
    semester = None

    for phrase, value in level_map.items():
        if phrase in query:
            level = value
            break

    for phrase, value in semester_map.items():
        if phrase in query:
            semester = value
            break

    return level, semester


def smart_filter(results: list, query: str) -> list:
    q = query.strip()
    has_courses_intent = any(k in q for k in ["مواد", "مقررات", "الخطة", "المستوى", "الفصل"])

    if has_courses_intent:
        level, semester = extract_level_semester(q)
        courses = [r for r in results if r.get("type") == "courses"]
        if level is not None:
            courses = [r for r in courses if r.get("level") == level]
        if semester is not None:
            courses = [r for r in courses if r.get("semester") == semester]
        return courses or results

    if any(k in q for k in ["مرتبة الشرف", "شرف"]):
        honor = [r for r in results if "مرتبة الشرف" in (r.get("title") or "")]
        return honor or results

    if any(k in q for k in ["التخرج", "ساعة", "ساعات", "144"]):
        grad = [r for r in results if r.get("category") == "graduation"]
        return grad or results

    if any(k in q for k in ["النجاح", "راسب", "امتحان", "درجة"]):
        exams = [r for r in results if r.get("category") in ("exams", "grading")]
        return exams or results

    if any(k in q for k in ["يفصل", "فصل", "إنذار", "انذار"]):
        dismissal = [r for r in results if r.get("category") == "dismissal"]
        return dismissal or results

    return results


def retrieve(query: str, index, vectorizer, chunks: list, top_k: int = 5) -> list:
    """Hybrid: TF-IDF cosine + keyword overlap, re-ranked."""
    # TF-IDF scores
    q_vec = vectorizer.transform([query]).toarray().astype(np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm
    scores, indices = index.search(q_vec, min(top_k * 3, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx].copy()
        tfidf_s  = float(score)
        search_text = prepare_text(chunk)
        kw_s     = keyword_score(query, search_text)
        # Combined score: 60% TF-IDF + 40% keyword
        chunk["score"] = round(tfidf_s * 0.6 + kw_s * 0.4, 4)
        results.append(chunk)

    results = smart_filter(results, query)

    # Re-rank and return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ─── GENERATION (Ollama) ─────────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: list, api_key=None) -> str:
    import urllib.request

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks[:GEN_MAX_CONTEXT_CHUNKS]):
        page = chunk.get("page", "-")
        if chunk.get("type") == "courses" and chunk.get("courses"):
            text = "\n".join([f"- {c}" for c in chunk.get("courses", [])])
        else:
            text = chunk.get("text") or chunk.get("text_ar") or chunk.get("description_en", "")
        context_parts.append(f"[مقطع {i+1} - {page}]:\n{text[:GEN_MAX_CHARS_PER_CHUNK]}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""<|im_start|>system
أنت مساعد أكاديمي متخصص في لائحة كلية الذكاء الاصطناعي بجامعة كفر الشيخ.
قواعد صارمة:
1. أجب باللغة العربية فقط — ممنوع الإنجليزية
2. استخدم فقط المعلومات الموجودة في السياق
3. إذا كانت الإجابة رقماً أو شرطاً محدداً، اذكره مباشرة
4. اذكر رقم المادة إن وجد
5. أجب في 3 جمل أو أقل
<|im_end|>
<|im_start|>user
السياق:
{context}

السؤال: {query}
<|im_end|>
<|im_start|>assistant
"""

    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.05, "num_predict": GEN_NUM_PREDICT}
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())["response"].strip()


def check_ollama() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        return True
    except:
        return False


# ─── RAG CLASS ───────────────────────────────────────────────────────────────

class LaihaRAG:
    def __init__(self, index_dir: str = "./index"):
        self.index_dir = index_dir
        self.index = self.vectorizer = self.chunks = None

    def ensure_index(self, json_path: str = "data.json"):
        index_files = [
            Path(self.index_dir) / "faiss.index",
            Path(self.index_dir) / "vectorizer.pkl",
            Path(self.index_dir) / "chunks.json",
        ]

        if not all(p.exists() for p in index_files):
            self.build_from_json(json_path)
            return

        data_file = Path(json_path)
        if data_file.exists():
            data_mtime = data_file.stat().st_mtime
            index_mtime = min(p.stat().st_mtime for p in index_files)
            if data_mtime > index_mtime:
                self.build_from_json(json_path)
                return

        self.load()

    def build_from_json(self, json_path: str = "data.json"):
        print(f"Loading JSON: {json_path}")
        chunks = load_json_data(json_path)
        self.index, self.vectorizer, self.chunks = build_index(chunks, self.index_dir)
        print("RAG system ready from JSON!\n")

    def build(self, pdf_path: str):
        print(f"Processing: {pdf_path}")
        chunks = extract_and_chunk(pdf_path)
        self.index, self.vectorizer, self.chunks = build_index(chunks, self.index_dir)
        print("RAG system ready!\n")

    def load(self):
        self.index, self.vectorizer, self.chunks = load_index(self.index_dir)

    def search(self, query: str, top_k: int = 5) -> list:
        if self.index is None:
            raise RuntimeError("Call build() or load() first.")
        return retrieve(query, self.index, self.vectorizer, self.chunks, top_k)

    def ask(self, query: str, top_k: int = 5, **kwargs) -> dict:
        retrieved = self.search(query, top_k)
        answer = generate_answer(query, retrieved)
        return {
            "query": query,
            "answer": answer,
            "sources": [{
                "page": c.get("page", "-"),
                "score": c.get("score", 0.0),
                "text": (c.get("text") or c.get("text_ar") or c.get("description_en", "")),
                "preview": (c.get("text") or c.get("text_ar") or c.get("description_en", ""))[:120] + "..."
            } for c in retrieved]
        }

    def ask_no_llm(self, query: str, top_k: int = 3) -> str:
        retrieved = self.search(query, top_k)
        out = f"نتائج: '{query}'\n{'='*50}\n\n"
        for c in retrieved:
            page = c.get("page", "-")
            text = c.get("text") or c.get("text_ar") or c.get("description_en", "")
            out += f"صفحة {page} | تطابق: {c['score']:.3f}\n{text}\n\n{'─'*40}\n\n"
        return out


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INDEX_DIR = "./index"
    rag = LaihaRAG(INDEX_DIR)
    rag.ensure_index("data.json")

    ollama_ok = check_ollama()
    print(f"\n{'='*60}\nمساعد لائحة كلية الذكاء الاصطناعي")
    print(f"الموديل: {OLLAMA_MODEL if ollama_ok else 'بحث فقط'}")
    print(f"{'='*60}\n")

    while True:
        q = input("سؤالك: ").strip()
        if q in ("خروج","exit","quit","q"):
            break
        if not q:
            continue
        if ollama_ok:
            r = rag.ask(q)
            print(f"\nالإجابة:\n{r['answer']}\n")
        else:
            print(rag.ask_no_llm(q))
        print()
