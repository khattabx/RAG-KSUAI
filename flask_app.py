import os
import html
import time
from flask import Flask, render_template, request, redirect, url_for
from rag_system import LaihaRAG, check_ollama, generate_answer, OLLAMA_MODEL

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

rag = LaihaRAG("./index")
rag.ensure_index("data.json")

# Simple in-memory caches to reduce repeated latency on local usage.
OLLAMA_STATUS_TTL = 8
ANSWER_CACHE_TTL = 300
_ollama_cache = {"value": None, "ts": 0.0}
_answer_cache = {}

SUGGESTED_QUESTIONS = [
    (
        "التخرج والنجاح",
        [
            "كم ساعة للتخرج؟",
            "ما درجة النجاح في المقرر؟",
            "ما شروط مرتبة الشرف؟",
            "متى يفصل الطالب؟",
        ],
    ),
    (
        "التسجيل والدراسة",
        [
            "ما الحد الأقصى لساعات التسجيل؟",
            "ما قواعد الحذف والاضافة؟",
            "ما نسبة الحضور المطلوبة؟",
            "ما شروط الانسحاب من مقرر؟",
        ],
    ),
    (
        "الخطة الدراسية",
        [
            "مواد المستوى الأول الفصل الأول",
            "مواد المستوى الثاني الفصل الثاني",
            "مواد المستوى الرابع الفصل الأول",
            "ما شروط مشروع التخرج؟",
        ],
    ),
]


def format_retrieved_answer(chunks):
    lines = []
    for chunk in chunks[:3]:
        text = chunk.get("text") or chunk.get("text_ar") or chunk.get("description_en", "")
        if text.strip():
            lines.append(f"- {text}")
    return "\n\n".join(lines) if lines else "لا توجد نتيجة مناسبة."


def answer_to_html(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""

    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = []
    prev_key = None
    for ln in raw_lines:
        key = reformat_compare_key(ln)
        if key == prev_key:
            continue
        lines.append(ln)
        prev_key = key

    if not lines:
        return ""

    has_list = any(ln.startswith("-") or re_match_numbered_item(ln) for ln in lines)
    heading = None
    if has_list and lines and lines[0].endswith(":") and not lines[0].startswith("-"):
        heading = lines[0]
        lines = lines[1:]

    if has_list:
        html_parts = []
        if heading:
            html_parts.append(f"<h4 class=\"answer-subtitle\">{html.escape(heading.rstrip(':'))}</h4>")
        opened = False
        for ln in lines:
            if ln.startswith("-") or re_match_numbered_item(ln):
                if not opened:
                    html_parts.append("<ul>")
                    opened = True
                cleaned = ln.lstrip("- ")
                cleaned = remove_numbering_prefix(cleaned)
                html_parts.append(f"<li>{html.escape(cleaned)}</li>")
            else:
                if opened:
                    html_parts.append("</ul>")
                    opened = False
                html_parts.append(f"<p>{html.escape(ln)}</p>")
        if opened:
            html_parts.append("</ul>")
        return "\n".join(html_parts)

    paragraphs = [f"<p>{html.escape(ln)}</p>" for ln in lines]
    return "\n".join(paragraphs)


def reformat_compare_key(line: str) -> str:
    return " ".join(line.replace(":", "").replace("-", " ").split()).strip().lower()


def re_match_numbered_item(line: str) -> bool:
    if not line:
        return False
    if line[0].isdigit() and len(line) > 1 and line[1] in ".)":
        return True
    return False


def remove_numbering_prefix(line: str) -> str:
    if len(line) > 1 and line[0].isdigit() and line[1] in ".)":
        return line[2:].strip()
    return line


def build_answer_preview(answer: str, limit: int = 150) -> str:
    if not answer:
        return ""
    cleaned = " ".join(answer.replace("\n", " ").split()).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def source_to_html(text: str, max_lines: int = 5) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:max_lines]
    if not lines:
        return ""

    has_list = any(ln.startswith("-") or re_match_numbered_item(ln) for ln in lines)
    if has_list:
        items = []
        for ln in lines:
            if ln.startswith("-") or re_match_numbered_item(ln):
                cleaned = remove_numbering_prefix(ln.lstrip("- "))
                items.append(f"<li>{html.escape(cleaned)}</li>")
            else:
                items.append(f"<li>{html.escape(ln)}</li>")
        return "<ul class=\"source-list\">" + "".join(items) + "</ul>"

    joined = " ".join(lines)
    return f"<p>{html.escape(joined)}</p>"


def prepare_sources_for_view(sources: list) -> list:
    rendered = []
    for src in sources:
        text = src.get("text") or src.get("text_ar") or src.get("description_en", "")
        score = float(src.get("score", 0) or 0)
        rendered.append({
            "title": src.get("title") or src.get("title_ar") or "محتوى",
            "score_text": f"{score * 100:.1f}%",
            "type": src.get("type"),
            "category": src.get("category"),
            "level": src.get("level"),
            "semester": src.get("semester"),
            "snippet_html": source_to_html(text),
        })
    return rendered


def get_ollama_status_cached() -> bool:
    now = time.time()
    if _ollama_cache["value"] is not None and (now - _ollama_cache["ts"]) < OLLAMA_STATUS_TTL:
        return _ollama_cache["value"]
    value = check_ollama()
    _ollama_cache["value"] = value
    _ollama_cache["ts"] = now
    return value


def get_answer_cache(query: str):
    now = time.time()
    row = _answer_cache.get(query)
    if not row:
        return None
    if (now - row["ts"]) > ANSWER_CACHE_TTL:
        _answer_cache.pop(query, None)
        return None
    return row["payload"]


def set_answer_cache(query: str, payload: dict):
    _answer_cache[query] = {
        "ts": time.time(),
        "payload": payload,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    answer = ""
    answer_html = ""
    answer_preview = ""
    sources = []
    rendered_sources = []
    mode = "retrieval"
    error = ""

    ollama_running = get_ollama_status_cached()

    if request.method == "POST":
        query = (request.form.get("query") or "").strip()
        if query:
            cached = get_answer_cache(query)
            if cached:
                answer = cached["answer"]
                answer_html = cached["answer_html"]
                answer_preview = cached["answer_preview"]
                sources = cached["sources"]
                rendered_sources = cached.get("rendered_sources", [])
                mode = cached["mode"]
            else:
                try:
                    sources = rag.search(query, top_k=5)
                    if sources and sources[0].get("type") == "courses":
                        answer = "\n".join([f"- {c}" for c in sources[0].get("courses", [])])
                        mode = "retrieval"
                    elif ollama_running:
                        try:
                            answer = generate_answer(query, sources)
                            mode = "ollama"
                        except Exception:
                            answer = format_retrieved_answer(sources)
                            mode = "retrieval"
                    else:
                        answer = format_retrieved_answer(sources)
                        mode = "retrieval"

                    answer_html = answer_to_html(answer)
                    answer_preview = build_answer_preview(answer)
                    rendered_sources = prepare_sources_for_view(sources)
                    set_answer_cache(query, {
                        "answer": answer,
                        "answer_html": answer_html,
                        "answer_preview": answer_preview,
                        "sources": sources,
                        "rendered_sources": rendered_sources,
                        "mode": mode,
                    })
                except Exception as ex:
                    error = f"حدث خطأ أثناء معالجة السؤال: {ex}"

    return render_template(
        "index.html",
        query=query,
        answer=answer,
        answer_html=answer_html,
        answer_preview=answer_preview,
        sources=sources,
        rendered_sources=rendered_sources,
        mode=mode,
        error=error,
        ollama_running=ollama_running,
        model_name=OLLAMA_MODEL,
        suggested_questions=SUGGESTED_QUESTIONS,
    )


@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.svg"), code=302)


@app.route("/clear-history", methods=["GET"])
def clear_history():
    _answer_cache.clear()
    return redirect(url_for("index"), code=302)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
