# 🎓 RAG System — لائحة كلية الذكاء الاصطناعي

نظام RAG (Retrieval-Augmented Generation) للبحث والإجابة عن أسئلة لائحة كلية الذكاء الاصطناعي بجامعة كفر الشيخ.

---

## 📁 هيكل المشروع

```
rag_laiha/
├── rag_system.py       ← الكود الأساسي (extraction, indexing, retrieval, generation)
├── app.py              ← Streamlit web UI
├── requirements.txt    ← المكتبات المطلوبة
├── README.md
└── index/              ← يُبنى تلقائياً
    ├── faiss.index     ← FAISS vector index
    ├── vectorizer.pkl  ← TF-IDF vectorizer
    └── chunks.json     ← النصوص المقسّمة
```

---

## ⚙️ التثبيت

```bash
pip install -r requirements.txt
```

---

## 🚀 طريقة الاستخدام

### 1. بناء الـ Index (مرة واحدة)
```bash
python rag_system.py laiha.pdf
```

### 2. واجهة ويب (Streamlit)
```bash
streamlit run app.py
```
ثم افتح `http://localhost:8501`

### 3. CLI تفاعلي
```bash
# مع Claude API
ANTHROPIC_API_KEY=sk-ant-... python rag_system.py

# بحث فقط (بدون API)
python rag_system.py
```

### 4. كـ Module في كودك
```python
from rag_system import LaihaRAG

rag = LaihaRAG()
rag.build("laiha.pdf")   # أول مرة فقط
# rag.load()             # بعد كده أسرع

# مع API
result = rag.ask("كم ساعة للتخرج؟", api_key="sk-ant-...")
print(result["answer"])
print(result["sources"])  # المقاطع المسترجعة مع أرقام الصفحات

# بدون API (retrieval فقط)
print(rag.ask_no_llm("شروط مرتبة الشرف"))
```

---

## 🏗️ المعمارية

```
PDF
 └── PyMuPDF (fitz)
      └── NFKC Normalization  ← يحوّل Arabic Presentation Forms لـ Standard Arabic
           └── Chunking (600 chars + 150 overlap)
                └── TF-IDF Vectorizer
                │    ├── analyzer: char_wb (n-grams حرفية)
                │    ├── ngram_range: (2, 4)
                │    └── max_features: 10,000
                └── FAISS FlatIP Index (cosine similarity)
                         └── Query → Top-K Chunks → Claude Haiku → Answer
```

### ليه char n-grams للعربي؟
- العربية لغة اشتقاقية (كلمة → كلمات → الكلمة → كلمتك)
- الـ character n-grams بتشارك features بين الأشكال المختلفة للكلمة
- مفيش حاجة لـ Arabic tokenizer
- قادر يتعامل مع كلمات مكسورة في الـ PDF

### ليه TF-IDF وليس Sentence Transformers؟
| | TF-IDF | Sentence Transformers |
|---|---|---|
| السرعة | ⚡ فوري | 🐌 أبطأ |
| الحجم | ~50MB | ~500MB+ |
| GPU | ❌ مش محتاج | ✅ مستحسن |
| Arabic | ✅ كويس مع char n-grams | ✅ ممتاز |

للترقية لاحقاً: `intfloat/multilingual-e5-large`

---

## 🔧 المشاكل الشائعة

### النص العربي بيطلع رموز غريبة؟
الـ PDF بيستخدم **Arabic Presentation Forms** (U+FE70-FEFF).
الحل: `unicodedata.normalize('NFKC', text)` — موجود في الكود.

### الـ retrieval بيرجع نتائج غلط؟
- زوّد `top_k` (مثلاً 8 بدل 5)
- جرب أسئلة أكثر تحديداً
- الكلمات المكسورة في الـ PDF بتأثر على الدقة

---

## 📊 إحصائيات
- **الصفحات:** 43
- **الـ Chunks:** 215
- **Chunk size:** 600 حرف + 150 overlap
- **Index dimensions:** 10,000
- **Model:** claude-haiku-4-5 (generation)
