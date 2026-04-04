# 🎓 مساعد لائحة كلية الذكاء الاصطناعي
### RAG System — كلية الذكاء الاصطناعي، جامعة كفر الشيخ

---

## 📌 نظرة عامة

نظام **RAG (Retrieval-Augmented Generation)** يتيح للطلاب والأساتذة الاستفسار عن أي بند في لائحة كلية الذكاء الاصطناعي بجامعة كفر الشيخ باللغة العربية، مع إجابات دقيقة مبنية على نص اللائحة الرسمية.

**الفكرة الأساسية:**
```
سؤال المستخدم → بحث في اللائحة → أفضل المقاطع → Qwen2.5-Coder → إجابة عربية دقيقة
```

---

## 🏗️ معمارية المشروع

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│           (app.py — واجهة المستخدم)              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              LaihaRAG (rag_system.py)            │
│                                                  │
│  ┌─────────────┐    ┌──────────────────────────┐ │
│  │   Retrieval  │    │      Generation          │ │
│  │             │    │                          │ │
│  │ TF-IDF      │    │  Ollama Local API        │ │
│  │ char n-gram │ →  │  qwen2.5-coder:latest    │ │
│  │ +           │    │  localhost:11434         │ │
│  │ Keyword     │    │                          │ │
│  │ matching    │    │                          │ │
│  └──────┬──────┘    └──────────────────────────┘ │
│         │                                        │
│  ┌──────▼──────────────────────────────────────┐ │
│  │              FAISS Index                    │ │
│  │   (index/faiss.index + vectorizer.pkl)      │ │
│  │   28 sections — كل مادة chunk مستقل        │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 📁 هيكل الملفات

```
files/
├── rag_system.py          ← الكود الأساسي (Extraction, Indexing, Retrieval, Generation)
├── app.py                 ← Streamlit Web UI
├── build_clean_index.py   ← بناء الـ Index من النص النظيف للمواد
├── requirements.txt       ← المكتبات المطلوبة
├── laiha.pdf              ← ملف اللائحة الأصلي
├── README.md              ← هذا الملف
└── index/
    ├── faiss.index        ← FAISS vector index
    ├── vectorizer.pkl     ← TF-IDF vectorizer
    └── chunks.json        ← 28 section — نص كل مادة
```

---

## ⚙️ التقنيات المستخدمة

| التقنية | الاستخدام | السبب |
|---------|-----------|-------|
| **PyMuPDF (fitz)** | استخراج النص من PDF | أسرع وأدق مكتبة لـ PDF في Python |
| **NFKC Normalization** | تحويل Arabic Presentation Forms | الـ PDF يستخدم U+FE70-FEFF بدل U+0600 |
| **TF-IDF (char n-gram 2-4)** | تحويل النص لـ vectors | يتعامل مع التصريف العربي بدون tokenizer |
| **FAISS (IndexFlatIP)** | بحث بالتشابه (cosine similarity) | سريع جداً للـ vectors، يعمل offline |
| **Keyword Matching** | دعم الـ TF-IDF | يرفع دقة النتائج للأسئلة الواضحة |
| **Ollama** | تشغيل LLM محلياً | بدون نت، بدون API key، privacy كاملة |
| **Qwen2.5-Coder:latest** | توليد الإجابات | يفهم العربية ويدعم ChatML format |
| **Streamlit** | واجهة المستخدم | سهل، سريع، يدعم RTL |
| **scikit-learn** | TF-IDF Vectorizer | مكتبة ML موثوقة |
| **numpy** | العمليات الحسابية على الـ vectors | أساسية لـ FAISS |

---

## 🔧 آلية عمل الـ Hybrid Search

```python
# Combined Score = 60% TF-IDF + 40% Keyword
score = tfidf_score * 0.6 + keyword_score * 0.4
```

**ليه Hybrid؟**
- TF-IDF وحده كان بيرجع نتائج بـ score 0.2 ← ضعيف
- Keyword وحده مش كافي للأسئلة المعقدة
- الـ Hybrid رفع الـ scores لـ 0.6-0.7 ← دقيق

---

## 🚨 المشاكل اللي تم حلها

### 1. Arabic Presentation Forms
**المشكلة:** الـ PDF يستخدم characters من U+FE70-FEFF (visual encoding) بدل Standard Arabic U+0600.

**الحل:**
```python
unicodedata.normalize('NFKC', text)
```
بيحول `ﻣﺎﺩﺓ` → `مادة` تلقائياً.

---

### 2. كلمات مكسورة في PDF
**المشكلة:** الـ PDF كان بيطلع كلمات مقسومة مثل `زجتاا ذىلا` بدل `اجتاز الطالب`.

**الحل:** بدل الاعتماد على PDF extraction، حطينا النص النظيف لكل مادة مباشرةً في `build_clean_index.py`.

---

### 3. Chunk overlap مكسور
**المشكلة:** الـ overlapping chunking كانت بتقطع الجملة في النص، فالـ chunk بدأ بـ "جاح 144 ساعة" بدل "يجتاز 144 ساعة".

**الحل:** كل مادة من اللائحة = chunk مستقل كامل (28 section).

---

### 4. Python PATH على Windows
**المشكلة:** Windows عنده 3 Python installations متعارضة (Inkscape Python, Python 3.8, Python 3.13).

**الحل:** استخدام `py` launcher بدل `python` للوصول لـ Python 3.13.

---

### 5. Qwen بيجيب إجابات إنجليزية
**المشكلة:** الموديل Coder مش Chat، فكان بيرد بالإنجليزي.

**الحل:** استخدام **ChatML format** الرسمي:
```
<|im_start|>system ... <|im_end|>
<|im_start|>user ... <|im_end|>
<|im_start|>assistant
```
مع تعليمات صريحة "أجب بالعربية فقط".

---

## 🚀 طريقة التشغيل

### 1. التثبيت (مرة واحدة)
```powershell
py -m pip install pymupdf scikit-learn faiss-cpu numpy streamlit
```

### 2. بناء الـ Index (مرة واحدة)
```powershell
cd D:\Downloads\files
py build_clean_index.py
```

### 3. تشغيل Ollama
```powershell
# terminal منفصل
ollama serve
```

### 4. تشغيل الـ UI
```powershell
py -m streamlit run app.py
```
افتح: `http://localhost:8501`

---

## 💬 أمثلة على الأسئلة

| السؤال | الإجابة المتوقعة |
|--------|-----------------|
| كم ساعة معتمدة للتخرج؟ | 144 ساعة معتمدة (مادة 5) |
| ما درجة النجاح في المقرر؟ | 50% من إجمالي الدرجات (مادة 13) |
| ما شروط مرتبة الشرف؟ | CGPA ≥ 3، خلال 4 سنوات، بدون رسوب (مادة 17) |
| الحد الأقصى لساعات التسجيل؟ | 21 ساعة لمن CGPA ≥ 3 (مادة 8) |
| متى يُفصل الطالب؟ | بعد 4 فصول متتالية بإنذار أو 8 سنوات (مادة 22) |
| ما مقررات المستوى الأول؟ | BC111، MA112، MA113، HU111، ... (مادة 27) |

---

## 📊 أداء الـ Retrieval

| الإصدار | الطريقة | أعلى Score | دقة الإجابة |
|---------|---------|-----------|-------------|
| v1 | TF-IDF من PDF المكسور | 0.29 | ضعيفة |
| v2 | TF-IDF من نص نظيف | 0.19 | متوسطة |
| v3 | Hybrid (TF-IDF + Keyword) من نص نظيف | **0.69** | ✅ ممتازة |

---

## 🔮 تحسينات مستقبلية

- [ ] إضافة **Sentence Transformers** متعدد اللغات للـ embeddings
- [ ] دعم **streaming** في الإجابات (بدل الانتظار)
- [ ] إضافة **history** حقيقي للمحادثة
- [ ] دعم PDF محدّث تلقائياً عند تغيير اللائحة
- [ ] نشر على **Streamlit Cloud** للوصول من أي مكان

---

## 👨‍💻 معلومات المشروع

- **المطور:** Mohamed — طالب كلية الذكاء الاصطناعي، جامعة كفر الشيخ (2023-2027)
- **التاريخ:** أبريل 2026
- **اللغة:** Python 3.13
- **الـ LLM:** Qwen2.5-Coder:latest via Ollama (Local)
- **المصدر:** لائحة كلية الذكاء الاصطناعي، جامعة كفر الشيخ، 2023
