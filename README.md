# مساعد لائحة كلية الذكاء الاصطناعي

نظام RAG محلي للإجابة على أسئلة لائحة كلية الذكاء الاصطناعي (جامعة كفر الشيخ) بدقة وسرعة، مع عرض مصادر الإجابة بشكل منظم.

## الفكرة باختصار
- استرجاع المقاطع الأقرب من بيانات اللائحة (FAISS + TF-IDF + Keyword Boost).
- توليد إجابة عربية عبر Ollama (محلي) عند توفره.
- عرض واجهة ويب Flask محسنة مع:
	- ملخص سريع للإجابة
	- تنسيق واضح للقوائم والفقرات
	- مصادر مسترجعة منسقة

## المتطلبات
- Python 3.13+
- Ollama مثبت ومحلي (اختياري لكنه يرفع جودة الصياغة)
- نموذج Ollama:
	- `qwen2.5:1.5b-instruct`

## التشغيل (Windows)
### 1) تفعيل البيئة الافتراضية
```powershell
& .\.venv\Scripts\Activate.ps1
```

### 2) تثبيت المتطلبات
```powershell
d:/Downloads/files/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 3) بناء الفهرس من بيانات اللائحة
```powershell
d:/Downloads/files/.venv/Scripts/python.exe build_clean_index.py
```

### 4) تشغيل Ollama
```powershell
ollama serve
```

### 5) تشغيل الواجهة
```powershell
d:/Downloads/files/.venv/Scripts/python.exe flask_app.py
```

افتح:
- http://127.0.0.1:5000

## تشغيل سريع بأمر واحد
يمكنك استخدام:
```powershell
.\run.ps1
```

## هيكل المشروع
```
files/
├── flask_app.py          # واجهة Flask + منطق العرض
├── rag_system.py         # محرك RAG (index/search/generation)
├── build_clean_index.py  # بناء فهرس البحث من data.json
├── data.json             # بيانات اللائحة
├── requirements.txt      # الاعتماديات
├── run.ps1               # سكربت تشغيل سريع
├── templates/
│   └── index.html        # قالب الصفحة
├── static/
│   ├── style.css         # التصميم
│   ├── app.js            # تفاعلات الواجهة
│   └── favicon.svg       # أيقونة
└── index/
		├── faiss.index
		├── vectorizer.pkl
		└── chunks.json
```

## الميزات الحالية
- بحث دلالي محلي سريع على بيانات اللائحة.
- فلترة ذكية لأسئلة المواد (المستوى/الفصل).
- إجابة عربية عبر Ollama عند توفره.
- Fallback تلقائي لو Ollama غير متاح.
- Cache للأسئلة المتكررة لتسريع الرد.
- زر "مسح السابق" لمسح النتائج المخزنة مؤقتًا.

## الأداء
تم تطبيق تحسينات لتقليل زمن الرد:
- تقليل السياق المرسل للموديل.
- تقليل عدد tokens للتوليد.
- Cache لحالة Ollama.
- Cache لإجابات الأسئلة المتكررة.

## استكشاف الأخطاء
### الواجهة لا تفتح
1. تأكد أن Flask يعمل على 5000.
2. أعد تشغيل:
```powershell
d:/Downloads/files/.venv/Scripts/python.exe flask_app.py
```

### الرد بطيء
1. تأكد أن Ollama يعمل.
2. جرب إعادة نفس السؤال (سيستفيد من الكاش).

### النتائج غير محدثة بعد تعديل data.json
أعد بناء الفهرس:
```powershell
d:/Downloads/files/.venv/Scripts/python.exe build_clean_index.py
```

## ملاحظات مهمة
- المشروع Local-first بالكامل.
- يمكن التشغيل بدون Ollama (Retrieval فقط).
- يفضل كتابة السؤال بصيغة مباشرة مع كلمات مفتاحية.
