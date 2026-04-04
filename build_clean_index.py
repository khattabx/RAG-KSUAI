"""
Build Index from data.json
Run: py build_clean_index.py
"""
import json, pickle, os
import numpy as np


def prepare_text(entry):
    parts = []
    for field in ['summary', 'title', 'title_ar']:
        if entry.get(field): parts.append(str(entry[field]))
    if entry.get('keywords'): parts.append(' '.join(entry['keywords']))
    if entry.get('text_ar'): parts.append(entry['text_ar'])
    if entry.get('description_en'): parts.append(entry['description_en'])
    if entry.get('courses'): parts.append(' '.join(entry['courses']))
    if entry.get('level'): parts.append(f'مستوى {entry["level"]} level {entry["level"]}')
    if entry.get('semester'): parts.append(f'فصل {entry["semester"]} semester {entry["semester"]}')
    return ' '.join(parts)


def build_from_json(json_path='data.json', index_dir='./index'):
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f'Loading {json_path}...')
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    texts = [prepare_text(d) for d in data]

    print('Building TF-IDF index...')
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=10000, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, f'{index_dir}/faiss.index')
    with open(f'{index_dir}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'{index_dir}/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'Index built: {len(data)} entries, {matrix.shape[1]} dims')
    return index, vectorizer, data


if __name__ == '__main__':
    index, vectorizer, data = build_from_json()

    # Quick test
    from rag_system import retrieve
    print('\n=== Test ===')
    tests = [
        'كم ساعة معتمدة للتخرج',
        'شروط مرتبة الشرف',
        'مواد المستوى الأول الفصل الأول',
        'مواد المستوى الثاني الفصل الثاني',
        'درجة النجاح في المقرر',
        'متى يفصل الطالب',
        'الحد الأقصى لساعات التسجيل',
        'شروط التدريب الميداني',
    ]
    for q in tests:
        r = retrieve(q, index, vectorizer, data, top_k=1)
        if r:
            print(f'  [{r[0]["score"]:.3f}] {q}')
            print(f'         -> {r[0]["title"]}')
