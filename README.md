# RAG_PROJECT — E5 va Qwen bilan lokal RAG tizimi

Ushbu loyiha Windows kompyuterida o‘zbekcha hujjatlar asosida savollarga javob beradi. Foydalanuvchi savol yozadi, dastur hujjatlardan mos parchani topadi va Qwen shu parchaga tayanib javob beradi. Terminalda asl matn va manba nomi ham ko‘rinadi.

Loyihada olingan natija:

```text
Savol: Kitobni necha kunda qaytarish kerak?

Manba: kutubxona.txt | 2-parcha
Topilgan parcha: Kitobni qaytarish muddati 14 kun.
Qwen javobi: 14 kunda qaytarish kerak.
```

**Muhit:** Windows 11 · VS Code · Git Bash · Python 3.13.12  
**Embedding modeli:** `intfloat/multilingual-e5-base`  
**Lokal LLM:** `qwen2.5:7b`  
**GitHub:** [rasulbekdeveloper907/RAG_PROJECT](https://github.com/rasulbekdeveloper907/RAG_PROJECT)

---

## 1. RAG qanday ishlaydi?

**RAG — Retrieval-Augmented Generation.** Ushbu loyihada u ikki jarayondan iborat.

### Indeks yaratish

1. Python `data/docs/` ichidagi `.txt` fayllarni o‘qiydi.
2. Har bir hujjat bo‘sh satr bilan ajratilgan abzaslarga bo‘linadi.
3. E5 har bir abzasni 768 ta sondan iborat vektorga aylantiradi.
4. NumPy vektorlar, asl matnlar va manba nomlarini `data/index/rag.npz` fayliga saqlaydi.

### Savolga javob berish

1. E5 foydalanuvchi savolini ham vektorga aylantiradi.
2. Dastur savol vektorini indeksdagi parcha vektorlari bilan solishtiradi.
3. Eng yaqin bitta parcha tanlanadi va terminalda ko‘rsatiladi.
4. Python savol va parchani Ollama’ning lokal API manziliga yuboradi.
5. Qwen shu ma’lumotga tayanib javob yozadi.

| Texnologiya | Loyihadagi vazifasi |
|---|---|
| Hugging Face | Tayyor E5 embedding modelini yuklash |
| Sentence Transformers | E5’ni Python orqali ishlatish |
| NumPy | Vektorlarni saqlash va o‘xshashlikni hisoblash |
| Ollama | Qwen modelini kompyuterda ishga tushirish |
| Qwen 2.5 7B | Hujjatdan topilgan matn asosida javob yozish |
| Requests | Python’dan Ollama API’ga HTTP so‘rov yuborish |

## 2. Loyiha tuzilmasi

```text
RAG_PROJECT/
├── app.py                         # Terminaldan savol berish
├── requirements.txt               # Python kutubxonalari
├── README.md
├── .gitignore
├── rag/
│   ├── __init__.py
│   ├── embeddings.py              # E5 modelini ochish va vektor olish
│   ├── ingest.py                  # Hujjatlardan indeks yaratish
│   ├── retrieve.py                # Savolga eng yaqin parchani topish
│   └── llm.py                     # Qwen bilan lokal API orqali ishlash
├── data/
│   ├── docs/
│   │   ├── kutubxona.txt
│   │   ├── dars.txt
│   │   └── loyiha_topshirigi.txt
│   └── index/
│       └── rag.npz                # Indekslash vaqtida yaratiladi
├── models/
│   └── e5-base/                   # Lokal E5 modeli
└── .venv/                         # Python virtual muhiti
```

`models/e5-base/` ichida embedding modeli saqlanadi. Qwen modelining fayllarini Ollama alohida boshqaradi.

## 3. Windows va Git Bash muhitini tayyorlash

Buyruqlar VS Code ichidagi **Git Bash** terminalida bajariladi. Agar loyiha allaqachon ishlayotgan bo‘lsa, kundalik foydalanish uchun **12-bo‘lim** yetarli.

### 3.1. Loyiha papkasiga kirish

```bash
mkdir -p ~/Desktop/RAG_PROJECT
cd ~/Desktop/RAG_PROJECT
```

Versiyalarni tekshirish:

```bash
git --version
py --version
```

Amaliyotda quyidagi versiyalar ishladi:

```text
git version 2.51.0.windows.1
Python 3.13.12
```

### 3.2. Virtual muhit yaratish

Agar `.venv` hali yaratilmagan bo‘lsa:

```bash
py -m venv .venv
```

Virtual muhitni faollashtirish:

```bash
source .venv/Scripts/activate
```

Tekshirish:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

Python manzili `RAG_PROJECT\.venv\Scripts\python.exe` bilan tugashi kerak.

### 3.3. Papkalarni yaratish

```bash
mkdir -p rag data/docs data/index models
touch rag/__init__.py
```

### 3.4. Kutubxonalarni o‘rnatish

Yangi loyihada `requirements.txt` yaratish:

```bash
cat > requirements.txt <<'EOF'
sentence-transformers>=5,<6
numpy>=1.26,<3
requests>=2.32,<3
EOF
```

O‘rnatish:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Ollama va Qwen modelini o‘rnatish

### 4.1. Ollama’ni tekshirish

```bash
ollama --version
```

Agar `bash: ollama: command not found` chiqsa:

```bash
winget install --id Ollama.Ollama -e
```

O‘rnatish tugagach, VS Code’ni to‘liq yopib qayta oching. Windows Start menyusidan **Ollama** dasturini ishga tushiring. Loyihamiz Ollama bilan Git Bash va Python orqali ishlaydi.

### 4.2. Qwen modelini yuklash

```bash
ollama list
ollama pull qwen2.5:7b
ollama list
```

Model yuklamasi taxminan **4,7 GB**. Yuklash tugagach, `ollama list` ichida `qwen2.5:7b` ko‘rinadi. Keyingi safar saqlangan modeldan foydalaniladi.

### 4.3. Qwen’ni mustaqil sinash

```bash
ollama run qwen2.5:7b "RAG nima? O‘zbek tilida bitta sodda gap bilan tushuntir."
```

Bu buyruqda Qwen faqat savolni oladi. Amaliyotdagi birinchi javobda model RAG qisqartmasini noto‘g‘ri izohladi. Shu sababli to‘liq dastur Qwen’ga savol bilan birga **hujjatdan topilgan asl parchani** ham yuboradi.

## 5. Hugging Face’dan E5 embedding modelini yuklash

Git Bash’da quyidagi blokni oxirgi `PY` qatori bilan birga bajaring:

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device="cpu",
)
model.save_pretrained("models/e5-base")

print("Model saqlandi: models/e5-base")
print("Vektor o‘lchami:", model.get_sentence_embedding_dimension())
PY
```

Kutiladigan natija:

```text
Model saqlandi: models/e5-base
Vektor o‘lchami: 768
```

Keyingi bosqichlarda Python E5’ni `models/e5-base/` papkasidan ochadi.

### E5 qidiruvini alohida tekshirish

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "models/e5-base",
    device="cpu",
    local_files_only=True,
)

matnlar = [
    "Dars seshanba kuni soat 18:30 da boshlanadi.",
    "Kutubxonada har bir o‘quvchi bir vaqtning o‘zida ikkita kitob olishi mumkin.",
    "HDMI kabeli administrator stolining ikkinchi tortmasida.",
]
savol = "Bir o‘quvchi nechta kitob olishi mumkin?"

matn_vektorlari = model.encode(
    ["passage: " + matn for matn in matnlar],
    normalize_embeddings=True,
)
savol_vektori = model.encode(
    "query: " + savol,
    normalize_embeddings=True,
)

ballar = matn_vektorlari @ savol_vektori
eng_yaqin = int(ballar.argmax())

print("Savol:", savol)
print("Topilgan matn:", matnlar[eng_yaqin])
PY
```

Amaliyotda E5 uchta jumla orasidan **“ikkita kitob”** yozilgan jumlani topdi. `query: ` prefiksi savol uchun, `passage: ` esa hujjat uchun ishlatiladi. O‘zbekcha matnda ham bu prefikslar shu ko‘rinishda qoladi.

## 6. Python modullarini yaratish

Quyidagi kodlarni sarlavhada ko‘rsatilgan fayllarga saqlang. `rag/__init__.py` bo‘sh qoladi.

### 6.1. `rag/embeddings.py`

Bu modul E5’ni lokal papkadan ochadi. `lru_cache` bitta Python jarayoni davomida yuklangan modelni qayta ishlatadi.

```python
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(
        str(ROOT / "models" / "e5-base"),
        device="cpu",
        local_files_only=True,
    )


def embed_passages(texts):
    passages = ["passage: " + text for text in texts]
    return get_model().encode(passages, normalize_embeddings=True)


def embed_question(question):
    return get_model().encode(
        "query: " + question,
        normalize_embeddings=True,
    )
```

### 6.2. `rag/ingest.py`

Bu modul `.txt` fayllarni abzaslarga ajratadi. Vektor bilan birga matn va `kutubxona.txt | 2-parcha` shaklidagi manba ham saqlanadi.

```python
import re

import numpy as np

from rag.embeddings import ROOT, embed_passages


DOCS = ROOT / "data" / "docs"
INDEX = ROOT / "data" / "index" / "rag.npz"


def main():
    texts = []
    sources = []

    for path in sorted(DOCS.rglob("*.txt")):
        content = path.read_text(encoding="utf-8-sig").strip()
        paragraphs = re.split(r"\n\s*\n", content)

        for number, paragraph in enumerate(paragraphs, start=1):
            if paragraph.strip():
                texts.append(paragraph.strip())
                sources.append(f"{path.relative_to(DOCS)} | {number}-parcha")

    if not texts:
        raise SystemExit("data/docs ichiga matn yozilgan .txt fayl qo‘shing.")

    vectors = embed_passages(texts)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDEX, vectors=vectors, texts=texts, sources=sources)
    print(f"Indeks tayyor. Parchalar soni: {len(texts)}")


if __name__ == "__main__":
    main()
```

### 6.3. `rag/retrieve.py`

Savol va hujjat vektorlari normallashtirilgani uchun ularning skalyar ko‘paytmasi kosinus o‘xshashligiga teng. `argmax()` eng yuqori ballga ega parchaning indeksini beradi.

```python
import numpy as np

from rag.embeddings import embed_question
from rag.ingest import INDEX


def search(question):
    if not INDEX.exists():
        raise SystemExit("Avval python -m rag.ingest buyrug‘ini bajaring.")

    query = embed_question(question)

    with np.load(INDEX, allow_pickle=False) as index:
        scores = index["vectors"] @ query
        best = int(scores.argmax())
        return str(index["sources"][best]), str(index["texts"][best])
```

### 6.4. `rag/llm.py`

Bu modul savol va topilgan parchani lokal Ollama API’ga yuboradi. `stream=False` javobni bitta JSON ko‘rinishida olish uchun ishlatiladi. Server xato qaytarsa, dastur HTTP kodi bilan birga xato matnini ham chiqaradi.

```python
import requests


def answer(question, passage):
    response = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "qwen2.5:7b",
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Savolga faqat berilgan parchaga asoslanib, "
                        "o‘zbek tilida qisqa javob ber. Sonlarni aniq saqla. "
                        "Javob parchada bo‘lmasa, «Ma’lumot topilmadi» deb yoz."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Parcha: {passage}\nSavol: {question}",
                },
            ],
        },
        timeout=(5, 300),
    )

    if not response.ok:
        raise SystemExit(f"Ollama HTTP {response.status_code}: {response.text}")

    return response.json()["message"]["content"]
```

### 6.5. `app.py`

Bu fayl loyiha ildizida, `rag/` papkasi yonida turadi. `--search-only` berilganda faqat qidiruv bajariladi. Oddiy rejimda topilgan parcha Qwen’ga ham yuboriladi.

```python
import argparse

from rag.llm import answer
from rag.retrieve import search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()

    source, passage = search(args.question)

    print("Manba:", source)
    print("Topilgan parcha:", passage)

    if not args.search_only:
        print("Qwen javobi:", answer(args.question, passage))


if __name__ == "__main__":
    main()
```

## 7. Birinchi hujjatlarni yaratish va indekslash

Quyidagi buyruqlar boshlang‘ich namuna fayllarini yaratadi. Fayllarga avval o‘z matningizni yozgan bo‘lsangiz, keyingi indekslash buyrug‘idan davom eting.

```bash
cat > data/docs/kutubxona.txt <<'EOF'
Kutubxonada har bir o‘quvchi bir vaqtning o‘zida ikkita kitob olishi mumkin.

Kitobni qaytarish muddati 14 kun.
EOF

cat > data/docs/dars.txt <<'EOF'
RAG amaliyot darsi seshanba kuni soat 18:30 da boshlanadi.
EOF
```

Indeks yaratish:

```bash
python -m rag.ingest
```

Faqat yuqoridagi ikki hujjat uchun kutiladigan natija:

```text
Indeks tayyor. Parchalar soni: 3
```

`kutubxona.txt` ichida ikkita abzas, `dars.txt` ichida bitta abzas bor. Indeks `data/index/rag.npz` manzilida yaratiladi.

## 8. Birinchi to‘liq RAG sinovi

### Faqat qidiruv

```bash
python app.py --search-only "Kitobni necha kunda qaytarish kerak?"
```

Natija:

```text
Manba: kutubxona.txt | 2-parcha
Topilgan parcha: Kitobni qaytarish muddati 14 kun.
```

### Qidiruv va Qwen javobi

```bash
python app.py "Kitobni necha kunda qaytarish kerak?"
```

Amaliyotdagi natija:

```text
Manba: kutubxona.txt | 2-parcha
Topilgan parcha: Kitobni qaytarish muddati 14 kun.
Qwen javobi: 14 kunda qaytarish kerak.
```

Boshqa hujjat bilan sinash:

```bash
python app.py "RAG darsi qaysi kuni va soat nechada boshlanadi?"
```

Kutiladigan manba — `dars.txt`. Javobda **seshanba kuni soat 18:30** ko‘rsatilishi kerak.

## 9. Yangi hujjat bilan sinov — Anor-27

Loyihaga yangi ma’lumot qo‘shamiz:

```bash
cat > data/docs/loyiha_topshirigi.txt <<'EOF'
RAG amaliyoti uchun yakuniy loyihaning kod nomi Anor-27.

Yakuniy loyihani topshirishda o‘quvchi uchta namuna savol va har bir javobning manbasini ko‘rsatadi.
EOF
```

Indeksni yangilaymiz va savol beramiz:

```bash
python -m rag.ingest
python app.py --search-only "Yakuniy loyihaning kod nomi nima?"
python app.py "Yakuniy loyihaning kod nomi nima?"
```

Amaliyotdagi natija:

```text
Manba: loyiha_topshirigi.txt | 1-parcha
Topilgan parcha: RAG amaliyoti uchun yakuniy loyihaning kod nomi Anor-27.
Qwen javobi: Yakuniy loyihaning kod nomi Anor-27.
```

Bu sinovda foydalanuvchi savol ichida `Anor-27` so‘zini yozmadi. E5 tegishli parchani topdi va Qwen kod nomini o‘sha parchadan oldi.

## 10. Hujjat yangilanganda nima bo‘ladi?

Avval fayldagi kod nomini o‘zgartiramiz:

```bash
sed -i 's/Anor-27/Anor-28/' data/docs/loyiha_topshirigi.txt
```

Indeksni yangilashdan oldin qidiramiz:

```bash
python app.py --search-only "Yakuniy loyihaning kod nomi nima?"
```

Bu vaqtda qidiruvda hali `Anor-27` chiqadi. Chunki `retrieve.py` matnni saqlangan `rag.npz` indeksidan o‘qiydi.

Endi indeksni yangilaymiz:

```bash
python -m rag.ingest
python app.py "Yakuniy loyihaning kod nomi nima?"
```

| Bosqich | TXT faylida | Qidiruvda |
|---|---|---|
| Dastlabki indeks | Anor-27 | Anor-27 |
| Fayl tahrirlandi | Anor-28 | Anor-27 |
| Indeks qayta yaratildi | Anor-28 | Anor-28 |

**Amaliy qoida:** `data/docs/` ichiga fayl qo‘shilganda, tahrirlanganda yoki olib tashlanganda `python -m rag.ingest` buyrug‘ini bir marta bajaring.

## 11. Uchragan xatolar va ularning yechimlari

### `bash: ollama: command not found`

Git Bash `ollama` dasturini topa olmadi. Ollama’ni o‘rnating, VS Code’ni qayta oching va tekshiring:

```bash
ollama --version
```

### `ollama list` faqat ustun nomlarini ko‘rsatdi

Ollama ishlayotgan, lekin lokal model hali yuklanmagan edi:

```bash
ollama pull qwen2.5:7b
ollama list
```

### `can't open file ... app.py`

Amaliyot boshida papkada faqat `.venv` mavjud edi. Avval loyiha joylashuvini tekshiring:

```bash
pwd
ls -la
```

6-bo‘limdagi `app.py` faylini loyiha ildizida, qolgan modullarni `rag/` ichida yarating.

### Manba topildi, lekin Ollama `500 Server Error` qaytardi

Terminal `Manba` va `Topilgan parcha` qatorlarini chiqargan bo‘lsa, qidiruv bajarilgan. Ollama’ni alohida tekshirish uchun Git Bash’da:

```bash
curl -sS -w '\nHTTP %{http_code}\n' \
  http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:7b","stream":false,"messages":[{"role":"user","content":"Salom"}]}'
```

Amaliyotda bu so‘rovga `HTTP 200` va `"done": true` qaytdi. Keyingi urinishda to‘liq RAG so‘rovi ham ishladi. Vaqti-vaqti bilan chiqqan `500` xatosining aniq sababi tasdiqlanmadi.

`rag/llm.py` ichidagi quyidagi kod xato takrorlanganda Ollama javobini ko‘rsatadi:

```python
if not response.ok:
    raise SystemExit(f"Ollama HTTP {response.status_code}: {response.text}")
```

### Har bir savolda `Loading weights` chiqdi

Har bir `python app.py ...` buyrug‘i yangi Python jarayonini boshlaydi. Shu sababli E5 vaznlari lokal diskdan xotiraga qayta o‘qiladi. Bu bosqichda dastur `models/e5-base/` papkasidan foydalanadi.

## 12. Kompyuterni qayta yoqqandan keyin

Git Bash’ni oching:

```bash
cd ~/Desktop/RAG_PROJECT
source .venv/Scripts/activate
ollama list
python app.py "Yakuniy loyihaning kod nomi nima?"
```

Agar Ollama bilan ulanish xatosi chiqsa, Windows Start menyusidan **Ollama** dasturini oching va savolni qayta yuboring.

| Vazifa | Buyruq |
|---|---|
| Loyihani VS Code’da ochish | `code .` |
| Virtual muhitni faollashtirish | `source .venv/Scripts/activate` |
| Yuklangan LLM’larni ko‘rish | `ollama list` |
| Faqat qidiruvni tekshirish | `python app.py --search-only "Savolingiz"` |
| Hujjat asosida javob olish | `python app.py "Savolingiz"` |
| Hujjatlar o‘zgargach indeksni yangilash | `python -m rag.ingest` |
| Virtual muhitdan chiqish | `deactivate` |

## 13. GitHub bilan ishlash

Repo manzili:

```text
https://github.com/rasulbekdeveloper907/RAG_PROJECT.git
```

Joriy ulanish va branchni tekshirish:

```bash
git remote -v
git status -sb
```

Agar yangi lokal papka hali Git reposi sifatida ochilmagan bo‘lsa:

```bash
git init
git branch -M main
```

Agar `origin` hali mavjud bo‘lmasa:

```bash
git remote add origin https://github.com/rasulbekdeveloper907/RAG_PROJECT.git
```

Agar `origin` mavjud, lekin manzili boshqa bo‘lsa:

```bash
git remote set-url origin https://github.com/rasulbekdeveloper907/RAG_PROJECT.git
```

### `.gitignore`

Lokal virtual muhit, modellar, yaratilgan indeks va shaxsiy hujjatlarni kompyuterda saqlash uchun:

```gitignore
.venv/
__pycache__/
models/
data/index/
data/docs/
.env
```

Ushbu sozlamada boshqa kompyuterga repo klonlanganda namuna hujjatlarini 7- va 9-bo‘limlardan qayta yaratish mumkin.

### Faqat ushbu README’ni push qilish

`README.md` faylini `app.py` yoniga saqlang:

```bash
cd ~/Desktop/RAG_PROJECT
git add README.md
git diff --cached --stat
git commit -m "Add detailed Uzbek README for local RAG project"
git push -u origin main
```

Agar Python modullari ham hali GitHub’ga yuborilmagan bo‘lsa, ularni alohida tanlash mumkin:

```bash
git add app.py rag/ requirements.txt .gitignore
git diff --cached --stat
git commit -m "Add local RAG pipeline with E5 and Qwen"
git push
```

## 14. Natijani qanday baholaymiz?

Har bir savolda uchta narsani solishtiring:

1. **Manba:** savolga javob bor fayl tanlandimi?
2. **Parcha:** kerakli son, nom yoki fakt aynan shu matnda bormi?
3. **Javob:** Qwen shu faktni to‘g‘ri yetkazdimi?

Masalan, muddat haqidagi savolda `kutubxona.txt | 2-parcha` tanlanishi, parchada **14 kun** yozilishi va javobda ham **14 kun** saqlanishi kerak. Kod nomi haqidagi sinovda esa indeks yangilangach, parcha va javobda **Anor-28** chiqishi kerak.

**Joriy imkoniyat chegarasi:** dastur qisqa UTF-8 abzaslar bilan ishlaydi va har bir savol uchun eng yaqin bitta parchani oladi. Hujjatlarda javob bo‘lmasa ham qidiruv eng yaqin parchani qaytaradi. Shu holatni tekshirish uchun hujjatlarda ko‘rsatilmagan ma’lumotni, masalan, kurs narxini so‘rang va Qwen javobini chiqarilgan parcha bilan solishtiring.