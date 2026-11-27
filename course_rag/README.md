# RAG_PROJECT — AI kursi yakuniy imtihoni uchun lokal RAG yordamchisi

Ushbu hujjatda `RAG_PROJECT` loyihasining ikki bosqichi tushuntirilgan. Birinchi bosqichda o‘zbekcha hujjatlardan bitta mos parchani topadigan **E5 + NumPy + Qwen** tizimi qurildi. Ikkinchi bosqichda shu loyiha asosida **TACTIQ arxitekturasiga mos, Qdrant bilan ishlaydigan RAG indexing pipeline** yaratildi.

Yakuniy loyihada foydalanuvchi terminalga o‘zbekcha savol yozadi. Dastur lokal hujjatlardan tegishli parchalarni topadi, qaysi fayllardan olinganini ko‘rsatadi va Qwen orqali javob beradi.

**Amaliyotda olingan yakuniy javob:**

```text
Savol: Yakuniy video qancha davom etadi va unga necha ball beriladi?

Qwen javobi: Yakuniy video 20 soniya davom etadi va unga 40 ball beriladi.
```

Bu savol loyiha uchun yaxshi sinov bo‘ldi. **20 soniya** haqidagi ma’lumot `yakuniy_topshiriq.md` faylida, **40 ball** haqidagi ma’lumot esa `baholash.md` faylida yozilgan. To‘liq javob uchun ikki manbadagi dalil kerak.

---

## 1. Loyiha mavzusi va maqsadi

**Mavzu:** Sun’iy intellekt kursining yakuniy imtihoni bo‘yicha savol-javob yordamchisi.

O‘quvchi imtihon haqida quyidagi savollarni berishi mumkin:

- Yakuniy imtihon mavzusi nima?
- Nechta ish topshiriladi?
- Rasm qanday o‘lchamda tayyorlanadi?
- Video necha soniya davom etadi?
- Videoga necha ball ajratilgan?
- Taqdimot nechta slayddan iborat?
- Ishlar qaysi formatlarda topshiriladi?
- Ishlatilgan promptlar qayerga yoziladi?

**Maqsad** — javobni kurs hujjatlariga bog‘lash. Masalan, boshqa kursda video 30 soniya bo‘lishi mumkin. Bizning yordamchimiz esa aynan indekslangan topshiriqda ko‘rsatilgan **20 soniya**ni topishi kerak.

Baholash ballari va topshirish tartibi ushbu amaliyot uchun namunaviy yozilgan. Haqiqiy kursda shu fayllarga tasdiqlangan talablar kiritiladi va indeks qayta yaratiladi.

## 2. Ishlatilgan texnologiyalar

| Texnologiya | Loyihada qayerda ishlatildi? | Vazifasi |
|---|---|---|
| Windows va Git Bash | Lokal ish muhiti | Papkalar yaratish, Python dasturlarini va Ollama buyruqlarini ishga tushirish |
| Python 3.13.12 | `.venv` ichida | Loyihaning asosiy dasturlash tili |
| Hugging Face | Modelni birinchi marta yuklashda | `intfloat/multilingual-e5-base` modelini olish |
| Sentence Transformers | `rag/embeddings.py` | E5 orqali matn va savol vektorlarini hisoblash |
| E5 tokenizer | `course_rag/index.py` | Uzun matnlarni token chegaralari bo‘yicha parchalash |
| NumPy | Dastlabki RAG versiyasida | Vektorlarni `.npz` fayliga saqlash va o‘xshashlikni hisoblash |
| Qdrant Client | Kurs RAG versiyasida | Vektorlar va manba ma’lumotlarini lokal diskda saqlash, o‘xshash parchalarni qidirish |
| Ollama | Windows’da lokal xizmat | Qwen modelini ishga tushirish va Python uchun HTTP API berish |
| Qwen 2.5 7B | `rag/llm.py` orqali | Topilgan parchalar asosida javob yozish |
| Requests | `rag/llm.py` | Python’dan Ollama API’ga savol va kontekst yuborish |
| Git va GitHub | Loyiha kodini boshqarish | Kod, README va o‘zgarishlar tarixini saqlash |

## 3. Ikki pipeline o‘rtasidagi farq

Loyihani tushunish uchun eng muhim savol: **qaysi ish savol berilishidan oldin, qaysi ish savol berilgandan keyin bajariladi?**

### A. Indexing pipeline — hujjatlarni qidiruvga tayyorlash

```text
Hujjatlar
  → fayllarni o‘qish va tozalash
  → matnni tokenlar bo‘yicha parchalash
  → E5 bilan vektorlar yaratish
  → vektor, matn va manbani Qdrant’ga yozish
```

Ishga tushirish buyrug‘i:

```bash
python -m course_rag.index
```

Bu bosqichda hujjatlar tayyorlanadi. Qwen’ga savol yuborilmaydi.

### B. Retrieval & Generation — savolga javob berish

```text
Foydalanuvchi savoli
  → savolni E5 bilan vektorga aylantirish
  → Qdrant’dan eng yaqin parchalarni olish
  → savol va topilgan matnlarni Qwen’ga yuborish
  → javobni terminalga chiqarish
```

Ishga tushirish buyrug‘i:

```bash
python -m course_rag.ask "Yakuniy video necha soniya davom etadi?"
```

**Amaliy qoida:** hujjatlar o‘zgarmagan bo‘lsa, har bir savoldan oldin indeksni qayta yaratish shart emas. Bitta tayyor indeksdan ko‘p savol uchun foydalaniladi.

## 4. Birinchi bosqichda nima qurildi?

Yakuniy tizimdan oldin kichik RAG loyihasi yaratildi. Bu bosqich har bir komponentni alohida tekshirish imkonini berdi.

### 4.1. Lokal Python muhiti tayyorlandi

Loyiha papkasi:

```text
C:\Users\Rasulbekk\Desktop\RAG_PROJECT
```

Git Bash’da loyiha papkasiga kirish va mavjud virtual muhitni faollashtirish:

```bash
cd ~/Desktop/RAG_PROJECT
source .venv/Scripts/activate
python --version
```

Ishlagan Python versiyasi:

```text
Python 3.13.12
```

`.venv` loyiha uchun alohida Python muhiti. Terminaldagi `(.venv)` belgisi shu muhit faolligini ko‘rsatadi. `python -m pip` bilan o‘rnatilgan kutubxonalar shu muhitga tushadi.

### 4.2. Ollama o‘rnatildi va Qwen yuklandi

Boshlanishida Git Bash quyidagi xatoni qaytardi:

```text
bash: ollama: command not found
```

Ollama o‘rnatilgach, `ollama list` ishladi. Dastlab ro‘yxatda faqat ustun nomlari bor edi, chunki model hali yuklanmagan edi.

```bash
ollama pull qwen2.5:7b
ollama list
```

Qwen yuklamasi taxminan **4,7 GB** bo‘ldi. Keyin model mustaqil tekshirildi:

```bash
ollama run qwen2.5:7b "RAG nima? O‘zbek tilida bitta sodda gap bilan tushuntir."
```

Qwen javob qaytardi. Bu Ollama va model ishlayotganini ko‘rsatdi. Birinchi javobda Qwen RAG qisqartmasini noto‘g‘ri izohladi. Ushbu tajriba model ishlashi va javobning dalilga mosligi ikki alohida masala ekanini ko‘rsatdi.

### 4.3. E5 embedding modeli yuklandi

Model nomi:

```text
intfloat/multilingual-e5-base
```

Lokal saqlangan joy:

```text
models/e5-base/
```

Birinchi marta yuklash uchun ishlatilgan kod:

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
model.save_pretrained("models/e5-base")
print("Vektor o‘lchami:", model.get_sentence_embedding_dimension())
PY
```

E5 har bir kirish matni uchun **768 o‘lchamli vektor** beradi. Savol va hujjatni bir xil model yordamida vektorga aylantirganimiz uchun ularni bir vektor fazosida solishtira olamiz.

Loyihada E5’ga matn quyidagi shaklda beriladi:

```text
query: Bir o‘quvchi nechta kitob olishi mumkin?
passage: Kutubxonada har bir o‘quvchi bir vaqtning o‘zida ikkita kitob olishi mumkin.
```

`query: ` va `passage: ` prefikslari mos ravishda savol va hujjat matni uchun ishlatiladi. Ular foydalanuvchiga chiqadigan asl parcha tarkibiga kiritilmaydi.

### 4.4. E5 qidiruvi LLM’dan alohida tekshirildi

E5’ga uchta turli mavzudagi jumla berildi: dars vaqti, kutubxona qoidasi va HDMI kabelining joylashuvi.

Amaliy natija:

```text
Savol: Bir o‘quvchi nechta kitob olishi mumkin?
Topilgan matn: Kutubxonada har bir o‘quvchi bir vaqtning o‘zida ikkita kitob olishi mumkin.
```

Bu sinovda Qwen ishlatilmadi. Natija **qidiruv qismi** to‘g‘ri ishlaganini ko‘rsatdi.

### 4.5. NumPy asosidagi birinchi indeks yaratildi

Birinchi versiyada hujjatlar `data/docs/` ichiga joylandi. Har bir abzas bitta parcha hisoblandi. Vektorlar, matnlar va manba nomlari bitta faylga yozildi:

```text
data/index/rag.npz
```

Buyruqlar:

```bash
python -m rag.ingest
python app.py --search-only "Kitobni necha kunda qaytarish kerak?"
python app.py "Kitobni necha kunda qaytarish kerak?"
```

Olingan natija:

```text
Manba: kutubxona.txt | 2-parcha
Topilgan parcha: Kitobni qaytarish muddati 14 kun.
Qwen javobi: 14 kunda qaytarish kerak.
```

### 4.6. Indeksni yangilash tajribasi o‘tkazildi

`loyiha_topshirigi.txt` fayliga **Anor-27** kod nomi yozildi. Indeks yaratilgach, tizim kod nomini topdi. Keyin fayldagi qiymat **Anor-28** ga o‘zgartirildi.

| Holat | Hujjat ichida | Indeksdagi qiymat |
|---|---|---|
| Birinchi indeks yaratildi | Anor-27 | Anor-27 |
| TXT fayli tahrirlandi | Anor-28 | Anor-27 |
| Indeks yana yaratildi | Anor-28 | Anor-28 |

Bu tajribada muhim xulosa olindi: **faylni tahrirlash va qidiruv indeksini yangilash ikkita alohida amal**.

## 5. TACTIQ asosida loyiha qanday kengaytirildi?

TACTIQ loyihasidan indekslashni beshta tushunarli bosqichga ajratish yondashuvi olindi. Biz shu arxitekturani o‘zbekcha kurs hujjatlari va lokal modellarimizga mosladik.

| Bosqich | Bizning loyihada qanday bajarildi? |
|---|---|
| 01. Documents | Imtihon talablari, baholash va topshirish tartibi uchta alohida faylga yozildi |
| 02. Document loading | TXT va Markdown fayllar o‘qildi, matn tozalandi va manba nomi saqlandi |
| 03. Chunking | Uzun matnlar E5 tokenizeri bo‘yicha 160 tokenli qismlarga bo‘lindi |
| 04. Embedding | Oldindan yuklangan E5 orqali har bir parcha uchun 768 o‘lchamli vektor olindi |
| 05. Vector database | Parcha vektori va unga tegishli ma’lumotlar lokal Qdrant bazasiga yozildi |

Besh bosqichni ishga tushirish uchun bitta buyruq yetadi. Bizning versiyada bu bosqichlar `course_rag/index.py` ichidagi funksiyalarga ajratilgan. Qidiruv va javob yaratish esa `course_rag/ask.py` ichida.

**Arxitektura manbasi:** [TACTIQ — Indexing Pipeline](https://github.com/davronbek-malikov/TACTIQ/tree/edeee31510b5a35a6cfc8f2d354b990eb1881dd0/Indexing)

## 6. Yakuniy loyiha tuzilmasi

```text
RAG_PROJECT/
├── .venv/                         # Lokal Python muhiti
├── .gitignore
├── README.md
├── requirements.txt               # Asosiy Python kutubxonalari
├── app.py                         # NumPy versiyasi uchun terminal dasturi
│
├── models/
│   └── e5-base/                   # Lokal E5 modeli va tokenizer
│
├── rag/
│   ├── __init__.py
│   ├── embeddings.py              # Ikkala versiya ishlatadigan E5 funksiyalari
│   ├── ingest.py                  # Birinchi versiya uchun NumPy indeksi
│   ├── retrieve.py                # Birinchi versiyada eng yaqin bitta parcha
│   └── llm.py                     # Ikkala versiya ishlatadigan Qwen API moduli
│
├── data/
│   ├── docs/                      # Birinchi versiya hujjatlari
│   └── index/
│       ├── rag.npz                # Birinchi versiya indeksi
│       └── course/                # Kurs uchun Qdrant indekslari
│           ├── current            # Faol build papkasining nomi
│           └── build-.../
│               ├── db/            # Lokal Qdrant ma’lumotlari
│               └── chunks.jsonl    # Aynan qaysi parchalar indekslanganini ko‘rish
│
└── course_rag/
    ├── __init__.py
    ├── requirements.txt           # Qdrant uchun qo‘shimcha kutubxona
    ├── index.py                   # Beshta indekslash bosqichi
    ├── ask.py                     # Qidiruv, manbalarni chiqarish va Qwen javobi
    └── docs/
        ├── yakuniy_topshiriq.md
        ├── baholash.md
        └── topshirish.md
```

**Qaysi papkaga hujjat qo‘shish kerak?**

- `python app.py ...` orqali ishlatiladigan birinchi versiya uchun — `data/docs/`.
- `python -m course_rag.ask ...` orqali ishlatiladigan yakuniy kurs loyihasi uchun — `course_rag/docs/`.

Bu papkalar alohida indekslanadi. Masalan, `data/docs/` ichiga yozilgan yangi hujjat `course_rag.index` buyrug‘i bilan kurs indeksiga qo‘shilmaydi.

## 7. Kurs uchun qanday hujjatlar tayyorlandi?

### 7.1. `yakuniy_topshiriq.md` — o‘quvchi nima tayyorlaydi?

| Maydon | Hujjatdagi qiymat |
|---|---|
| Imtihon mavzusi | Sun’iy intellekt kelajagi |
| Ishlar soni | 3 ta |
| Ishlar turi | Rasm, videorolik, taqdimot |
| Rasm o‘lchami | 1920 × 1080 piksel |
| Video davomiyligi | 20 soniya |
| Taqdimot hajmi | 6 slayd |

### 7.2. `baholash.md` — ishlar qanday baholanadi?

| Ish turi | Ball |
|---|---:|
| Rasm | 25 |
| Videorolik | 40 |
| Taqdimot | 35 |
| **Jami** | **100** |

Video uchun ajratilgan 40 ballning taqsimoti ham hujjatda bor:

| Mezon | Ball |
|---|---:|
| G‘oya | 15 |
| Sahnalar izchilligi | 15 |
| Ovoz sifati | 10 |
| **Jami** | **40** |

### 7.3. `topshirish.md` — ishlar qanday topshiriladi?

| Topshiriladigan material | Format yoki tartib |
|---|---|
| Rasm | PNG |
| Video | MP4 |
| Taqdimot | PPTX |
| Ishlatilgan promptlar | `prompts.txt` |
| Umumiy tartib | Barcha ishlar bitta papkaga joylanadi |

Uchta faylning ajratilishi qidiruv sifatini tekshirishga yordam beradi. Masalan, *“Video qaysi formatda topshiriladi?”* savoliga javob `topshirish.md` ichida. *“Video uchun necha ball beriladi?”* savoliga javob esa `baholash.md` ichida.

## 8. `course_rag/index.py` qanday ishlaydi?

### 8.1. Fayllarni topish

`load_documents()` funksiyasi `course_rag/docs/` va uning ichki papkalarini ko‘rib chiqadi. `.txt` va `.md` kengaytmali fayllar tanlanadi. Fayllar tartiblangan ketma-ketlikda o‘qiladi.

### 8.2. Matnni o‘qish va tozalash

Fayllar `utf-8-sig` kodirovkasida o‘qiladi. Bu Windows’da UTF-8 BOM bilan saqlangan matnlarni ham ochish imkonini beradi.

Yuklash vaqtida:

- har bir satr oxiridagi ortiqcha bo‘shliqlar olib tashlanadi;
- satrlar bir xil yangi qator belgisi bilan yig‘iladi;
- hujjat boshi va oxiridagi ortiqcha bo‘shliqlar olib tashlanadi;
- bo‘sh hujjatlar o‘tkazib yuboriladi;
- manba sifatida hujjatning `docs/` papkasiga nisbatan yo‘li saqlanadi.

Agar papkada matnli hujjat topilmasa, indekslash tushunarli xabar bilan to‘xtaydi.

### 8.3. Tokenlar bo‘yicha parchalash

`split_text()` funksiyasi E5 tokenizeridan foydalanadi.

| Parametr | Qiymat | Ma’nosi |
|---|---:|---|
| `size` | 160 token | Bitta parchaning maksimal uzunligi |
| `overlap` | 24 token | Qo‘shni parchalar orasida takrorlanadigan qism |
| Keyingi boshlanish qadami | 136 token | `160 - 24` |

Masalan, uzun hujjatning birinchi parchasi indeks bo‘yicha `0–159` tokenlarni qamrasa, ikkinchi parcha `136`-tokendan boshlanadi. Shu yo‘l bilan chegaraga yaqin matnning bir qismi keyingi parchaga ham tushadi.

**Muhim tafsilot:** `160 token` — chegara. Qisqa hujjat bitta parcha bo‘lib qolishi tabiiy. Shuning uchun terminalda `topshirish.md | 1-parcha` ko‘rindi.

Tokenizer tokenlarning asl matndagi boshlanish va tugash o‘rinlarini ham beradi. Kod shu o‘rinlar orqali parchani asl hujjatdan kesib oladi. Natijada terminalda o‘qiladigan asl matn chiqadi.

### 8.4. Parcha uchun metadata tayyorlash

Har bir parcha uchun uchta maydon tayyorlanadi:

```json
{
  "source": "topshirish.md",
  "chunk": 1,
  "text": "Yakuniy rasm PNG formatida topshiriladi..."
}
```

Bu yerda:

- `source` — matn qaysi hujjatdan olingani;
- `chunk` — shu hujjat ichidagi parcha raqami, hisob 1 dan boshlanadi;
- `text` — Qwen’ga yuborilishi va terminalda ko‘rsatilishi mumkin bo‘lgan asl matn.

### 8.5. Embedding yaratish

`index.py` tayyor `rag/embeddings.py` modulidagi `embed_passages()` funksiyasini chaqiradi. Har bir parchaga `passage: ` prefiksi qo‘shiladi va E5 yordamida vektor hisoblanadi.

Model lokal papkadan ochiladi:

```text
models/e5-base/
```

Vektorlar normallashtiriladi. Vektor o‘lchami kodda tayyor natijaning `shape` qiymatidan olinadi. Bizning E5 modelimiz uchun bu **768**.

### 8.6. Qdrant collection yaratish

Collection nomi:

```text
course_chunks
```

Collection quyidagi sozlama bilan yaratiladi:

- vektor o‘lchami — E5 natijasidan olingan o‘lcham;
- o‘xshashlik turi — `COSINE`;
- saqlash turi — lokal disk;
- fayllar joylashuvi — tegishli `build-.../db/` papkasi.

Ushbu versiya uchun alohida Qdrant serveri yoki Docker ishga tushirish talab qilinmaydi.

### 8.7. Vektorlar va matnlarni yozish

Qdrant’dagi bitta yozuv `point` deyiladi. Loyiha har bir parcha uchun quyidagilarni yozadi:

| Qism | Tarkibi |
|---|---|
| `id` | Shu indeks qurilishidagi sonli identifikator |
| `vector` | E5 hisoblagan vektor |
| `payload.source` | Manba fayli |
| `payload.chunk` | Parcha raqami |
| `payload.text` | Parchaning matni |

`upsert(..., wait=True)` yozish yakunlanishini kutadi. Keyin `count()` yordamida yozuvlar soni tekshiriladi. Saqlangan vektorlar soni tayyorlangan parchalar soniga teng bo‘lishi kerak.

### 8.8. `chunks.jsonl` nima uchun kerak?

Har bir indekslangan parcha `chunks.jsonl` fayliga ham yoziladi. Har bir qatorda bitta JSON obyekt bo‘ladi.

Bu fayl orqali quyidagilarni tekshirish mumkin:

- Hujjat o‘qildimi?
- Matn qayerda bo‘lindi?
- Fayl nomi to‘g‘ri yozildimi?
- Kerakli fakt parchada bormi?
- Bir hujjatdan nechta parcha chiqdi?

Savolga noto‘g‘ri javob kelganda, model promptini o‘zgartirishdan oldin asl fakt indeksga kirganini shu fayldan tekshirish mumkin.

### 8.9. Faol indeks qanday tanlanadi?

Har safar `python -m course_rag.index` bajarilganda yangi `build-...` papkasi yaratiladi. Yangi Qdrant bazasi shu yerga yoziladi.

Indeks yozilib, vektorlar soni tekshirilgach, quyidagi fayl yangilanadi:

```text
data/index/course/current
```

`current` ichida faol `build-...` papkasining nomi turadi. Keyingi savolda `ask.py` aynan shu papkadagi bazani ochadi. Agar yangi indeks tayyorlash avvalroq xato bilan to‘xtasa, oldingi faol indeks ko‘rsatkichi saqlanadi.

Bu versiya har safar hujjatlar to‘plamini **to‘liq qayta indekslaydi**. Oldingi `build-...` papkalari diskda qoladi.

## 9. `course_rag/ask.py` qanday ishlaydi?

### 9.1. Savolni qabul qiladi

```bash
python -m course_rag.ask "Yakuniy video necha soniya davom etadi?"
```

`argparse` terminaldan savolni va qo‘shimcha parametrlarni oladi:

| Parametr | Vazifasi |
|---|---|
| `"Savol"` | Qidiruv uchun foydalanuvchi savoli |
| `--search-only` | Topilgan parchalarni ko‘rsatadi va shu yerda yakunlaydi |
| `--k` | Eng yaqin nechta parcha olinishini belgilaydi |

Standart qiymat `k=2`. Kod `1` dan `5` gacha qiymat qabul qiladi. Indeksda kamroq yozuv bo‘lsa, mavjud parchalar qaytadi.

### 9.2. Savol uchun vektor yaratadi

Savol `rag.embeddings.embed_question()` funksiyasiga beriladi. Ichkarida `query: ` prefiksi qo‘shiladi va E5 yordamida savol vektori hisoblanadi.

Hujjatlar va savol bir xil model bilan vektorga aylantirilishi muhim. Agar embedding model almashtirilsa, hujjat indeksini ham yangi model bilan qayta yaratish kerak.

### 9.3. Qdrant’dan eng yaqin parchalarni oladi

Dastur `current` faylini o‘qiydi, faol bazani ochadi va `query_points()` orqali qidiradi. `with_payload=True` bo‘lgani sababli qidiruv natijasida matn va manba ham keladi.

**Muhim:** indeksda savolga yaqindan bog‘liq parcha topilishi javob hujjatda aniq borligini kafolatlamaydi. Masalan, “Imtihon qachon?” savoliga ham imtihon haqidagi hujjatlar chiqishi mumkin, lekin ularning ichida sana bo‘lmasligi mumkin.

### 9.4. Manbalarni terminalga chiqaradi

Misol:

```text
[3] topshirish.md | 1-parcha
```

Bu yozuvda uchta turli ma’lumot bor:

- `[3]` — ushbu savolning qidiruv natijalari ichidagi uchinchi o‘rin;
- `topshirish.md` — manba fayli;
- `1-parcha` — shu faylning birinchi parchasi.

`[3]` hujjatning uchinchi parchasi degani emas. Natija tartibi har bir savolda o‘zgarishi mumkin.

### 9.5. Savol va kontekstni Qwen’ga yuboradi

Topilgan parchalar raqam va manba nomlari bilan birlashtiriladi. `rag/llm.py` ichidagi `answer()` funksiyasi shu matnlarni va savolni lokal API’ga yuboradi.

```text
POST http://127.0.0.1:11434/api/chat
```

| Sozlama | Qiymat | Vazifasi |
|---|---|---|
| `model` | `qwen2.5:7b` | Javob yozadigan lokal model |
| `stream` | `False` | To‘liq javobni bir JSON javobida olish |
| `temperature` | `0` | Javob yaratishdagi tasodifiylikni kamaytirish |
| `system` xabari | Parchaga tayanish, sonlarni saqlash, o‘zbekcha javob berish | Modelga javob yozish talablarini berish |
| `user` xabari | Topilgan parchalar va foydalanuvchi savoli | Javob uchun dalil va vazifani berish |
| `timeout` | `(5, 300)` | Ulanish va javobni kutish uchun alohida vaqt chegaralari |

Dastur Qwen yozgan matnni `response.json()["message"]["content"]` orqali oladi va `Qwen javobi:` qatorida chiqaradi.

Terminalda yuqorida ko‘rsatilgan manbalar — **Qwen’ga berilgan parchalar**. Qaysi parcha javobdagi qaysi fikrni tasdiqlashini asl matn bilan solishtirish mumkin.

## 10. Asosiy amaliy sinovlar va natijalar

### Sinov 1. Bitta manbadan aniq fakt olish

```bash
python app.py "Kitobni necha kunda qaytarish kerak?"
```

Kuzatilgan natija:

```text
Manba: kutubxona.txt | 2-parcha
Topilgan parcha: Kitobni qaytarish muddati 14 kun.
Qwen javobi: 14 kunda qaytarish kerak.
```

**Nima tekshirildi?** E5 tegishli parchani topdi, Qwen esa undagi sonni saqlab javob berdi.

### Sinov 2. Yangi hujjatdagi kod nomini topish

```bash
python app.py "Yakuniy loyihaning kod nomi nima?"
```

Kuzatilgan natija:

```text
Manba: loyiha_topshirigi.txt | 1-parcha
Topilgan parcha: RAG amaliyoti uchun yakuniy loyihaning kod nomi Anor-27.
Qwen javobi: Yakuniy loyihaning kod nomi Anor-27.
```

**Nima tekshirildi?** Yangi hujjat indeksga qo‘shildi va savolda aytilmagan maxsus nom manbadan olindi.

### Sinov 3. Ikkita manbadagi faktni birlashtirish

```bash
python -m course_rag.ask --k 3 "Yakuniy video qancha davom etadi va unga necha ball ajratilgan?"
```

Kuzatilgan natija:

```text
Qwen javobi: Yakuniy video 20 soniya davom etadi va unga 40 ball beriladi.
```

| Javobdagi fakt | Uni tasdiqlaydigan hujjat |
|---|---|
| Video 20 soniya | `yakuniy_topshiriq.md` |
| Video uchun 40 ball | `baholash.md` |

Terminalda uchinchi manba sifatida `topshirish.md` ham ko‘rindi. Unda video haqida ma’lumot bor, lekin u **MP4 formatiga** tegishli. Bu misol `top-k` natijalari ichida savolga to‘liq javob bermaydigan, mavzuga yaqin parchalar ham kelishi mumkinligini ko‘rsatadi.

### Sinov 4. Dalil yetarli bo‘lmagan holat

Terminalda quyidagi natija ham kuzatildi:

```text
Qwen javobi: Ma’lumot topilmadi
```

Ushbu natija yuborilgan terminal parchasida savolning o‘zi ko‘rinmagan. Shuning uchun uni talqin qilishda savolni ham hisobga olish kerak:

- Agar savol **“Yakuniy imtihon qaysi sanada bo‘ladi?”** bo‘lsa, javob o‘rinli. Uchta namuna hujjatida aniq imtihon sanasi yozilmagan.
- Agar savol **“Video qanday formatda topshiriladi?”** bo‘lsa, to‘g‘ri javob `topshirish.md` ichida bor: **MP4**. Bunday holatda kerakli hujjat Qwen’ga yetib borganini tekshirish kerak.

Ikkala holatni alohida tekshirish uchun:

```bash
python -m course_rag.ask --search-only --k 3 "Yakuniy video qaysi formatda topshiriladi?"
python -m course_rag.ask --k 3 "Yakuniy video qaysi formatda topshiriladi?"
python -m course_rag.ask --k 3 "Yakuniy imtihonning aniq sanasi qaysi?"
```

## 11. Uchragan xatolar va ularni qanday tekshirdik?

| Holat | Sabab yoki kuzatuv | Bajarilgan ish |
|---|---|---|
| `ollama: command not found` | Terminal Ollama buyrug‘ini topa olmadi | Ollama o‘rnatildi va terminal qayta ochildi |
| `ollama list` bo‘sh | Ollama bor, lokal model hali yuklanmagan | `ollama pull qwen2.5:7b` bajarildi |
| `can't open file ... app.py` | Dastlab loyihada faqat `.venv` bor edi | `app.py` va `rag/` modullari yaratildi |
| E5 har buyruqda `Loading weights` chiqardi | Har bir terminal buyrug‘i yangi Python jarayonini boshlaydi | Lokal modelning diskdan xotiraga yuklanishi kuzatildi |
| Manba topildi, so‘ng `HTTP 500` chiqdi | Qidiruv bajarildi; xato Ollama javob yaratish bosqichida yuz berdi | API alohida tekshirildi va asl so‘rov qayta bajarildi |
| ZIP faylni yuklash imkoni bo‘lmadi | Tayyor arxivdan foydalanib bo‘lmadi | `course_rag/` modullari va hujjatlari bitta Git Bash bloki bilan lokal yaratildi |

### Ollama API’ni alohida tekshirish buyrug‘i

```bash
curl -sS -w '\nHTTP %{http_code}\n' \
  http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:7b","stream":false,"messages":[{"role":"user","content":"Salom"}]}'
```

Amaliyotda ushbu so‘rov `HTTP 200` qaytardi. Keyingi urinishda to‘liq RAG savoli ham ishladi. Oldingi `500` xatosining aniq ichki sababi tasdiqlanmadi.

Xato takrorlanganda tafsilotni ko‘rish uchun `rag/llm.py` ichida server javobi ham chiqariladi:

```python
if not response.ok:
    raise SystemExit(f"Ollama HTTP {response.status_code}: {response.text}")
```

## 12. Kundalik foydalanish

### Kompyuter qayta yoqilgach

```bash
cd ~/Desktop/RAG_PROJECT
source .venv/Scripts/activate
ollama list
```

Ro‘yxatda `qwen2.5:7b` ko‘rinsa, mavjud kurs indeksiga savol bering:

```bash
python -m course_rag.ask "Yakuniy taqdimot nechta slayddan iborat?"
```

### Faqat topilgan manbalarni tekshirish

```bash
python -m course_rag.ask --search-only "Yakuniy imtihonda videoga necha ball beriladi?"
```

### Uchta parchani olish

```bash
python -m course_rag.ask --k 3 "Yakuniy video qancha davom etadi, qaysi formatda topshiriladi va necha ball beradi?"
```

Ushbu savol uchun kutiladigan uchta dalil:

| Dalil | Manba |
|---|---|
| 20 soniya | `yakuniy_topshiriq.md` |
| MP4 | `topshirish.md` |
| 40 ball | `baholash.md` |

### Yangi hujjat qo‘shish

Masalan, kurs jadvali uchun `course_rag/docs/jadval.md` faylini yarating va unga haqiqiy jadvalni yozing. So‘ng:

```bash
python -m course_rag.index
python -m course_rag.ask "Dars qaysi kuni boshlanadi?"
```

### Birinchi NumPy versiyasi bilan ishlash

```bash
python -m rag.ingest
python app.py --search-only "Kitobni necha kunda qaytarish kerak?"
python app.py "Kitobni necha kunda qaytarish kerak?"
```

### Qisqa buyruqlar jadvali

| Maqsad | Buyruq |
|---|---|
| Virtual muhitni faollashtirish | `source .venv/Scripts/activate` |
| VS Code’da papkani ochish | `code .` |
| Qwen yuklanganini tekshirish | `ollama list` |
| Kurs indeksini yaratish yoki yangilash | `python -m course_rag.index` |
| Faqat kurs hujjatlaridan qidirish | `python -m course_rag.ask --search-only "Savol"` |
| Kurs bo‘yicha manbali javob olish | `python -m course_rag.ask "Savol"` |
| Qidiriladigan parchalar sonini tanlash | `python -m course_rag.ask --k 3 "Savol"` |
| Eski NumPy indeksini yangilash | `python -m rag.ingest` |
| Virtual muhitdan chiqish | `deactivate` |

## 13. GitHub’da nimalar saqlanadi?

Loyiha reposi: [rasulbekdeveloper907/RAG_PROJECT](https://github.com/rasulbekdeveloper907/RAG_PROJECT)

Repo uchun asosiy fayllar — Python kodi, kutubxonalar ro‘yxati, README va qayta ishlatish mumkin bo‘lgan namuna hujjatlar. `.venv/`, yuklangan E5 modeli va avtomatik yaratilgan indekslar lokal kompyuterda saqlanadi.

Muhim farq: avvalgi `.gitignore` ichidagi `data/docs/` qoidasi birinchi versiya hujjatlariga tegishli. Kursning namunaviy hujjatlari **`course_rag/docs/`** ichida. Ularni kod bilan birga saqlasangiz, boshqa kompyuterda ham aynan shu misolni qayta indekslash mumkin.

Yangi modul va README’ni ko‘rib chiqib GitHub’ga yuborish:

```bash
cd ~/Desktop/RAG_PROJECT
git status -sb
git add course_rag/ README.md
git diff --cached --stat
git commit -m "Add course exam RAG indexing pipeline with local Qdrant"
git push origin main
```

## 14. Yakuniy natija va loyiha chegarasi

**Amalda tasdiqlangan natija:** lokal hujjatlar indekslandi, E5 orqali qidirildi, Qdrant’dan manbalar qaytdi va Qwen ikki hujjatdagi faktni birlashtirib **“20 soniya va 40 ball”** deb javob berdi. Shu bilan namuna hujjatlar uchun indekslashdan yakuniy javobgacha bo‘lgan jarayon ishladi.

Joriy versiya quyidagi sharoitga mo‘ljallangan:

- kirish hujjatlari — UTF-8 formatidagi `.txt` va `.md`;
- ishlash muhiti — bitta lokal Windows kompyuteri;
- interfeys — Git Bash terminali;
- ma’lumot yangilash usuli — hujjatlarni tahrirlash va to‘liq qayta indekslash;
- javobni tekshirish usuli — chiqarilgan manba va asl parchani Qwen javobi bilan solishtirish.

Loyihani namoyish qilishda eng foydali ketma-ketlik: avval `python -m course_rag.index` bilan nechta hujjat va vektor yozilganini ko‘rsating. Keyin `--search-only` bilan dalillarni chiqaring. Oxirida aynan shu savolni odatiy rejimda bering. Shunda **indekslash, qidiruv va javob yaratish** bosqichlarining har biri alohida ko‘rinadi.