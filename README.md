# TUSAS DOCQA

PDF ve gorsel belgeler uzerinde calisan belge soru-cevap sistemi. Pipeline, belgeyi ingest eder, yapiyi cikartir, hiyerarsik chunk'lara boler, Chroma + BM25 ile retrieval yapar ve cevabi kaynakla birlikte uretir.

## Ozet

- Belge turleri: `PDF`, `PNG`, `JPG`
- Retrieval: dense + sparse + RRF
- Query routing: `section_list` ve `normal_qa`
- Generation: Gemini, OpenAI, local Ollama veya extractive mod
- VLM extraction: Gemini veya local vision model
- Demo icin onerilen profil: Vertex AI + Gemini

## Onerilen Demo Profili

Mevcut stabil profil:

```ini
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.1-pro-preview
GEMINI_FALLBACK_MODEL=gemini-2.5-pro

EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=3072

VLM_PROVIDER=gemini
VLM_MODE=force
OCR_ENABLED=0

VERTEX_ENABLED=1
VERTEX_PROJECT_ID=your-gcp-project
VERTEX_LOCATION=global
VERTEX_REQUEST_TIMEOUT_MS=120000
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/service-account.json
```

Bu profilin mantigi:

- generation: `gemini-3.1-pro-preview`
- fallback: `gemini-2.5-pro`
- embedding: `gemini-embedding-001` `3072d`
- VLM extraction: Gemini, varsayilan olarak acik
- klasik Tesseract OCR: kapali

## Mimari

```text
PDF/Image
  -> Ingestion
  -> Structure Detection
  -> Hierarchical Chunking
  -> Indexing (Chroma + BM25)
  -> Retrieval (Hybrid + RRF + section fetch)
  -> Generation (Gemini/OpenAI/Ollama/Extractive)
```

Ana teknik kararlar:

- `Hybrid retrieval`: exact keyword ve semantic intent birlikte yakalansin
- `Hierarchical chunking`: belge baslik yapisi korunarak section-level retrieval yapilsin
- `Section-list routing`: "nelerdir", "listele" gibi sorularda tum bolum getirilsin
- `Guarded generation`: baglam disi cevap engellensin, citation zorunlu olsun

## Kurulum

### 1) Ortam

Python `3.11+` onerilir.

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Not:

- Varsayilan profil Gemini embedding kullandigi icin lokal GPU zorunlu degildir.
- Lokal `sentence-transformers` embedding kullanacaksan `GPU_REQUIREMENTS.md` icindeki CUDA PyTorch adimini da uygula.

### 2) Konfigurasyon

Ornek dosyayi kopyala:

```bash
cp .env.example .env
```

#### Vertex AI ile calisma

Onerilen yol budur.

```ini
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.1-pro-preview
GEMINI_FALLBACK_MODEL=gemini-2.5-pro

EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=3072

VLM_PROVIDER=gemini
VLM_MODE=force
OCR_ENABLED=0

VERTEX_ENABLED=1
VERTEX_PROJECT_ID=your-gcp-project
VERTEX_LOCATION=global
VERTEX_REQUEST_TIMEOUT_MS=120000
GOOGLE_APPLICATION_CREDENTIALS=/abs/path/service-account.json
```

Gerekli GCP adimlari:

1. Vertex AI API'yi ac
2. Billing aktif olsun
3. Service account JSON olustur
4. `GOOGLE_APPLICATION_CREDENTIALS` ile JSON dosyasini ver

#### AI Studio ile calisma

Vertex kullanmayacaksan:

```ini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-ai-studio-key
GEMINI_MODEL=gemini-3.1-pro-preview
GEMINI_FALLBACK_MODEL=gemini-2.5-pro

EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=3072

VLM_PROVIDER=gemini
VLM_MODE=force
OCR_ENABLED=0

VERTEX_ENABLED=0
```

Not:

- Vertex env degiskenleri set ise AI Studio key ile `401 UNAUTHENTICATED` alirsin.
- AI Studio ile calisacaksan `VERTEX_ENABLED=0` ve eski `GOOGLE_GENAI_USE_VERTEXAI` benzeri shell env'leri kapat.

#### Tamamen lokal mod

```ini
LLM_PROVIDER=local
VLM_PROVIDER=local
EMBEDDING_MODEL=auto
```

Bu profil Ollama gerektirir. Ayrinti icin `chainlit.md` ve `.env.example`.

## Calistirma

Tek onerilen launch yolu:

```bash
./run.sh
```

Bu script:

- `.env` dosyasini yukler
- `.venv-gpu/bin/python` kullanir
- `scripts/preflight.py` calistirir
- sonra Chainlit uygulamasini baslatir

Varsayilan adres:

```text
http://127.0.0.1:8000
```

Host ve port override:

```bash
HOST=0.0.0.0 PORT=8001 ./run.sh
```

Preflight'i tek basina calistirmak icin:

```bash
.venv-gpu/bin/python scripts/preflight.py
```

Preflight su kontrolleri yapar:

- auth / client kurulumu
- generation modeli erisilebilir mi
- embedding cagrisi calisiyor mu
- VLM extraction cagrisi donuyor mu

## UI ve Demo Notlari

Chainlit UI runtime ayarlari sunar:

- `Embedding Model`
- `Embedding Device`
- `VLM Mode`
- `VLM Provider`
- `VLM Max Pages`

Ancak demo sirasinda bunlari degistirmemen tavsiye edilir. Ozellikle:

- embedding model degisirse indeks yeniden olusturulur
- VLM ayarlari yeni yuklemelerde fark yaratir
- provider degisikligi, state'i karmasiklastirir

Mülakat icin en guvenli yol: `.env` ile sabit profil + `./run.sh`.

## OCR, VLM ve GPU

### OCR

- `OCR_ENABLED=0`: klasik Tesseract kapali
- `OCR_ENABLED=1`: Tesseract ek extraction adayi olarak devreye girer

Tesseract gerekiyorsa:

- `TESSERACT_CMD`
- `TESSDATA_PREFIX`
- `TESSERACT_CONFIG`

ayarlarini `.env` icinde ver.

### VLM

Varsayilan demo profilinde:

```ini
VLM_PROVIDER=gemini
VLM_MODE=force
```

Bu, gorsel/PDF extraction sirasinda Gemini VLM'in zorlanarak kullanilmasi anlamina gelir.

### GPU

GPU kullanimi profile gore degisir:

- Gemini generation / Gemini embedding / Gemini VLM: uzak servis, lokal GPU kullanmaz
- Lokal `sentence-transformers` embedding: lokal GPU kullanabilir
- Ollama local LLM/VLM: kendi surecinde GPU kullanabilir

Detayli notlar icin `GPU_REQUIREMENTS.md`.

## Test ve Dogrulama

Onerilen siralama:

```bash
.venv-gpu/bin/python scripts/baseline_gate.py
.venv-gpu/bin/python scripts/lang_gate.py
.venv-gpu/bin/python scripts/eval_retrieval.py --pdf test_data/Case_Study_20260205.pdf
.venv-gpu/bin/python scripts/eval_case_study.py --pdf test_data/Case_Study_20260205.pdf
```

Hizli smoke:

```bash
.venv-gpu/bin/python scripts/smoke_suite.py test_data/Case_Study_20260205.pdf
```

Tum test notlari icin `TESTING.md`.

## CI

Workflow: `.github/workflows/ci.yml`

- `baseline_gate.py`
- `lang_gate.py`
- `eval_case_study.py` `GEMINI_API_KEY` veya uygun Gemini auth varsa opsiyonel

## Troubleshooting

### 401 UNAUTHENTICATED

Tipik neden: AI Studio API key ile Vertex endpoint'ine gitmek.

Kontrol et:

- `VERTEX_ENABLED`
- `GOOGLE_APPLICATION_CREDENTIALS`
- shell icindeki eski `GOOGLE_GENAI_USE_VERTEXAI`

AI Studio kullanacaksan Vertex'i kapat. Vertex kullanacaksan service account ile devam et.

### 404 NOT_FOUND / model bulunamadi

Tipik nedenler:

- model ilgili proje/hesap icin acik degil
- yanlis region
- preview model erisimi yok

Oneri:

- `VERTEX_LOCATION=global`
- primary: `gemini-3.1-pro-preview`
- fallback: `gemini-2.5-pro`

### Port 8000 dolu

Farkli portla baslat:

```bash
PORT=8001 ./run.sh
```

### Belge yuklendi ama indeks olusmadi

Kontrol et:

- extraction bossa VLM/OCR gerekli olabilir
- `VLM_MAX_PAGES` limiti dusuk olabilir
- taranmis PDF ise `OCR_ENABLED=1` ve Tesseract kurulu mu bak

### Lokal GPU gorunuyor ama hizlanma yok

Bu normal olabilir. Varsayilan profil uzak Gemini embedding kullaniyor. Lokal GPU sadece lokal embedding/Ollama yolunda fark yaratir.

## Proje Yapisi

```text
app.py
run.sh
requirements.txt
README.md
TESTING.md
GPU_REQUIREMENTS.md
chainlit.md
.env.example

scripts/
  baseline_gate.py
  eval_case_study.py
  eval_retrieval.py
  hallucination_test.py
  preflight.py
  smoke_suite.py

src/
  config.py
  core/
    embedding.py
    gemini_client.py
    generation.py
    indexing.py
    ingestion.py
    pipeline.py
    retrieval.py
    structure.py
    vlm_extract.py
```

## Not

Bu repo halen case-study / MVP karakterinde. Guclu taraflari retrieval tasarimi, kaynakli cevap ve coklu calisma modlari. Uretim ortamina tasinacaksa auth, tenancy, observability ve cost control taraflari ayrica sertlestirilmelidir.
