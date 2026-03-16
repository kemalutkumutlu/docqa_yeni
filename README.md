# TUSAS DOCQA

PDF ve gorsel belgeler uzerinde calisan belge soru-cevap sistemi. Proje, belgeyi ingest eder, gerekirse OCR ve gorsel extract uygular, hiyerarsik chunk'lara boler, Chroma + BM25 ile retrieval yapar ve cevabi kaynaklariyla birlikte uretir.

## Bugunku Durum

- Belge turleri: `PDF`, `PNG`, `JPG`
- Retrieval: dense + sparse + RRF
- Generation: `gemini`, `openai`, `local` veya `extractive`
- OCR backends: `docai`, `paddle_vl`, `paddle`, `tesseract_legacy`
- Layout detector backends: `none`, `sidecar`, `docai`, `docling`
- Table structure backends: `off`, `auto`, `docai`, `gemini`, `heuristic`
- PDF text backends: `pymupdf` (default), `docling`, `auto`
- Processing mode: `classic` veya `multimodal`
- UI: Chainlit tabanli; `Basic` ve `Advanced` ayar sekmeleri var
- Desktop UI: sol tarafta custom sohbet gecmisi paneli, sag tarafta custom runtime settings paneli

## Mimari

Ana akis:

```text
PDF/Image
  -> Ingestion (OCR + VLM Fusion)
  -> Structure Detection
  -> Hierarchical Chunking (Token-based)
  -> Indexing (Chroma + BM25)
  -> Retrieval (Hybrid + RRF + section fetch)
  -> Generation (Gemini/OpenAI/Ollama/Extractive)
```

Multimodal akis etkinse:

```text
PDF/Image
  -> Page/Region Visual Assets
  -> Layout Regions (heuristic / sidecar / docai / docling)
  -> OCR & VLM Fusion Extraction
  -> Optional Table Structure Stage
  -> Visual + Text Chunks
  -> Hybrid Retrieval
  -> Answer Generation
```

Gercek davranis:

- `classic`: text-first akis
- `multimodal`: text akis korunur, buna ek olarak visual chunks ve visual evidence yolu eklenir
- `VISUAL_CHUNK_LEVEL=region`: detector veya heuristic region planning kullanilir
- `TABLE_STRUCTURE_ENABLED=1`: sadece tablo gorunen region'larda table extraction denenir

## Online / Local Yol Haritasi

### Online taraf

- `Document AI OCR`
- `Document AI Layout Parser`
- `Document AI Form/Table Parser`
- `Gemini generation`
- `Gemini VLM extract`
- `Gemini embedding`

### Local taraf

- `PaddleOCR-VL-1.5`
- `PaddleOCR`
- `Tesseract legacy`
- `Docling layout detector`
- `Docling text extraction` (PDF text backend)
- `Ollama LLM`
- `Ollama VLM`
- local `sentence-transformers` embedding

### Hybrid pratikte ne demek?

Bu proje tek bir "online mode" veya "local mode" ile sinirli degil. Ornegin:

- OCR local olabilir, generation Gemini olabilir
- layout detector local `docling`, table parser `gemini` olabilir
- generation `local`, embedding `gemini` olabilir

## Onerilen Profiller

### 1. Online Best

Genel demo ve en guclu kalite icin:

```ini
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.1-pro-preview
GEMINI_FALLBACK_MODEL=gemini-2.5-pro

DOC_PROCESSING_MODE=multimodal
MULTIMODAL_ANSWER_MODE=auto
EMBEDDING_MODEL=gemini-embedding-2-preview

OCR_ENABLED=1
OCR_BACKEND=docai

VISUAL_CHUNK_LEVEL=region
VISUAL_REGION_SOURCE=detector
VISUAL_DETECTOR_BACKEND=docai

TABLE_STRUCTURE_ENABLED=1
TABLE_STRUCTURE_BACKEND=auto

VLM_PROVIDER=gemini
VLM_MODE=force
```

Gerekli Google alanlari:

- `DOCAI_PROJECT_ID`
- `DOCAI_LOCATION`
- `DOCAI_OCR_PROCESSOR_ID`
- `DOCAI_LAYOUT_PROCESSOR_ID`
- opsiyonel `DOCAI_TABLE_PROCESSOR_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`

### 2. Hybrid Best

OCR local, layout online:

```ini
LLM_PROVIDER=gemini
DOC_PROCESSING_MODE=multimodal
MULTIMODAL_ANSWER_MODE=auto

OCR_ENABLED=1
OCR_BACKEND=paddle_vl
OCR_DEVICE=cuda

VISUAL_CHUNK_LEVEL=region
VISUAL_REGION_SOURCE=detector
VISUAL_DETECTOR_BACKEND=docai

TABLE_STRUCTURE_ENABLED=1
TABLE_STRUCTURE_BACKEND=auto

VLM_PROVIDER=gemini
VLM_MODE=force
```

Bu profil, Paddle OCR GPU ve Google layout/table servislerini birlikte kullanir.

### 3. Local Best

Maksimum offline taraf:

```ini
LLM_PROVIDER=local
VLM_PROVIDER=local
DOC_PROCESSING_MODE=multimodal

OCR_ENABLED=1
OCR_BACKEND=paddle_vl
OCR_DEVICE=cuda

VISUAL_CHUNK_LEVEL=region
VISUAL_REGION_SOURCE=detector
VISUAL_DETECTOR_BACKEND=docling

TABLE_STRUCTURE_ENABLED=1
TABLE_STRUCTURE_BACKEND=heuristic

EMBEDDING_MODEL=auto
EMBEDDING_DEVICE=cuda
```

Not:

- Bu profil icin `ollama`, `paddle` ve `docling` kurulumlari ayrica gerekli olabilir.
- Table stage local tarafta simdilik `heuristic` veya `gemini` fallback ile en guclu hale geliyor; tam local Document AI esleniği yok.

## OCR, Layout ve Table Stage

### OCR

Desteklenen backend'ler:

- `docai`: online OCR
- `paddle_vl`: `PaddleOCR-VL-1.5`
- `paddle`: standart PaddleOCR
- `tesseract_legacy`: son fallback

Gercek fallback zinciri:

- `paddle_vl` secilirse: `paddle_vl -> paddle -> tesseract_legacy`
- `paddle` secilirse: `paddle -> tesseract_legacy`
- `docai` secilirse: `docai -> tesseract_legacy`

**OCR + VLM Fusion (10/10 Synergy):**
Eğer VLM devreye girerse, OCR metni (eğer varsa) VLM'e "Grounding Truth" (Dayanak) olarak `ocr_context` biçiminde gönderilir. VLM numaraları veya harfleri halüsinasyon yapmaz, sadece OCR metninin mizanpajını orijinal görsele bakarak mükemmel seviyede düzenler.

### Layout detector

Desteklenen backend'ler:

- `none`: detector yok
- `sidecar`: JSON bbox input
- `docai`: Google Document AI Layout Parser
- `docling`: lokal layout detector

Docling backend icin:

- `DOCLING_PYTHON_BIN` ayarlanabilir
- `DOCLING_LAYOUT_MODEL` varsayilan: `docling-layout-heron-101`
- detector subprocess ile calisir; ana uygulamayi dusurmez

### PDF text backend

PDF metin cikarimi icin kullanilan backend secimi:

- `pymupdf` (default): PyMuPDF ile hizli metin cikarimi
- `docling`: Docling ile yapisal metin + tablo markdown cikarimi
- `auto`: `DOCLING_PYTHON_BIN` ayarliysa Docling kullanir, degilse PyMuPDF

Secim icin:

- `PDF_TEXT_BACKEND=docling` `.env` dosyasina eklenebilir
- veya UI Advanced ayarlarindan runtime'da degistirilebilir

Not:

- `docling` secilirse `DOCLING_PYTHON_BIN` ile ayri bir venv gosterilmesi gerekir (`.venv-docling`)
- Docling PDF uzerinde tablo yapisini markdown olarak cikarir; ayri table structure stage'ine gerek kalmaz
- Docling kazandigi sayfalarda OCR tetiklenmez

### Table structure

Table stage artik ayri bir katmandir.

- `TABLE_STRUCTURE_ENABLED=1` ise sadece table benzeri region'larda calisir
- `TABLE_STRUCTURE_BACKEND=auto` ise oncelik:
  - `docai`
  - `gemini`
  - `heuristic`

Bu stage mevcut text/OCR akisinin yerine gecmez; ona ek bilgi uretir.

## UI

Chainlit UI su an:

- chat/doc modlarini destekler
- birden fazla belgeyi ayni session icinde tutabilir
- desktop'ta custom `Basic` ve `Advanced` settings paneli sunar
- `Runtime Preset` ile hazir kombinasyon secilebilir
- disabled alanlarla bagimliliklari gorunur hale getirir
- `Current Draft`, `Applied Pipeline`, `Fallback Notes` ve `Document Context` ozet kartlari vardir
- onceki ayri `Belge Durumu` element sidebar'i kaldirilmistir; belge baglami runtime paneline tasinmistir
- sol sohbet panelinde TUSAS markalama alani bulunur

Onemli sinir:

- custom runtime panel degisiklikleri backend'e `Apply` ile yollar
- draft secimleri panel icinde hemen gorunur, ancak pipeline sadece `Apply` sonrasi degisir
- mode/backend degisikligi mevcut yuklu belgeleri geriye donuk degistirmez; gerekirse belge yeniden yuklenmelidir

Detay icin: `chainlit.md`

## Kurulum

### 1. Ortam

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Opsiyonel OCR/layout bagimliliklari:

```bash
python -m pip install -r requirements-ocr-offline.txt
```

Not:

- `requirements-ocr-offline.txt` tek basina her seyi bitirmez; `paddlepaddle` ve benzeri platform-spesifik paketleri ayri kurman gerekebilir.
- `docling` icin ayri bir venv kullanmak tavsiye edilir.

### 2. Konfigurasyon

```bash
cp .env.example .env
```

En kritik alanlar:

- `LLM_PROVIDER`
- `DOC_PROCESSING_MODE`
- `OCR_ENABLED`
- `OCR_BACKEND`
- `VISUAL_CHUNK_LEVEL`
- `VISUAL_REGION_SOURCE`
- `VISUAL_DETECTOR_BACKEND`
- `TABLE_STRUCTURE_ENABLED`
- `TABLE_STRUCTURE_BACKEND`
- `VLM_MODE`
- `VLM_PROVIDER`
- `PDF_TEXT_BACKEND` (opsiyonel, default: `pymupdf`)

Google kullaniyorsan:

- `DOCAI_PROJECT_ID`
- `DOCAI_LOCATION`
- processor id alanlari
- `GOOGLE_APPLICATION_CREDENTIALS`

## Calistirma

Tek onerilen launch yolu:

```bash
./run.sh
```

Bu script:

- `.env` dosyasini yukler
- `.venv-gpu/bin/python` kullanir
- istenirse `ollama serve` baslatir
- `scripts/preflight.py` calistirir
- Chainlit uygulamasini baslatir

Varsayilan adres:

```text
http://127.0.0.1:8000
```

Host ve port override:

```bash
HOST=0.0.0.0 PORT=8001 ./run.sh
```

## Yardimci Scriptler

Ornek layout scriptleri:

```bash
.venv-gpu/bin/python scripts/generate_layout_sidecar.py test_data/Case_Study_20260205.pdf --level region --output ./data/layout_sidecars/Case_Study_20260205.regions.json
.venv-gpu/bin/python scripts/validate_layout_sidecar.py ./data/layout_sidecars/Case_Study_20260205.regions.json --document test_data/Case_Study_20260205.pdf
.venv-gpu/bin/python scripts/inspect_regions.py test_data/Case_Study_20260205.pdf --level region --source detector --detector-backend docai
```

## Test

Hizli sira:

```bash
.venv-gpu/bin/python scripts/preflight.py
.venv-gpu/bin/python scripts/baseline_gate.py
.venv-gpu/bin/python scripts/lang_gate.py
.venv-gpu/bin/python scripts/smoke_suite.py test_data/Case_Study_20260205.pdf
```

Detaylar icin: `TESTING.md`

## Repo Yapisi

```text
app.py
run.sh
README.md
TESTING.md
DEVLOG.md
GPU_REQUIREMENTS.md
SUNUM_ICERIGI.md
chainlit.md
.env.example

scripts/
  baseline_gate.py
  build_index.py
  docling_layout_runner.py
  docling_text_runner.py
  eval_case_study.py
  eval_retrieval.py
  extract_text.py
  folder_suite.py
  generate_layout_sidecar.py
  hallucination_test.py
  inspect_regions.py
  lang_gate.py
  paddle_ocr_runner.py
  preflight.py
  preview_structure.py
  search_index.py
  smoke_suite.py
  validate_layout_sidecar.py

src/
  config.py
  core/
    content_normalization.py
    doc_cache.py
    embedding.py
    eventlog.py
    evidence.py
    gemini_client.py
    generation.py
    hybrid.py
    indexing.py
    ingestion.py
    layout_detector.py
    layout_regions.py
    local_llm.py
    models.py
    multimodal.py
    ocr_backend.py
    pipeline.py
    prompts.py
    query_classification.py
    ranking.py
    retrieval.py
    sparse.py
    structure.py
    table_structure.py
    utils.py
    vectorstore.py
    vlm_extract.py
```

## Kisa Not

Proje bugun itibariyla "tek bir sade RAG" olmaktan cikmis durumda; artik OCR, layout, table structure ve local/online karisik yollari olan bir belge zekasi sistemi. Bu nedenle dogru kullanim yolu:

- once profil/preset sec
- sonra belgeyi yukle
- mode veya backend degisirse belgeyi yeniden yukle
