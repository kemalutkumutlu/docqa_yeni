# TESTING

Bu belge, projenin bugunku hali icin onerilen test akisini ozetler. Amaç sadece LLM cevabini degil, OCR, layout, table stage ve UI davranisini da dogrulamaktir.

## Onerilen Sira

### 0. Launch ve preflight

Tek tavsiye edilen baslatma yolu:

```bash
./run.sh
```

Bu komut:

- `.env` yukler
- gerekirse `ollama serve` baslatir
- `scripts/preflight.py` calistirir
- sonra Chainlit'i acar

Preflight tek basina:

```bash
.venv-gpu/bin/python scripts/preflight.py
```

Beklenen:

- `[OK] generation ...`
- `[OK] embedding ...`
- `[OK] vlm ...`
- `[OK] preflight passed`

### 1. Syntax ve import kapisi

En hizli regresyon kapisi:

```bash
.venv-gpu/bin/python -m py_compile app.py src/config.py src/core/*.py scripts/*.py
```

Bu adim:

- import hatalari
- syntax hatalari
- yeni dosyalarin bozuk olmasi

gibi sorunlari en hizli yakalar.

### 2. LLM-free baseline gate

```bash
.venv-gpu/bin/python scripts/baseline_gate.py
.venv-gpu/bin/python scripts/lang_gate.py
```

Amac:

- ingestion -> structure -> chunk -> retrieval zinciri
- section-list routing
- coklu belge izolasyonu
- TR/EN dil davranisi

### 3. Retrieval kalitesi

```bash
.venv-gpu/bin/python scripts/eval_retrieval.py --pdf test_data/Case_Study_20260205.pdf
```

Olculen basliklar:

- intent accuracy
- heading hit
- section hit
- evidence recall
- latency

### 4. Gemini kabul kapisi

```bash
.venv-gpu/bin/python scripts/eval_case_study.py --pdf test_data/Case_Study_20260205.pdf
.venv-gpu/bin/python scripts/hallucination_test.py --pdf test_data/Case_Study_20260205.pdf
```

Kontrol edilenler:

- citation zorunlulugu
- coverage
- hallucination guard
- negatif sorularda "Belgede bu bilgi bulunamadi."

### 5. Smoke suite

```bash
.venv-gpu/bin/python scripts/smoke_suite.py test_data/Case_Study_20260205.pdf
```

Birden fazla PDF ile:

```bash
.venv-gpu/bin/python scripts/folder_suite.py --dir test_data --mode retrieval --isolate 1
.venv-gpu/bin/python scripts/folder_suite.py --dir test_data --mode ask --isolate 1 --max_pdfs 3
```

## OCR ve Layout Ozel Testleri

### OCR backend smoke

Document AI OCR:

```bash
.venv-gpu/bin/python - <<'PY'
from PIL import Image
from src.config import load_settings
from src.core.ocr_backend import OCRConfig, ocr_image_text
s = load_settings()
img = Image.new("RGB", (800, 200), "white")
cfg = OCRConfig(
    enabled=True,
    backend="docai",
    lang=s.ocr_lang,
    device=s.ocr_device,
    paddle_ocr_version=s.paddle_ocr_version,
    tesseract_cmd=s.tesseract_cmd,
    tessdata_prefix=s.tessdata_prefix,
    tesseract_config=s.tesseract_config,
    docai_project_id=s.docai_project_id,
    docai_location=s.docai_location,
    docai_processor_id=s.docai_ocr_processor_id,
    docai_processor_version=s.docai_ocr_processor_version,
    docai_timeout_seconds=s.docai_timeout_seconds,
)
print(ocr_image_text(img, cfg=cfg, document_name="smoke", page_number=1, image_kind="image"))
PY
```

Paddle OCR:

```bash
.venv-gpu/bin/python scripts/paddle_ocr_runner.py --help
```

Beklenen:

- script acilmali
- subprocess crash olsa bile ana app dusmemeli

### Layout detector smoke

Heuristic / sidecar / docai / docling yolunu inspect etmek icin:

```bash
.venv-gpu/bin/python scripts/inspect_regions.py test_data/Case_Study_20260205.pdf --level region --source heuristic
.venv-gpu/bin/python scripts/inspect_regions.py test_data/Case_Study_20260205.pdf --level region --source detector --detector-backend sidecar --detector-dir ./data/layout_sidecars
.venv-gpu/bin/python scripts/inspect_regions.py test_data/Case_Study_20260205.pdf --level region --source detector --detector-backend docai
```

Docling backend ayri python ile kuruluysa:

```bash
DOCLING_PYTHON_BIN=/abs/path/to/.venv-docling/bin/python \
.venv-gpu/bin/python scripts/inspect_regions.py test_data/Case_Study_20260205.pdf --level region --source detector --detector-backend docling
```

### Table structure smoke

Table stage aciksa:

- table region olan bir PDF yukle
- retrieval sonrasi `table` chunk'larin olustugunu kontrol et
- generation tarafinda tablo sorusuna cevap dene

Pratik sorular:

- `Tablodaki kalemler nelerdir?`
- `Bu tabloda hangi sutunlar var?`
- `Tablonun ikinci satirinda ne yaziyor?`

## UI Testleri

### Settings panel

Manual kontrol:

1. `./run.sh`
2. Settings panel ac
3. `Basic` tabda sadece ust seviye ayarlar gorunmeli
4. `Advanced` tabda detay alanlar gorunmeli
5. Desteklenmeyen alanlar kaybolmamak yerine `disabled` olmali

Beklenen:

- `OCR=off` ise `OCR Backend` gorunur ama secilemez
- `VLM Mode=off` ise `VLM Provider` ve `VLM Max Pages` gorunur ama secilemez
- `classic` modda layout/table zinciri gorunur ama secilemez

Onemli sinir:

- Chainlit settings backend'e `Confirm` ile uygular
- yani secim aninda pipeline backend tarafinda yeniden hesaplanmaz
- bu bug degil, mevcut panel yuzeyinin siniridir

### Belge yukleme

Kontrol:

1. belge yukle
2. soru sor
3. mode/backend degistir
4. ayni belgeyi yeniden yukle

Beklenen:

- yeni mode/backend yalnizca yeni ingestionlarda etkili olur
- yuklu belge eski ayarlarla kalir

## Demo Oncesi Minimum Checklist

1. `./run.sh`
2. preflight passed
3. `Case_Study_20260205.pdf` yukle
4. Su sorulari sor:
   - `Fonksiyonel gereksinimler nelerdir?`
   - `Teslimatlar nelerdir?`
   - `Teslim suresi nedir?`
   - `Projenin amaci nedir?`
   - `Araba kac beygir?`

Beklenen:

- ilk iki soru `section_list`
- ortadakiler normal QA
- sonuncusu negatif guard
- cevaplarda citation olsun

Ek multimodal kontrol:

1. `Processing Mode=multimodal`
2. `Visual Chunk Level=region`
3. detector backend sec
4. belgeyi yeniden yukle
5. tablo/layout sorusu sor

## Sik Karsilasilan Sorunlar

### Preflight generation fail

Kontrol et:

- `LLM_PROVIDER`
- `GEMINI_MODEL` veya `OPENAI_MODEL`
- Vertex kullaniyorsan auth ve location
- AI Studio kullaniyorsan `VERTEX_ENABLED=0`

### OCR var ama sonuc bos

Kontrol et:

- `OCR_ENABLED=1`
- backend gercekten kurulu mu
- `docai` ise processor id dolu mu
- `paddle` ise opsiyonel paketler kurulu mu
- `tesseract_legacy` ise binary erisilebilir mi

### Layout detector calismiyor

Kontrol et:

- `VISUAL_REGION_SOURCE=detector`
- `VISUAL_DETECTOR_BACKEND`
- `docai` ise processor id
- `docling` ise `DOCLING_PYTHON_BIN`
- `sidecar` ise JSON dosyasi

### Table stage calismiyor

Kontrol et:

- `TABLE_STRUCTURE_ENABLED=1`
- table region gercekten olusuyor mu
- `auto` moddaysan arka backendlerden biri hazir mi

## Not

Bu belge "bugun neyi nasil test ederiz" sorusunun cevabidir. Tarihsel test sonuclari ve karar gunlugu icin `DEVLOG.md` dosyasina bak.

