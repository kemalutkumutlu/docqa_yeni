# GPU_REQUIREMENTS

Bu belge, projede GPU'nun hangi parcalarda gercekten fayda sagladigini ozetler. Kritik nokta: bu sistemde her sey GPU ile hizlanmaz.

## Kisa Ozet

- `Gemini generation`: lokal GPU kullanmaz
- `Gemini embedding`: lokal GPU kullanmaz
- `Document AI OCR / Layout / Table`: lokal GPU kullanmaz
- local `sentence-transformers` embedding: lokal GPU kullanabilir
- `PaddleOCR-VL-1.5` / `PaddleOCR`: lokal GPU kullanabilir
- `Docling layout detector`: lokal GPU kullanabilir
- `Ollama` local LLM/VLM: kendi surecinde lokal GPU kullanabilir

Yani GPU'nun asil etkili oldugu local kisimlar:

- embedding
- Paddle OCR
- Docling layout
- Ollama

## GPU Ne Zaman Fayda Saglar?

### 1. Local embedding

```ini
EMBEDDING_MODEL=auto
EMBEDDING_DEVICE=cuda
```

veya:

```ini
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DEVICE=cuda
```

Kazanc:

- ilk index olusturma
- sorgu embedding latency

### 2. Paddle OCR

```ini
OCR_ENABLED=1
OCR_BACKEND=paddle_vl
OCR_DEVICE=cuda
```

veya:

```ini
OCR_ENABLED=1
OCR_BACKEND=paddle
OCR_DEVICE=cuda
```

Kazanc:

- scan/image belgelerde OCR hizi
- PaddleOCR-VL-1.5 inference

Not:

- environment kirli ise Paddle import zinciri torch/CUDA tarafinda kirilabilir
- bu durumda ayri OCR venv kullanmak daha stabil olur

### 3. Docling layout detector

```ini
VISUAL_REGION_SOURCE=detector
VISUAL_DETECTOR_BACKEND=docling
DOCLING_DEVICE=cuda
```

Kazanc:

- bbox/layout detection latency
- table/block detection kalitesi icin lokal hiz kazanimi

Onemli not:

- `docling` icin ayri bir venv kullanmak tavsiye edilir
- `DOCLING_PYTHON_BIN` ile projeye baglanabilir

### 4. Ollama local LLM/VLM

```ini
LLM_PROVIDER=local
VLM_PROVIDER=local
```

Bu durumda GPU:

- model inference hizina
- ilk token gecikmesine
- vision extraction surelerine

dogrudan etki eder.

## Onerilen Ortam Stratejisi

Bu projede tek venv'e her seyi yigmak yerine ayri ortam tutmak daha guvenli:

- `.venv-gpu`: ana uygulama
- ayri `docling` venv: layout detector
- gerekirse ayri `paddle` venv: OCR

Bunun sebebi:

- torch / CUDA / paddle / docling zincirleri birbirini bozabilir
- subprocess ile ayrik venv kullanmak ana uygulamayi daha kararlı tutar

## Ana Ortam Kurulumu

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Local OCR/layout bagimliliklari icin:

```bash
python -m pip install -r requirements-ocr-offline.txt
```

Platforma gore ayrica:

- uygun `torch` build'i
- uygun `paddlepaddle` veya `paddlepaddle-gpu`

kurulmalidir.

## Dogrulama

### Torch GPU goruyor mu?

```bash
.venv-gpu/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
PY
```

### Local embedding gercekten GPU'da mi?

```bash
.venv-gpu/bin/python - <<'PY'
from src.config import load_settings
from src.core.embedding import Embedder
s = load_settings()
print('selected', s.embedding_model, s.embedding_device)
e = Embedder(s.embedding_model, device=s.embedding_device)
e.embed_query('test')
print('device:', getattr(getattr(e, '_model', None), 'device', 'remote-or-n/a'))
PY
```

### Paddle OCR smoke

```bash
.venv-gpu/bin/python scripts/paddle_ocr_runner.py --help
```

### Docling smoke

```bash
DOCLING_PYTHON_BIN=/abs/path/to/.venv-docling/bin/python \
.venv-gpu/bin/python scripts/inspect_regions.py test_data/Case_Study_20260205.pdf --level region --source detector --detector-backend docling
```

## VRAM Notlari

### Ollama

Pratikte:

- 7B quantize modeller guvenli baslangictir
- uzun context daha fazla VRAM ister
- vision modelleri de ek maliyet getirir

Orta seviye kartlarda:

- `qwen2.5:7b`
- `llava:7b`

makul baslangic secenekleridir.

### Paddle ve Docling

PaddleOCR-VL-1.5 ve layout detector ayni anda ayni GPU uzerinde kosacaksa:

- VRAM baskisi artar
- ayri surecler gecikmeyi ve bellek kullanimini etkileyebilir

Bu nedenle production benzeri denemelerde:

- her adimi tek tek smoke test etmek
- ayri venv / ayri surec tercih etmek

daha sagliklidir.

## Sik Sorunlar

### CUDA gorunuyor ama hizlanma yok

Muhtemel nedenler:

- Gemini / Document AI kullaniyorsun
- ilgili path remote servis oldugu icin lokal GPU etkisiz

### `device: cpu` gorunuyor

Kontrol et:

- `EMBEDDING_MODEL` lokal mi
- `EMBEDDING_DEVICE=cuda` mi
- dogru venv aktif mi
- CUDA uyumlu torch kurulu mu

### Paddle segfault veya import crash

Muhtemel neden:

- torch / CUDA preload zinciri
- opsiyonel bagimlilik eksigi
- opencv / libgomp uyumsuzlugu

Pratik cozum:

- ayri temiz OCR venv
- subprocess ile izolasyon

### App'i hangi Python ile acmaliyim?

```bash
./run.sh
```

Bu script:

- `.venv-gpu/bin/python` kullanir
- preflight yapar
- shell/env farklarini azaltir

