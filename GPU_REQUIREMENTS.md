# GPU_REQUIREMENTS

Bu repo icin GPU kullanimi, secilen profile gore degisir. Guncel varsayilan profil Gemini tabanli uzaktaki servisleri kullandigi icin, lokal GPU her adimda etkili degildir.

## Kisa Ozet

- `Gemini generation`: lokal GPU kullanmaz
- `Gemini embedding`: lokal GPU kullanmaz
- `Gemini VLM`: lokal GPU kullanmaz
- `sentence-transformers` lokal embedding: lokal GPU kullanabilir
- `Ollama` local LLM/VLM: kendi surecinde lokal GPU kullanabilir

Yani bugunku demo profilinde RTX 4090 zorunlu degildir; ancak lokal embedding veya local Ollama yoluna gecersen fark yaratir.

## GPU Ne Zaman Fayda Saglar?

### 1. Lokal embedding

`EMBEDDING_MODEL=auto` veya sabit bir `sentence-transformers` modeli secersen:

- ilk index olusturmada hizlanma
- query embedding sirasinda hizlanma

Ornek:

```ini
EMBEDDING_MODEL=auto
EMBEDDING_DEVICE=cuda
```

veya

```ini
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DEVICE=cuda
```

### 2. Lokal Ollama

```ini
LLM_PROVIDER=local
VLM_PROVIDER=local
```

Bu durumda Ollama'nin sectigin modeline gore VRAM tuketimi onemli hale gelir.

## Onerilen Kurulum

GPU ortamini ayri sanal ortamda tut:

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -U pip
```

CUDA uyumlu PyTorch kur. Ornek `cu121`:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Sonra proje bagimliliklarini kur:

```bash
python -m pip install -r requirements.txt
```

## Dogrulama

### Torch GPU goruyor mu?

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Lokal embedding gercekten GPU'da mi?

Bu test yalnizca lokal embedding icin anlamlidir:

```bash
python -c "from src.config import load_settings; from src.core.embedding import Embedder; s = load_settings(); print('selected', s.embedding_model, s.embedding_device); e = Embedder(s.embedding_model, device=s.embedding_device); e.embed_query('test'); print('device:', getattr(getattr(e, '_model', None), 'device', 'remote-or-n/a'))"
```

Not:

- `gemini-embedding-001` kullanirken bu yol lokal `torch` device raporlamaz; embedding uzaktan gelir.

## VRAM Notu

Local Ollama kullaniyorsan:

- daha buyuk model daha fazla VRAM ister
- uzun context KV cache'i buyutur
- VRAM yetmezse CPU offload olabilir
- bu da latency'yi belirgin artirir

Pratikte:

- 7B quantize modeller guvenli baslangictir
- demo amaci offline gostermekse orta boy model secmek daha mantiklidir

## Troubleshooting

### CUDA gorunuyor ama hizlanma yok

Muhtemel neden: Gemini tabanli uzak embedding kullaniyorsun. Bu beklenen davranistir.

### `device: cpu` gorunuyor

Kontrol et:

- `EMBEDDING_MODEL` lokal mi
- `EMBEDDING_DEVICE=cuda` mi
- dogru venv aktif mi
- CUDA uyumlu torch kurulmus mu

### App'i hangi Python ile acayim?

Tek onerilen yol:

```bash
./run.sh
```

Bu script `.venv-gpu/bin/python` kullanir ve `preflight.py` ile acilis dogrulamasini yapar.
