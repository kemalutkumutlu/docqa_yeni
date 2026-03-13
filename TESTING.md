# TESTING

Bu belge, repo icin kullanilan dogrulama akisini ozetler. Onerilen mantik: uygulamayi acmadan once preflight, sonra LLM-free gate'ler, sonra Gemini tabanli kabul testleri.

## Onerilen Sira

### 0. Preflight

`run.sh` bunu zaten otomatik calistirir. Tek basina da kosabilirsin:

```bash
.venv-gpu/bin/python scripts/preflight.py
```

Preflight su alanlari dogrular:

- auth / Gemini client kurulumu
- secili generation modeli
- embedding cagrisi
- VLM extraction cagrisi
- Vertex kullaniliyorsa secilen location bilgisi

Beklenen:

- `mode=vertex` veya `mode=ai_studio`
- `[OK] generation model=... location=...`
- `[OK] embedding ...`
- `[OK] vlm ...`
- `[OK] preflight passed`

### 1. LLM-free baseline gate

```bash
.venv-gpu/bin/python scripts/baseline_gate.py
.venv-gpu/bin/python scripts/lang_gate.py
```

Amac:

- syntax/import regresyonu
- ingestion -> structure -> indexing -> retrieval akisi
- section-list subtree fetch
- coklu belge izolasyonu
- dil secimi

Bu iki komut, "uygulama kirik mi degil mi" sinyalini en hizli veren kapilar.

### 2. Retrieval kalitesi

```bash
.venv-gpu/bin/python scripts/eval_retrieval.py --pdf test_data/Case_Study_20260205.pdf
```

Olculen basliklar:

- intent accuracy
- heading / section hit
- evidence recall
- latency

### 3. Gemini kabul kapisi

```bash
.venv-gpu/bin/python scripts/eval_case_study.py --pdf test_data/Case_Study_20260205.pdf
.venv-gpu/bin/python scripts/hallucination_test.py --pdf test_data/Case_Study_20260205.pdf
```

Bu adimlar icin Gemini auth gerekir. Hem Vertex hem AI Studio profili ile calisabilir.

Kontrol edilenler:

- citation zorunlulugu
- coverage
- negatif sorularda "Belgede bu bilgi bulunamadi."
- hallucination guard

### 4. Klasor ve smoke suite

Birden fazla PDF ile hizli scan:

```bash
.venv-gpu/bin/python scripts/folder_suite.py --dir test_data --mode retrieval --isolate 1
.venv-gpu/bin/python scripts/folder_suite.py --dir test_data --mode ask --isolate 1 --max_pdfs 3
```

Genel smoke icin:

```bash
.venv-gpu/bin/python scripts/smoke_suite.py test_data/Case_Study_20260205.pdf
```

### 5. Multimodal mode sanity check

`multimodal` mode icin minimum kontrol:

```ini
DOC_PROCESSING_MODE=multimodal
EMBEDDING_MODEL=gemini-embedding-2-preview
```

Kontrol:

1. `./run.sh`
2. UI runtime ayarinda `Processing Mode=multimodal` gorunuyor mu
3. Belgeyi yeniden yukle
4. Karmaşık sayfa / tablo / gorsel iceren bir soru sor

Beklenen:

- belge yukleme classic moda gore daha yavas olabilir
- ayni belge yeni mode ile tekrar yuklenince cache yeniden kullanilmaz
- retrieval/generation classic akisi bozmadan calismaya devam eder
- Gemini generation yolunda visual evidence varsa cevap kalitesi artabilir

## Demo Oncesi Checklist

Mülakat ya da canli demo oncesi minimum kontrol:

1. `./run.sh`
2. UI acildiktan sonra `Case_Study_20260205.pdf` yukle
3. Su sorulari sor:
   - `Fonksiyonel gereksinimler nelerdir?`
   - `Teslimatlar nelerdir?`
   - `Teslim suresi nedir?`
   - `Projenin amaci nedir?`
   - `Araba kac beygir?`

Beklenen davranis:

- ilk iki soru `section_list`
- ortadakiler normal QA
- sonuncusu negatif guard
- cevaplarda citation olsun

Ek multimodal kontrol:

- runtime settings'te `Processing Mode`
- `Embedding Model=gemini-embedding-2-preview`
- belge yeniden yuklendikten sonra soru-cevap halen calisiyor mu

## Tavsiye Edilen Calistirma Sekli

Tek launch yolu:

```bash
./run.sh
```

Neden:

- `.env` yuklenir
- `preflight.py` otomatik kosar
- ayni Python runtime kullanilir
- demo sirasinda shell/env farki kaynakli surpriz azalir

## Sik Karsilasilan Problemler

### Preflight generation fail

Kontrol et:

- `GEMINI_MODEL`
- `GEMINI_FALLBACK_MODEL`
- Vertex kullaniyorsan `VERTEX_LOCATION`
- fallback model kullaniyorsan `VERTEX_FALLBACK_LOCATION`
- auth dosyasi / API key dogru mu

### Embedding fail

Kontrol et:

- `EMBEDDING_MODEL`
- Gemini embedding kullaniyorsan auth aktif mi
- lokal embedding kullaniyorsan torch/sentence-transformers kurulmus mu

### VLM fail

Kontrol et:

- `VLM_PROVIDER`
- `VLM_MODE`
- image/PDF extraction auth'i

### UI aciliyor ama PDF indexlenmiyor

Kontrol et:

- extraction bos mu
- `VLM_MAX_PAGES`
- `OCR_ENABLED`
- scan PDF ise Tesseract gerekli mi
- `DOC_PROCESSING_MODE=multimodal` ise belgeyi mode degisimi sonrasinda yeniden yukledin mi

## Not

Tarihsel ayrintili test sonuclari repo gecmisinde ve onceki bu dosya surumlerinde bulunabilir. Bu belge bilincli olarak mevcut, uygulanabilir test akisini ozetler.
