# DEVLOG

## Faz 10.0 — Multimodal Mode (MVP)

- Amaç: mevcut `classic` text-first akisi bozmadan, UI'dan secilebilen ayri bir `multimodal` mode eklemek.
- Yeni runtime ayari:
  - `DOC_PROCESSING_MODE=classic|multimodal`
  - Chainlit settings panelinde `Processing Mode`
- MVP kapsam:
  - PDF / image sayfalari icin page-level visual asset uretimi
  - `visual` chunk tipi
  - metadata'da `modality` ve `image_path`
  - `gemini-embedding-2-preview` ile visual chunk icin image+text embedding denemesi
  - visual evidence varsa Gemini generation prompt'una secili goruntulerin eklenmesi
- Bilincli sinir:
  - region/table crop seviyesinde degil, page-level
  - Graph RAG yok
  - Agentic RAG yok
  - mode degisikligi mevcut yuklu belgeleri geriye donuk cevirmez; belge yeniden yuklenmelidir

Bu dosya, projenin gelistirme surecini kronolojik olarak belgelemektedir.

## Faz 0 — Proje Iskeleti
- Proje yapisi olusturuldu: `app.py`, `src/config.py`, `requirements.txt`, `.env.example`
- Chainlit UI entrypoint (placeholder), config parsing, dependency listesi
- Karar: Chainlit secildi (chat-first UX, dosya yukleme desteği)

## Faz 1 — Belge Okuma (Ingestion)
- `src/core/ingestion.py`: PDF + image okuma pipeline'i
- PyMuPDF ile PDF text-layer cikarimi, text yetersizse pytesseract OCR fallback
- Turkce+Ingilizce OCR (`tur+eng` lang)
- (Opsiyonel) Gemini VLM ile **extract-only** cikarim (layout/tablolar icin)
- **Dual-quality secim**: PDF/OCR/VLM adaylari arasinda baslik/structure korunumu daha iyi olani secilir (VLM force aciksa bile regresyon onlenir)
- `scripts/extract_text.py` ile test edildi
- **Sorun**: Windows konsolda UnicodeEncodeError → `sys.stdout.reconfigure(encoding="utf-8")` ile cozuldu
- **Sorun**: `ModuleNotFoundError` script'lerden → `sys.path.insert` fallback eklendi

## Faz 2 — Yapisal Temsil (Structure + Chunking)
- `src/core/structure.py`: Dokumandan bagimsiz heading detection
  - Regex ile numarali basliklar (orn. "2.", "4.1", "A.4.1") algilama
  - Stack-based section tree olusturma
  - Boilerplate (header/footer) temizleme heuristic'i
- Parent-child chunking: her section icin parent (tam bolum) + child (kucuk, overlap) chunk'lar
- Metadata: `section_id`, `parent_id`, `heading_path`, `page_start`, `page_end`, `kind`
- `scripts/preview_structure.py` ile gorsel test

## Faz 3 — Indeksleme (Hybrid Retrieval)
- `src/core/embedding.py`: SentenceTransformer (`multilingual-e5-small`) wrapper
- `src/core/vectorstore.py`: ChromaDB persistent store (upsert, query, get)
- `src/core/sparse.py`: BM25 sparse index (rank-bm25), TR/EN friendly tokenizer
- `src/core/hybrid.py`: RRF (Reciprocal Rank Fusion) — dense + sparse sonuclarini birlestirme
- `src/core/indexing.py`: `LocalIndex` sinifi — Chroma + BM25 tek catida
- `scripts/build_index.py` ve `scripts/search_index.py` ile dogrulandir

## Faz 3.1 — Coklu Belge Izolasyonu (Stale Delete + doc_id Filter)
- Persisted Chroma kullaniminda eski chunk'larin karismasini onlemek icin doc_id bazli filtreleme ve stale temizleme eklendi
- Aktif belge politikasi: birden fazla dokumanda varsayilan hedef son yuklenen belge; `/use <dosya>` ile secilebilir

## Faz 4 — Query Routing + Complete Section Fetch + Coverage
- `src/core/retrieval.py`: Tam retrieval pipeline'i
  - **Query Classification**: Rule-based intent siniflandirici (section_list vs normal_qa)
    - TR/EN pattern'lar: "nelerdir", "listele", "what are the", "list all" vb.
  - **Heading-aware section matching**: Hybrid sonuclarindan en iyi section secimi
    - Heading-query token overlap bonus'u ile "teslimatlar nelerdir" → dogru bolumu bulur
  - **Complete section + subtree fetch**: section_list intent'te sadece top-k degil, tum bolum + alt bolumler
    - ChromaDB'den `section_id` ve `parent_id` filtreleriyle recursive fetch
  - **Coverage counting**: Parent chunk text'inde madde/satir sayisi heuristic'i
    - Numarali liste, bullet, tablo satiri, kisa etiket satirlari
  - **Sorun**: ChromaDB multi-field where → `$and` operatoru gerektiriyor, duzeltildi
  - **Test sonuclari**:
    - "fonksiyonel gereksinimler nelerdir" → section_list, section=2, coverage=6 (dogru)
    - "teslimatlar nelerdir" → section_list, section=4, evidences=6 (alt bolumler dahil)
    - "teslim suresi nedir" → normal_qa (dogru)
    - "projenin amaci nedir" → normal_qa, section=1 (dogru)

## Faz 5 — LLM Generation + Guardrails
- `src/core/generation.py`: Gemini API ile cevap uretimi
  - **System prompt**: Strict kurallar
    1. SADECE baglam'daki bilgiyi kullan
    2. Baglamda yoksa "Belgede bu bilgi bulunamadi."
    3. Her cumle sonuna citation: [DosyaAdi - Sayfa X]
    4. Turkce cevap (kullanici Ingilizce sorarsa Ingilizce)
  - **Section-list modu**: LLM'e "beklenen madde sayisi" bildirilir
  - **Coverage post-validation**: LLM cevabindaki madde sayisi vs beklenen → uyari
  - (Kalite) Citation/coverage eksikliginde 1 kez otomatik retry
  - **Context builder**: Parent chunk'lar tercih edilir, child duplicate onlenir
- `src/core/pipeline.py`: Tum adimlari birlestiren `RAGPipeline` sinifi
  - `add_document()`: ingest → structure → chunk → index
  - `ask()`: retrieve → generate
  - Multi-document desteqi (session icinde birden fazla belge)
- `google-genai` SDK eklendi requirements.txt'ye

## Faz 5.1 — Deterministic Section-List + Dayaniklilik (100/100 hedefi)
- **Amaç**: Liste/bölüm çıkarma sorularında eksik madde problemini düşürmek, halüsinasyonu azaltmak ve geçici Gemini API hatalarına dayanıklılık kazanmak.
- `src/core/generation.py`:
  - **Deterministic section_list rendering (doc-agnostic)**:
    - `section_list` intent'te uygun olduğunda LLM'e gitmeden, ilgili **parent section** metninden madde çıkarımı yapar.
    - Çıkarım türleri: numaralı/bullet listeler, indeksli tablo satırları, label/description çiftleri, alt başlıklar.
  - **Güvenlik kilidi (regresyon önleme)**:
    - Deterministic çıkarım sadece “yüksek güven” sinyali olduğunda devreye girer; aksi halde LLM yoluna **fallback** eder.
    - `coverage.expected_items` varsa ve `len(items) < expected` ise deterministic sonuç iptal edilir (eksik liste riski).
    - Önceki “her satırı listele” tarzı gürültülü fallback kaldırıldı (yanlış/çöp madde üretimini azaltır).
  - **Tablo satırı toparlama**:
    - `Label: açıklama` şeklindeki tablolarda açıklama birden fazla satıra bölünmüşse, sonraki label başlayana kadar satırlar birleştirilir (eksik içerik düşer).
    - Olası tablo header çiftleri (örn. `ColumnA: ColumnB`) yapısal olarak elenir (içerik/kelimeye özel kural yok).
  - **Gemini retry + backoff**:
    - `503/UNAVAILABLE`, `429/RESOURCE_EXHAUSTED`, timeout gibi geçici hatalarda otomatik tekrar deneme + artan bekleme eklendi.
    - Başarılı çağrılarda davranış değişmez; sadece geçici hata durumunda stabilite artar.
- `scripts/eval_case_study.py`:
  - Case Study için **katı kabul kapısı** (eval script'i dokümana özeldir; ürün mantığı doküman-agnostic kalır).
  - `section_list` ve `normal_qa` için intent/citation/eksik madde kontrolleri yapar; geçmezse exit code=1.

## Faz 6 — Chainlit UI
- `app.py` tam pipeline'a baglandi:
  - `on_chat_start`: uygulama acilir acilmaz mesaj yazilabilir (upload modal zorunlu degil)
  - Belge yukleme: drag&drop / paperclip ile PDF/PNG/JPG
  - Dosya isleme: async wrapper ile UI donmaz
  - `on_message`: pipeline.ask() ile soru-cevap
  - Debug bilgisi: collapsible section (intent, citation sayisi, coverage durumu)
  - Hata kontrolu: `.env` eksikse uyari, belge yuklenmemisse uyari
  - Modlar: `/chat` (sohbet), `/doc` (belge), `/use <dosya>` (aktif belge)
  - Dogal dil komutlari: “sohbet moduna gec”, “belge moduna nasil donecem” gibi istekler otomatik algilanir
- `chainlit.md`: Acilis ekrani metni

## Faz 6.1 — Dev UX: Port cakismasi ve otomatik cikis
- **Sorun**: Windows'ta `chainlit run app.py -w` sonrasi tab kapatilinca process arkaplanda kalabiliyor → tekrar baslatmada `Errno 10048` (port 8000 in use).
- **Cozum (opsiyonel, bozmadan)**:
  - `app.py` icine `on_chat_end` hook'u ile “son client disconnect olunca grace sure sonra exit” eklendi.
  - `.env` / `.env.example`:
    - `AUTO_EXIT_ON_NO_CLIENTS=1`
    - `AUTO_EXIT_GRACE_SECONDS=8`
  - Davranis sadece env ile acilinca aktif; default etkisiz.

## Faz 6.2 — GPU notu (embedding hizlandirma)
- GPU, bu projede **yalnizca embedding** tarafinda etkilidir (SentenceTransformers / PyTorch).
- Gemini/OpenAI LLM API (ve Gemini VLM API) uzak servis oldugu icin GPU ile hizlanmaz.
- README'ye “GPU notu + runtime device dogrulama” komutu eklendi.

## Faz 6.3 — Klasor Test Suite + JSONL Loglama (opsiyonel, bozmadan)
- Amaç: Birden fazla PDF ile daha iyi saha testi ve “neden bu cevap geldi?” sorusuna kanitli debug.
- `scripts/folder_suite.py` eklendi:
  - `test_data/` altindaki PDF'leri toplu indeksler ve sorgulari koşturur.
  - `--isolate 1` ile her PDF’i ayri indeksleyerek (coklu PDF’de) O(N^2) re-embedding riskini azaltir.
  - `--max_pdfs N` ile hizli sanity check yapilabilir.
- `src/core/eventlog.py` eklendi (JSONL event logger).
- `src/core/pipeline.py` icine **env ile acilan** log entegrasyonu eklendi (default OFF):
  - `RAG_LOG=1` olursa her `ask()` icin `query/answer/intent/evidence_count/citation/coverage` bilgileri loglanir.
  - Loglar: `data/logs/rag_<YYYYMMDD>_session_<id>.jsonl` ve `data/logs/by_doc/<dosya>.jsonl`
  - Gizlilik: `RAG_LOG_CONTEXT_PREVIEW=0` default (baglam preview kapali).
- Dokumantasyon guncellendi: `.env.example`, `README.md`, `TESTING.md`.

## Faz 6.4 — Dayaniklilik: PDF kalitesi dusuk text-layer + izolasyon kenar durumlari
- `src/core/ingestion.py`:
  - OCR tetikleyicisi iyilestirildi: sadece “metin cok kisa” degil, **low-quality text-layer** durumunda da OCR/VLM adayi olur.
- `src/core/indexing.py`:
  - Doc filter kenar durumu: Caller doc_ids verip filtre bos kalirsa **unrestricted search'e dusmez**, bos sonuc doner (cross-doc leak riski azalir).
- `src/core/generation.py`:
  - TR/EN dil secimi icin hafif heuristik + sistem prompt addendum eklendi (LLM-free gate ile dogrulandi).
- `scripts/baseline_gate.py`:
  - Bos/scan-like PDF'lerin crash etmemesi ve mixed-language retrieval icin ek kapilar eklendi.

## Faz 6.5 — Gorsel Ingestion Kalite Iyilestirmeleri (JPG/PNG)
- Amaç: PDF’teki “stabil metin + yapi korunumu” kalitesini gorsellere yaklastirmak (text-layer olmadigi icin OCR/VLM kritik).
- `src/core/ingestion.py` (image path):
  - **EXIF yon duzeltme**: Telefon fotosu / tarama goruntulerinde orientation normalize edilir.
  - **Kontrollu upscale**: Kucuk goruntuler OCR icin buyutulur (latency patlamasini onlemek icin boyut cap’i var).
  - **Preprocess varyantlari (Pillow-only)**: grayscale+autocontrast, hafif sharpen ve konservatif threshold denenir.
  - **Multi-pass OCR**: Birden fazla varyant OCR edilir; cikan metinler arasindan yapi/heading korunumu daha iyi olani secilir.
  - **VLM ↔ OCR ortak aday secimi**: VLM aciksa (Gemini extract-only) VLM cikarimi da aday listesine girer; yine en-iyi-secim uygulanir.
- `.env` / config:
  - `TESSERACT_CONFIG` eklendi (opsiyonel): pytesseract `config=` passthrough (orn. `--psm 6 --oem 3`). Default bos → davranis degismez.
- Dokumantasyon: `README.md`, `chainlit.md`, `.env.example` guncellendi.

## Faz 6.6 — Ayni Dokumanin Tekrar Yuklenmesi (Skip Reprocess)
- Amaç: Ayni dosya ayni ayarlarla tekrar yuklendiginde gereksiz OCR/VLM + embedding maliyetini onlemek.
- `src/core/pipeline.py`:
  - Ayni oturumda ayni dosya (sha256 `doc_id`) ve ayni ingestion/embedding ayarlari tespit edilirse dokuman **yeniden islenmez**; sadece aktif dokuman olarak set edilir.
  - Dosya degisirse veya OCR/VLM/embedding ayarlari degisirse yeniden islenir.
- `src/core/indexing.py` + `src/core/sparse.py`:
  - Yeniden indeksleme durumunda BM25 (sparse) tarafinda duplicate birikmemesi icin ilgili `doc_id` girdileri temizlenir.

## Faz 7 — Kalite ve Operasyon Iyilestirmeleri

### 7.1 — CI / GitHub Actions
- `.github/workflows/ci.yml`: Her push/PR'da `baseline_gate.py` + `lang_gate.py` otomatik calisir.
- Opsiyonel: `GEMINI_API_KEY` secret tanimlanmissa `eval_case_study.py` ayrı job olarak kosturulur.

### 7.2 — Mini Eval Set + Retrieval Metrikleri
- `test_data/eval_questions.json`: 25 soruluk eval set (section_list + normal_qa + TR/EN + negatif)
- `scripts/eval_retrieval.py`: LLM-free retrieval kalite olcumu (intent accuracy, section hit, heading hit, evidence recall, avg latency)

### 7.3 — Incremental Indexing
- `src/core/sparse.py`: `BM25Index.extend()` — yeni chunk'lari mevcut BM25 index'ine ekler (full rebuild yerine)
- `src/core/indexing.py`: `LocalIndex.add_chunks()` — sadece yeni belgein chunk'larini embed eder
- `src/core/pipeline.py`: `add_document()` artik incremental path kullaniyor (onceki belgeler tekrar embed edilmiyor)

### 7.4 — BM25 Kaliciligi (Persist)
- `src/core/sparse.py`: `save()` / `load()` metodlari eklendi — BM25 state pickle ile diske kaydedilir
- `src/core/indexing.py`: `build()` ve `add_chunks()` sonrasi otomatik BM25 persist (`data/chroma/bm25_index.pkl`)
- ChromaDB + BM25 birlikte kalici: restart sonrasi her iki index hazir

### 7.5 — LLM-Free Extractive QA (Local Mode)
- `src/core/generation.py`: `generate_extractive_answer()` — LLM olmadan belgeden dogrudan alinti + citation
- `src/core/pipeline.py`: `llm_provider` field + `ask()` icinde extractive branching
- `app.py`: `LLM_PROVIDER=none` iken extractive mod hoşgeldin mesaji (onceki hata yerine)
- Artik disariya hicbir API cagrisi yapilmadan belge sorusu cevaplanabiliyor

### 7.6 — Observability / Telemetry
- `src/core/pipeline.py`: Index build suresi (`index_time_ms`), retrieval suresi (`retrieval_ms`), generation suresi (`generation_ms`) event log'a eklendi
- `RAG_LOG=1` ile tum metrikler JSONL log'da gorunur

### 7.7 — Dokumantasyon Guncellemeleri
- `README.md`: CI badge, yeni ozellik tablosu, LLM_PROVIDER=none kullanimi, eval_retrieval komutu, proje agaci, tasarim kararlari, Gelecek Plan (Roadmap)
- `TESTING.md`: eval_retrieval, CI, incremental indexing, extractive QA, observability test bolumleri
- `DEVLOG.md`: Alternatif degerlendirmeler, zorluklar, zaman dagilimi, retrospektif

## Alternatif Degerlendirmeler ve Tasarim Kararlari

Bu bolum her fazda degerlendirilen alternatifleri ve neden mevcut yolu sectigimizi belgelemektedir.

### Faz 1 — Ingestion Alternatifleri
- **pdfplumber vs PyMuPDF**: pdfplumber tablo cikariminda daha iyi, ancak PyMuPDF hem daha hizli hem de sayfa gorseli (pixmap) destegiyle VLM pipeline'ina daha uygun. PyMuPDF secildi.
- **EasyOCR vs Tesseract**: EasyOCR GPU ile hizli ve TR destegi iyi, ancak dependency boyutu cok buyuk (~1.5 GB). Tesseract hafif ve yaygin; `tur+eng` langpack ile yeterli. Tesseract secildi.
- **VLM-only ingestion**: Tum sayfalari Gemini VLM ile isleme fikri dusunuldu. Sorunlar: maliyet (sayfa basi API call), latency, ve availability riski. Bunun yerine **dual-quality** yaklasimi secildi: VLM adayi da uretilir ama en iyi kalite otomatik secilir.

### Faz 2 — Chunking Alternatifleri
- **Naif fixed-size chunking vs Hierarchical**: Fixed-size daha basit ama "X nelerdir?" gibi sorularda alt maddeler kayboluyordu. Heading-based parent/child yapiyla tum bolum getirilmesi saglanip bu sorun tasarimsal olarak cozuldu.
- **LLM-based heading detection**: Baslik tespiti icin LLM kullanma fikri vardi. Ancak regex-based deterministic tespit hem LLM maliyetinden kacinir hem de tekrar uretilabilir sonuc verir. Regex secildi.

### Faz 3 — Retrieval Alternatifleri
- **Sadece Dense vs Hybrid**: Sadece vector search, keyword-specific sorgularda basarisiz (orn. "DEVLOG.md" aramasi). BM25 eklenerek RRF fusion ile her iki avantaj birlestirildi.
- **Pinecone/Weaviate vs ChromaDB**: Managed vector DB'ler production'da daha iyi, ancak MVP icin lokal persistence + sifir konfigrasyon avantaji ChromaDB'yi ideal kildi.
- **FAISS vs ChromaDB**: FAISS daha performansli ama metadata filtreleme (doc_id, section_id) icin ek katman gerektirir. ChromaDB bunu native destekler.

### Faz 4-5 — Generation Alternatifleri
- **LLM-only section list vs Deterministic rendering**: LLM bazen madde atliyordu. Deterministic rendering ile LLM'e gitmeden parent chunk text'inden madde cikarimi yapilarak %100 coverage saglandi. LLM yolu sadece deterministic cikarilamadiginda fallback olarak kullaniliyor.
- **OpenAI vs Gemini**: Her iki API de destekleniyor, ancak Gemini 2.0 Flash'in TR performansi, fiyat/performans orani ve multimodal yeteneginden dolayi varsayilan olarak Gemini secildi. Ayrica Google AI Studio **300$ free credit** verdigi icin (case study suresince) pratik bir tercih oldu.

## Zorluklar ve Cozumler

| Zorluk | Etki | Cozum |
|--------|------|-------|
| Windows Unicode konsol hatasi | Script'ler crash | `sys.stdout.reconfigure(encoding="utf-8")` |
| ChromaDB `$and` operatoru gerekliligi | Multi-field where calismiyor | `$and`/`$or` syntax'ine gecis |
| PDF tablo cikarimi (non-standard layout) | Maddeler kayip | Indexed-table + label/desc pair extraction heuristic'leri |
| Gemini 503/429 gecici hatalar | Eval script'ler fail | Exponential backoff retry (4 deneme) |
| Port 8000 cakismasi (Windows) | Gelistirme akisi bozuluyor | `AUTO_EXIT_ON_NO_CLIENTS` opsiyonel hook |
| VLM heading regresyonu | VLM force modunda heading kaybi | Dual-quality secim: PDF/OCR/VLM arasinda en iyi yapi secilir |
| Tekrarlanan numarali basliklar (section_id cakismasi) | Index build crash (Chroma DuplicateIDError) | Section ID'leri dokuman icinde unique hale getirildi (suffix) |

## Zaman Dagilimi (Tahmini)

| Faz | Tahmini Sure | Notlar |
|-----|-------------|--------|
| Faz 0 — Proje Iskeleti | ~1 saat | Chainlit + config + proje yapisi |
| Faz 1 — Ingestion | ~4 saat | PDF parsing + OCR + VLM + dual-quality |
| Faz 2 — Structure + Chunking | ~3 saat | Heading detection + section tree + parent-child |
| Faz 3 — Indexing (Hybrid) | ~4 saat | ChromaDB + BM25 + RRF + multi-doc izolasyon |
| Faz 4 — Query Routing + Coverage | ~4 saat | Intent classification + subtree fetch + coverage heuristic |
| Faz 5 — Generation + Guardrails | ~5 saat | System prompt + deterministic section-list + retry |
| Faz 6 — UI + Test Altyapisi | ~5 saat | Chainlit entegrasyon + baseline_gate + eval + loglama |
| Debug / Edge Case / Dokumantasyon | ~4 saat | Windows sorunlari, kenar durumlari, README/TESTING |
| **Toplam** | **~30 saat** | |

## Bastan Baslasam Neyi Farkli Yapardim

1. **Chunking stratejisini en basta karar verirdim.** Ilk versiyonda naif chunking deneyip sonra hierarchical'a gecmek zaman kaybetti. Section-tree yaklasiminin faydasini erkenden fark ederdim.

2. **Eval set'i Faz 1'den itibaren olustururdum.** Kalite olcumu en sonda yapildi; erkenden 10-15 soru hazirlamak her degisikligin etkisini gorunur kilardi.

3. **BM25 persistence'i baslatirdim.** In-memory BM25 her restart'ta rebuild oluyor. ChromaDB gibi persist eden bir BM25 wrapper'i MVP asamasinda eklemezdim ama mimaride ongorurddum.

4. **VLM'i daha gec eklerdim.** VLM dual-quality secimi onemli bir ozellik ama MVP icin gerekliydi mi tartismali. OCR + PDF text-layer cogu senaryo icin yeterliydi; VLM'i "Faz 2" yerine "Faz 7 / opsiyonel" olarak konumlandirirdim.

5. **Daha fazla unit test yazardim.** Mevcut test altyapisi (baseline_gate, eval_case_study) entegrasyon seviyesinde cok iyi, ancak birim testleri (orn. heading detection, coverage counting) ayri olsaydi refactoring'ler daha guvenli olurdu.

## Teknik Borc ve Sinirlamalar

- ~~**BM25 in-memory**: Her restart'ta rebuild.~~ → **COZULDU** (Faz 7.4): BM25 artik `data/chroma/bm25_index.pkl` olarak diske kaydediliyor.
- ~~**Tam rebuild indexing**: Yeni dokuman eklendiginde tum embedding'ler yeniden hesaplaniyor.~~ → **COZULDU** (Faz 7.3): Incremental upsert ile sadece yeni chunk'lar embed ediliyor.
- ~~**LLM bagimliligi**: normal_qa intent'te LLM olmadan cevap verilemiyor.~~ → **COZULDU** (Faz 7.5): `LLM_PROVIDER=none` ile extractive QA modu eklendi.
- **Tek embedding model**: Model degistiginde tum index rebuild gerekiyor. Model versiyonlama/migration mekanizmasi yok.
- **OCR kalite siniri**: Dusuk cozunurluklu tarama PDF'lerde OCR kalitesi degisken. Post-processing (spell check / denoise) yok.

## Faz 8 — Kapsamli Test Kosusu ve Halusinasyon Testi (2026-02-11)

### 8.1 — Halusinasyon Test Scripti
- `scripts/hallucination_test.py` eklendi:
  - 10 pozitif (belgede var) + 15 negatif (belgede yok) soruluk kapsamli test
  - Olcumlenen metrikler: halusinasyon orani, yanlis negatif orani, citation uyumu, anahtar kelime isabeti, gecikme
  - Sonuc: Pozitif 10/10 (100%), Negatif 14/15 (93%), Citation 100%
  - 1 edge case: "sunucu gereksinimleri nelerdir" → keyword overlap nedeniyle "Fonksiyonel Gereksinimler" bolumuyle eslesti (retrieval false-positive, LLM kaynakli degil)

### 8.2 — 8 PDF ile Tam Kosu
- `test_data/` altindaki 8 farkli PDF (TR/EN, akademik/teknik/CV/brosur) ile folder_suite calistirildi
- Retrieval: 32/32 basarili (8 PDF x 4 sorgu)
- Ask (Gemini): 12/12 basarili (3 PDF x 4 sorgu)
- Toplam: 118 sayfa, 192 chunk basariyla indekslendi

### 8.3 — Sayisal Retrieval Metrikleri
- 25 soruluk eval set (Case Study):
  - Intent Accuracy: 100% (25/25)
  - Heading Hit: 93% (14/15)
  - Section Hit: 83% (5/6)
  - Evidence Met: 96% (24/25)
  - Avg Retrieval Latency: 29 ms

### 8.4 — Dokumantasyon Guncellemesi
- `TESTING.md`: Tum sayisal sonuclar, halusinasyon detay tablosu, coklu belge tipi performans tablosu, metrik ozeti eklendi
- `DEVLOG.md`: Faz 8 eklendi

## Faz 9 — Retrieval Confidence Guard (Halusinasyon Duzeltmesi) (2026-02-11)

### 9.1 — Problem Analizi
- Faz 8 halusinasyon testinde "sunucu gereksinimleri nelerdir" sorgusu "Fonksiyonel Gereksinimler" bolumuyle eslesti
- Neden: BM25 + dense search "gereksinimler" keyword'unu yakaladi; deterministik section-list path LLM'i bypass ederek icerik sundu
- Koz neden: Deterministik rendering, baslik-sorgu anlam uyumunu kontrol etmiyordu

### 9.2 — Embedding Similarity Yaklasimininn Basarisizligi
- e5-small modeli cok sıkisik similarity dagilimine sahip (0.80–0.93 arasi hersey icin)
- "sunucu gereksinimleri" vs "Fonksiyonel Gereksinimler" = 0.88, "araba kac beygir" vs heading = 0.83
- Fark cok kucuk (0.05), threshold ile ayirt edilemez

### 9.3 — Lexical Topic Matching Cozumu
- `_topic_heading_relevant()` fonksiyonu eklendi (`src/core/retrieval.py`)
- Calisma prensibi:
  1. Sorgudan soru kelimeleri (nelerdir, nedir, listele vb.) cikarilir
  2. Kalan "topic" kelimeleri tespit edilir
  3. Her topic kelimesi baslik tokenlariyla prefix-tabanli karsilastirilir (min 4-5 karakter)
  4. Turk morfolojisini handle eder: "gereksinimleri" ~ "gereksinimler" (prefix "gereksinim")
  5. TUM topic kelimeleri baslikta karsilik bulmalidir; bulamazsa deterministik path atlanir
- Sonuc: "sunucu gereksinimleri" → "sunucu" baslikta YOK → LLM path'e duser → LLM "bulunamadi" der

### 9.4 — Sonuclar
- Halusinasyon orani: %7 → %0 (0/15)
- True positive: %100 (10/10) — degismedi
- Citation compliance: %100 — degismedi
- Keyword accuracy: %100 — degismedi
- Avg latency: ~3700 ms
- Baseline gate: PASSED (tum mevcut testler korundu)

### 9.5 — Borderline Test Case Duzeltmesi
- "hangi veritabani kullaniliyor" sorgusu cikarildi (Case Study teknik yaklasim bolumunde ChromaDB'den bahsediliyor → LLM bazen meşru olarak cevaplayabiliyor)
- Yerine tamamen konu disi "mars gezegeninin yuzey sicakligi kac derece" eklendi

## Faz 10 — Tamamen Yerel / Offline Mod (Ollama Entegrasyonu) (2026-02-11)

### 10.1 — Amac
- Savunma sanayii ve air-gapped ortamlar icin tum RAG pipeline'inin internet baglantisi olmadan calisabilmesi
- Mevcut mimariye **hicbir breaking change** yapmadan, yeni `LLM_PROVIDER=local` ve `VLM_PROVIDER=local` modlari eklenmesi
- Ollama uzerinde calisan yerel LLM (text) ve VLM (vision) modelleri ile Gemini API'nin birebir yerine gecmesi

### 10.2 — Teknik Degisiklikler
- **`src/core/local_llm.py`** (YENi): Ollama HTTP API istemcisi (`/api/generate`). Retry mekanizmasi, vision (base64 image) destegi
- **`src/core/vlm_extract.py`**: `VLMConfig` genisletildi: `provider` ("gemini"|"local"), `ollama_base_url`, `ollama_vlm_model`, `ollama_timeout` alanlari eklendi. `extract_text_from_image()` artik provider'a gore dispatch eder; `_extract_via_gemini()` ve `_extract_via_ollama()` ayristi. Varsayilan davranis (provider="gemini") degismedi
- **`src/core/generation.py`**: `generate_answer_local()` ve `generate_chat_answer_local()` fonksiyonlari eklendi. Ayni guardrails/prompts, citation retry, coverage retry mantigi korundu; sadece LLM cagrilari Ollama'ya yonlendirildi
- **`src/core/pipeline.py`**: `ollama_config` alani eklendi. `ask()` ve `chat()` metodlari `llm_provider == "local"` durumunu handle eder. Build fingerprint'e `vlm_provider` eklendi
- **`src/config.py`**: `LLMProvider` tipi genisletildi ("local" eklendi). `VLMProvider` tipi eklendi. `Settings`'e Ollama ayarlari eklendi (`ollama_base_url`, `ollama_llm_model`, `ollama_vlm_model`, `ollama_timeout`, `vlm_provider`)
- **`app.py`**: Local mod icin ozel karsilama mesaji. `OllamaConfig` ve `VLMConfig` provider/ollama alanlari pipeline'a aktarilir

### 10.3 — Mod Yapisi
| Mod | LLM | VLM | Embedding | Internet |
|-----|-----|-----|-----------|----------|
| `gemini` (default) | Gemini API | Gemini API | Lokal ST | Gerekli |
| `local` | Ollama LLM | Ollama VLM | Lokal ST | Gerekli degil |
| `none` | Yok | Yok | Lokal ST | Gerekli degil |

### 10.4 — Onerilen Modeller (GTX 1660 Super, 6 GB VRAM)
- Text LLM: `qwen2.5:7b` (Q4 quantize, ~4.5 GB VRAM)
- Vision VLM: `llava:7b` (Q4 quantize, ~5 GB VRAM)
- Not: Ollama modeller arasi dinamik VRAM yonetimi yapar; ayni anda ikisi yuklenmez
- Gerekce: Referans GPU dusuk/orta seviye oldugu icin (6 GB VRAM), burada hedef "en buyuk open-source modeli kosmak" degil; sistemin **tam offline** calisabildigini gosterecek pratik bir baseline secmekti.

### 10.5 — Geriye Uyumluluk
- Tum yeni alanlar varsayilan degerlere sahip; mevcut `.env` dosyalari ve scriptler **degisiklik gerektirmeden** calisir
- `LLM_PROVIDER=gemini` (veya bos) default davranisi birebir korunur
- Baseline gate ve diger LLM-free testler etkilenmez

### 10.6 — Not (UI)
- UI, Chainlit tabanli olup `.chainlit/config.toml` ve `public/` altindaki tema/static dosyalarla ozellestirilir.
- Uygulama acilisinda otomatik karsilama mesaji gonderilmez (ilk layout "jump" etkisini azaltmak icin).
- Sol tarafta (desktop) tarayici `localStorage` kullanan mini "Gecmis Sohbetler" paneli vardir (`public/history_sidebar.js`).
- Sol ust profil secicisi ile LLM provider secilebilir (Gemini/OpenAI/Local/Extractive); retrieval/indeksleme mantigi degismez.

### 10.7 — Performans Aciklamasi (Dosya Isleme Suresi) ve Tradeoff'lar
- Dokumantasyona "dosya isleme neden yuksek olabilir?" bolumu eklendi (`README.md`).
- Netlestirilen noktalar:
  - Ingestion tarafi kalite-oncelikli: OCR + (opsiyonel) VLM + dual-quality secimi, bu nedenle ilk index suresi artabilir.
  - GPU sadece embedding hizlandirir; Gemini/OpenAI LLM API (ve Gemini VLM API) uzak servis oldugu icin GPU'dan faydalanmaz.
  - `VLM_MODE=force` kaliteyi artirir ama latency/maliyet tradeoff'u vardir.
- Onerilen hiz/kalite dengesi profili dokumante edildi:
  - `VLM_MODE=auto`
  - `VLM_MAX_PAGES` dusurme (ornek: 10)
  - `EMBEDDING_DEVICE=auto`
- Mevcut hizlandirma mekanizmalari tekrar vurgulandi:
  - Ayni dosya + ayni ayar -> skip reprocess
  - Incremental indexing -> sadece yeni chunk embedding

### 10.8 — Retrieval Metrik Guncellemesi + VRAM Dokumantasyonu
- Retrieval metrikleri guncel kosuya gore revize edildi:
  - Intent: 25/25
  - Heading Hit: 15/15
  - Section Hit: 6/6
  - Evidence Met: 25/25
  - Avg Latency: 19 ms
- `README.md` ve `TESTING.md` tablolari yeni sonuclara gore guncellendi.
- `GPU_REQUIREMENTS.md` icine local LLM/VLM (Ollama) icin VRAM'in neden kritik olduguna dair notlar eklendi.

## Sonraki Adimlar
- ~~Retrieval kalitesi icin mini eval set + metrikler~~ -> TAMAM (Faz 7.2)
- ~~CI/CD pipeline (GitHub Actions)~~ -> TAMAM (Faz 7.1)
- ~~Incremental indexing (doc basi)~~ -> TAMAM (Faz 7.3)
- ~~LLM'siz extractive QA modu~~ -> TAMAM (Faz 7.5)
- ~~Observability / telemetry~~ -> TAMAM (Faz 7.6)
- ~~Uctan uca testleri farkli PDF tipleriyle genislet~~ -> TAMAM (Faz 8.2: 8 PDF, 118 sayfa)
- ~~Halusinasyon testi~~ -> TAMAM (Faz 8.1 + 9.4: 25 soru, %0 halusinasyon)
- ~~Halusinasyon duzeltmesi~~ -> TAMAM (Faz 9: topic-heading relevance guard)
- ~~Tamamen yerel/offline mod (Ollama)~~ -> TAMAM (Faz 10)
- Demo video
- Reranker (cross-encoder) degerlendirmesi (Roadmap'te)

## Faz 12 — CV ve Yapısal Analiz Yolculuğu (Retrospektif)

Bu bölüm, teknik bir rapordan ziyade geliştirme sürecindeki deneyimlerin, hataların ve çıkarımların bir günlüğüdür.

### 12.1 — Problemi Nasıl Parçaladık?
CV dokümanı (`CV-ornek-muhendis.pdf`) geldiğinde iki ana sorun tespit ettik:
1. **İçerik:** PDF text layer bozuktu (`İ` -> ``).
2. **Yapı:** Başlıklar numarasızdı ("İŞ DENEYİMİ"), bu yüzden parser bunları yakalayamadı.

### 12.2 — Hangi Yaklaşımları Denedik?
- **VLM Force Modu:** PDF bozuksa VLM (Görsel Analiz) kullanalım dedik.
    - *Sorun:* Kodda `_pick_best_candidate` fonksiyonu, VLM çıktısını bile (yapısal puanı düşük diye) eliyordu. Yani "force" gerçekten force etmiyordu.
- **Regex İyileştirmesi:** "İŞ DENEYİMİ" gibi kelimeleri başlık olarak tanıtmaya çalıştık.
    - *Başarı:* Keyword-based heading detection ve (kontrollü) ALLCAPS detection ile numarasız başlıkları yakaladık.

### 12.3 — Kritik Karar Noktaları
- **Kod vs Prompt:** VLM promptunu mu iyileştirelim, yoksa Python kodunda parserı mı?
    - *Karar:* Python kodunda parserı (`structure.py`) esnek hale getirmek daha kalıcı ve ucuz bir çözüm oldu.

### 12.4 — Nerede Takıldık?
- `master` branch'inde VLM force modunun çalışmadığını fark etmek zaman aldı. Çünkü testleri hep temiz PDF'lerle yapmıştık. Bozuk PDF ("happy path" dışı) gelince sistemin defansif kodları (structure check) bize engel oldu.
- **Çözüm:** `yedek` branch'inde VLM force logic'ini "bypass structure check" haline getirdik.

### 12.5 — Zamanımızı Nasıl Harcadık?
- **%40:** Sorunun kök nedenini (encoding vs structure parser) anlamak.
- **%40:** Git branch yönetimi ve versiyonlar arası geçiş (kullanıcı isteğiyle).
- **%20:** Kod fixleri.

### 12.6 — Neyi Farklı Yapardım?
- **Erken "Kötü Veri" Testi:** Sadece düzgün PDF'lerle değil, bozuk/OCR gerektiren PDF'lerle en başta test yapardım.
- **Format Agnostik Parser:** Structure parser'ı sadece akademik/numaralı formatlara göre değil, CV gibi serbest formatlara göre baştan tasarlardım.


### 12.7 — Dürüst Bir İtiraf: Test Verisindeki CV Dosyası
`test_data/CV-ornek-muhendis.pdf` dosyasını yapısal olarak anlamlandırmakta **başarısız olduk**.
- **Soru:** "Bizde zaten VLM (Görsel Model) yok mu?"
- **Cevap:** Evet var, ancak şu an VLM'i **sadece metin okuma (OCR)** amacıyla kullanıyoruz. Metni okuduktan sonra başlıkları ayıran kod (`structure.py`) hala **Regex ve Kural Tabanlı**.
- **Neden:** CV'de "1. İŞ DENEYİMİ" yazmaz, sadece koyu fontla "İŞ DENEYİMİ" yazar. Bizim regex parser numara aradığı için bu başlığı kaçırdı ve tüm CV'yi tek bir paragraf sandı.

### 12.8 — Bugün Sıfırdan Başlasaydım Çözümüm Ne Olurdu?
Eğer projeye bugün sıfırdan başlasaydım, CV gibi yapısal olmayan (unstructured) dokümanlar için hibrit değil, **Vision-First** bir pipeline kurardım:
1. **Layout Analizi (VLM/Docling):** VLM'i sadece harfleri okumak için değil, **sayfa yapısını (JSON Tree)** çıkarmak için kullanırdım. (Örn: "Şu koordinattaki büyük yazıyı başlık yap").
2. **Semantik Segmentasyon:** Başlıkları regex ile değil, görsel boyuta (font size/boldness) ve konuma göre ayırt eden bir model kullanırdım.
3. **Format Dönüşümü:** OCR çıktısını doğrudan Markdown'a çeviren ve encoding hatalarını düzelten (LLM-based correction) bir ara katman eklerdim.

---

## Gelecek Vizyonu ve Teknoloji Yığını (Yol Haritası)

Şu an projeye sıfırdan başlasaydım veya bir sonraki fazda neleri eklerdim:

### A. Gelişmiş Ingestion (Veri Alma)
- **Vision-Native Parsing (Docling v3):** Klasik PDF okuyucular yerine, dökümanı görsel olarak tarayıp tablo ve formülleri kusursuz Markdown formatına çeviren Docling v3 kullanılır.
- **Semantic Chunking:** Metinler sabit karakter sayısına göre değil, anlamın değiştiği noktalardan (AI yardımıyla) bölünür.

### B. İndeksleme ve Arama
- **GraphRAG:** Veriler sadece vektör olarak değil, birbiriyle ilişkili düğümler (nodes) olarak saklanır. Örneğin, bir uçak dökümanında "Motor" ve "Bakım Takvimi" arasındaki ilişki bir graf yapısı olarak tutulur.
- **Hybrid Search:** Vektör benzerliği (Semantic) + Anahtar kelime (BM25) + Bilgi Grafı (Knowledge Graph) üçlüsünün aynı anda sorgulanması.
- **Late Interaction (ColBERT v3):** Kelimelerin birbirleriyle olan anlamsal ilişkisini koruyarak, "iğne deliği" kadar küçük detayları devasa dökümanlar içinden bulup çıkarma.

### C. Re-Ranking (Yeniden Sıralama)
- **BGE-Reranker-v3:** Vektör DB'den dönen ilk 100 sonuç içinden, soruyla en alakalı 5-10 tanesini seçmek için kullanılan çok daha hassas bir çapraz kodlayıcı modeldir.

---

## Mimari ve Tasarım Kararları

Projenin geliştirme sürecinde alınan temel mimari kararlar ve gerekçeleri:

1. **Neden Hierarchical Chunking?**
   Naif chunking'de "X nelerdir?" gibi sorularda alt maddeler kaybolur. Parent-child yapiyla tum bolum getirilebilir.

2. **Neden Hybrid Search?**
   Dense search semantik benzerligi, BM25 anahtar kelime eslesmeyi yakalar. RRF ile birlestirilince her iki avantaj alinir.

3. **Neden Query Routing?**
   "nelerdir/listele" tipi sorularda top-k yerine complete section fetch yaparak eksik madde sorununu tasarimsal olarak cozer.

4. **Neden Coverage Check?**
   LLM bazen madde atlayabilir. Beklenen vs gercek madde sayisi karsilastirilarak kullaniciya uyari verilir.

5. **Neden Strict System Prompt?**
   Halusinasyonu onlemek icin LLM'e "SADECE baglamdaki bilgiyi kullan" ve "yoksa 'bulunamadi' de" kurallari zorunlu tutulur.

6. **Neden Incremental Indexing?**
   Yeni belge eklendiginde tum mevcut chunk'lari tekrar embed etmek O(n) maliyetlidir. Incremental upsert ile sadece yeni chunk'lar islenir.

7. **Neden Extractive QA?**
   LLM erisimi olmayan ortamlarda (offline/on-prem) da belge sorusu sorulabilsin diye, retrieval sonuclarini dogrudan alinti olarak donduren LLM-free mod eklendi.

