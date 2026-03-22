# SLAYT SAYFA SAYFA ICERIK

Bu dokuman, Gamma ile sunum uretmek icin hazirlanmis slayt-odakli iceriktir.
Amac: her sayfada gorunecek metni, yerlesim tipini ve gorsel yonunu net vermek.
Not: Konusma notlarinin detayli versiyonu `SUNUM_ICERIGI.md` icinde bulunuyor.

---

## Slayt 1 - Kapak

**Layout**
Hero cover

**Baslik**
TUSAS DOCQA

**Alt Baslik**
Hiyerarsik, Multimodal ve Guardrail Destekli Belge Soru-Cevap Sistemi

**Alt Bant**
- Technical Interview Presentation
- PDF ve gorsel belgeler icin structure-aware retrieval
- Belgeden cevap, kaynakla birlikte

**Gorsel Onerisi**
PDF + chat + citation akisini cagrıştıran temiz bir kapak gorseli

---

## Slayt 2 - Problem Tanimi

**Layout**
2 sutunlu problem slaydi

**Baslik**
Neyi cozuyorum?

**Sol Tarafta**
- Kurumsal PDF'lerde metin katmani her zaman temiz degil
- Scan/image PDF'lerde OCR gerekiyor
- Tablolar, formlar ve layout kritik bilgi tasiyor

**Sag Tarafta**
- Kullanici cogu zaman paragraf degil, bolum veya liste soruyor
- Sadece vector search ile eksiksiz cevap almak zor
- Halusinasyon ve kaynak gosterimi kritik risk

**Alt Mesaj**
Problem sadece arama degil; extraction, structure ve guvenilir cevap problemi

---

## Slayt 3 - Cozum Ozeti

**Layout**
Pipeline overview + capability cards

**Baslik**
Sistemin kisa ozeti

**Ana Akis**
Belge -> Ingestion -> Structure -> Chunking -> Indexing -> Retrieval -> Guardrailed Answer

**Kartlar**
- Girdi: PDF / PNG / JPG
- Extraction: PyMuPDF / Docling / OCR / VLM
- Retrieval: Dense + Sparse + RRF + rerank
- Cevap: Gemini / OpenAI / Local / Extractive

**Alt Mesaj**
Tek bir backend'e bagli olmayan, moduler bir belge QA mimarisi

---

## Slayt 4 - Uctan Uca Mimari

**Layout**
Flow diagram

**Baslik**
End-to-end pipeline

**Slaytta Gorunecek Akis**
```text
Belge
-> Ingestion
-> Structure Detection
-> Hierarchical Chunking
-> Dense + Sparse Indexing
-> Hybrid Retrieval
-> Guardrailed Answer Generation
```

**Ek Dal**
```text
Multimodal modda:
Visual Assets
-> Page / Region Planning
-> OCR / VLM / Table Structure
-> Visual + Text Evidence
```

**Alt Mesaj**
Text-first cekirdek korunuyor, multimodal katman buna ekleniyor

---

## Slayt 5 - Ingestion ve Extraction

**Layout**
4 kartli teknik slayt

**Baslik**
Belgeyi once dogru okumak zorundayim

**Kart 1**
PDF Text
- `pymupdf`
- `docling`
- `auto`
- `smart`

**Kart 2**
OCR
- `docai`
- `paddle_vl`
- `paddle`
- `tesseract_legacy`
- `smart`

**Kart 3**
VLM
- `off`
- `auto`
- `smart`
- `force`

**Kart 4**
Secim Mantigi
- Her sayfada en iyi extraction adayi seciliyor
- Multi-column okuma sirasi duzeltiliyor
- OCR metni VLM icin grounding olarak kullanilabiliyor

---

## Slayt 6 - Yapisal Temsil ve Chunking

**Layout**
Section tree + 2 kutu

**Baslik**
Neden hiyerarsik chunking kullandim?

**Ust Kisim**
- Numbered heading detection: `2.`, `4.1`, `A.4.1`
- Repeating header/footer temizligi
- Section tree olusumu

**Alt Sol**
Parent chunk
- Tam bolum
- Kapsam butunlugunu korur

**Alt Sag**
Child chunk
- Paragraf-hizali, token-bazli bolumleme
- `tiktoken cl100k_base`
- Retrieval hassasiyetini artirir

**Alt Mesaj**
Kullanici paragraf degil, cogu zaman bolum soruyor

---

## Slayt 7 - Hybrid Retrieval

**Layout**
3 katmanli retrieval slaydi

**Baslik**
Hybrid retrieval ve query routing

**Katman 1**
Dense retrieval
- Embedding tabanli anlamsal arama

**Katman 2**
Sparse retrieval
- BM25
- Child/table chunk odakli
- Turkce/Ing. dostu tokenizer + 3-gram destegi

**Katman 3**
Routing ve rerank
- `section_list`
- `multi_section`
- `normal_qa`
- RRF + embedding rerank

**Alt Mesaj**
Tum sorulari ayni retrieval yoluna sokmuyorum

---

## Slayt 8 - Section List Guvencesi

**Layout**
Step-by-step process

**Baslik**
Eksiksiz listeleme icin ozel mekanizma

**Akis**
1. Sorgu `section_list` olarak siniflanir
2. En uygun bolum heading-aware secilir
3. Topic-heading confidence guard uygulanir
4. Top-k yerine tum section + subtree fetch yapilir
5. Coverage heuristic ile beklenen madde sayisi hesaplanir
6. Uygunsa deterministic liste render edilir
7. Yetmezse LLM fallback devreye girer

**Alt Mesaj**
Kritik listelerde uretici modeli degil, veriyi otorite yaptim

---

## Slayt 9 - Multimodal Katman

**Layout**
Card grid

**Baslik**
Metin yetmediginde ne oluyor?

**Kartlar**
- Processing mode: `classic`, `multimodal`, `smart`
- Visual chunk level: `page` veya `region`
- Region source: `heuristic` veya `detector`
- Detector backend: `none`, `sidecar`, `docai`, `docling`
- Query tablo/form/layout sinyali tasiyorsa region-aware skor bonusu
- Multimodal answer generation: `off`, `auto`, `on`

**Alt Mesaj**
Multimodal katman sadece goruntu eklemek degil; retrieval mantigina gorsel bolge bilgisini katmak

---

## Slayt 10 - Table ve Layout

**Layout**
2 sutunlu neden-sonuc slaydi

**Baslik**
Table structure ve layout neden ayri stage?

**Sol Tarafta**
- Layout detection ayri
- OCR ayri
- Table parsing ayri

**Sag Tarafta**
- `TABLE_STRUCTURE_ENABLED` ile ac/kapat
- `smart` flag ile sadece OCR/VLM sayfalarinda calistir
- Backend: `docai`, `gemini`, `heuristic`, `auto`
- Sonuc `table` chunk olarak indekse girer

**Alt Mesaj**
OCR metin verir; table stage ise satir-sutun iliskisini korumaya calisir

---

## Slayt 11 - Guardrail'li Cevap Uretimi

**Layout**
Checklist + comparison

**Baslik**
LLM kullaniyorum ama kontrolsuz birakmiyorum

**Guardrail Checklist**
- Siki system prompt
- Baglam disi bilgi yasak
- Baglam yoksa: `Belgede bu bilgi bulunamadi.`
- Yanlis onculde: `Hayir, ...` ile duzeltici cevap
- Her bilgi cumlesinde citation zorunlu
- Weak evidence durumunda grounding check
- Deterministic section-list path
- Kanit paneli: yalnizca cevapta atifta bulunulan sayfalarin evidence ozeti

**Mini Karsilastirma**
- Non-streaming: citation retry + coverage retry + warning
- Streaming: token geri alma yok, silent citation duzeltme + warning

---

## Slayt 12 - Moduler Backend Mimarisi

**Layout**
Capability matrix

**Baslik**
Neden bu kadar konfigure edilebilir?

**Satirlar**
- LLM: `gemini`, `openai`, `local`, `none`
- Embedding: Gemini veya local sentence-transformers
- OCR: cloud ya da local
- VLM: cloud ya da local
- Layout detector: `none`, `sidecar`, `docai`, `docling`
- Detector basarisiz olursa heuristic fallback

**Alt Mesaj**
Kalite, maliyet, offline ihtiyaci ve donanim imkanina gore farkli pipeline kurabiliyorum

---

## Slayt 13 - UI ve Demo Deneyimi

**Layout**
Screenshot + callout labels

**Baslik**
Chainlit tabanli ama standart bir chat arayuzu degil

**Sag Tarafta Gosterilecekler**
- `Basic` ve `Advanced` sekmeleri
- Runtime preset'ler: `online_best`, `hybrid_best`, `local_best`, `fast`
- `Apply` / `Reset` akisi
- Ozet kartlari:
  - `Current Draft`
  - `Applied Pipeline`
  - `Fallback Notes`
  - `Document Context`
- Komutlar:
  - `/chat`
  - `/doc`
  - `/use <dosya>`

**Alt Mesaj**
UI sadece chat degil; aktif pipeline ve fallback mantigini gosteren bir demo arayuzu

---

## Slayt 14 - Operasyonel Dayaniklilik

**Layout**
5 kutulu ops slaydi

**Baslik**
Sistemi sadece dogru degil, dayanikli da yaptim

**Kutular**
- Document cache
- Fingerprint bazli cache invalidation
- Incremental indexing
- Persistent Chroma + BM25
- Multi-document session ve partial filename routing

**Ek Kutular**
- Embedding dimension bazli collection isimlendirme
- Optional event logging
- Graceful shutdown / lazy import

**Alt Mesaj**
Ayni belgeyi tekrar tekrar isletmeden, oturumlar arasi daha saglam bir kullanim

---

## Slayt 15 - Test ve Degerlendirme

**Layout**
2 sutunlu: otomatik testler + manuel QA sonuclari

**Baslik**
Kaliteyi nasil dogruladim?

**Sol Tarafta - Otomatik Testler**
- Preflight: generation + embedding + VLM smoke
- Syntax/import gate
- Baseline gate
- Retrieval eval:
  - intent accuracy
  - heading hit
  - section hit
  - evidence met
  - latency
- Hallucination test
- Smoke suite / folder suite

**Sag Tarafta - Manuel QA Ozeti**
- Kaynak: `test_sonuclari.md`
- Belgeler: `Case_Study_20260205.pdf` + `CV-ornek-muhendis.pdf`
- Online mode: `71/71` soru dogru referans kabul
- Local mode: `39/71` dogru, online moddan belirgin zayif
- Paraphrase tutarliligi, liste coverage ve belge disi no-answer davranisi gozlemlendi
- VLM etkisi: online testte `vlm_mode=force`, local testte `vlm_mode=off`; ozellikle CV gibi layout-duyarli belgelerde dogruluk belirgin dustu

**Alt Mesaj**
Hem tekrar uretilebilir script testleri hem de gercek kullaniciya benzeyen manuel QA akislarini kullandim

---

## Slayt 16 - Mevcut Demo Profili

**Layout**
Profile snapshot

**Baslik**
Bugun gosterecegim aktif konfigurasyon

**Profil Kartlari**
- Generation:
  - `LLM_PROVIDER=gemini`
  - `GEMINI_MODEL=gemini-3.1-pro-preview`
  - `GEMINI_FALLBACK_MODEL=gemini-2.5-pro`
- Embedding:
  - `EMBEDDING_MODEL=gemini-embedding-2-preview`
  - `EMBEDDING_VERTEX_ENABLED=0`
- Extraction:
  - `PDF_TEXT_BACKEND=docling`
  - `OCR_ENABLED=0`
- Multimodal:
  - `DOC_PROCESSING_MODE=multimodal`
  - `VISUAL_CHUNK_LEVEL=page`
  - `VLM_PROVIDER=gemini`
  - `VLM_MODE=force`
- Table / Layout:
  - `TABLE_STRUCTURE_ENABLED=0`
  - `TABLE_STRUCTURE_BACKEND=auto`
  - `VISUAL_REGION_SOURCE=detector`
  - `VISUAL_DETECTOR_BACKEND=docai`

**Alt Not**
Pratik demo odagi: page-level multimodal akis; region/detector capability sistemde mevcut ama bu profilde ana gosterim noktasi degil

---

## Slayt 17 - Teknik Kararlar ve Trade-off'lar

**Layout**
Decision cards

**Baslik**
Neleri bilerek sectim, nelerden bilerek vazgectim?

**Kartlar**
- Text-first cekirdegi korudum, multimodal'i additive ekledim
- Deterministic retrieval iyilestirmelerini LLM'den once koydum
- Moduler backend mimarisi sectim, tek bir servis yoluna kilitlenmedim
- UI'da tam reactive editor yerine `Apply/Reset` modeli kullandim
- Kritik listelerde generative degil deterministic yol kullandim
- Graph RAG ve agentic RAG bugun yok; bilincli olarak eklenmedi

**Alt Mesaj**
Onceligim once temel belge QA problemini guvenilir sekilde cozmekt

---

## Slayt 18 - Canli Demo Akisi

**Layout**
Numbered timeline

**Baslik**
Demo'yu nasil ilerletecegim?

**Adimlar**
1. Uygulamayi ac
2. Runtime panelde aktif pipeline'i goster
3. Test PDF'lerinden birini yukle
4. Normal QA sorusu sor
5. Liste/bolum sorusu sor
6. Tablo veya layout odakli soru sor
7. Citation ve evidence mantigini goster
8. Gerekirse `/use <dosya>` ile aktif belgeyi degistir

**Alt Mesaj**
Demo sirasi: temel akis -> deterministic liste -> multimodal fark

---

## Slayt 19 - Muhtemel Mulakat Sorulari

**Layout**
2 sutunlu Q/A slaydi

**Baslik**
Bekledigim teknik sorular

**Soru / Kisa Cevap**
- Neden sadece vector DB yetmiyor?
  Exact terimler, liste tamligi ve section secimi icin yetmiyor.
- Neden BM25 ekledin?
  Kelime bazli eslesmeleri ve teknik terimleri guclendiriyor.
- Neden section tree kurdun?
  Kullanici genelde paragraf degil, bolum soruyor.
- Neden OCR, Docling ve VLM birlikte var?
  Tek extractor tum sayfalarda surekli en iyi sonucu vermiyor.
- Neden layout ve table stage ayri?
  OCR metin verir; tablo yapisini garanti etmez.
- Halusinasyonu nasil azalttin?
  Guardrail, grounding, deterministic path ve citation zorunlulugu ile.

---

## Slayt 20 - Gelecek Calismalar

**Layout**
Roadmap

**Baslik**
Bir sonraki teknik adimlar

**Kisa Vadeli**
- Cross-encoder veya ColBERT benzeri ek reranker
- Region planning ve table extraction kalitesini artirma
- Form/diagram/technical drawing ayrimini guclendirme
- Daha genis benchmark havuzu ve sistematik raporlama
- Daha fazla preset/profil ve loglama altyapisi

**Uzun Vadeli Opsiyonlar**
- Graph RAG
- Agentic RAG

**Alt Mesaj**
Bugun sistemde olmayan seyleri varmis gibi anlatmiyorum; roadmap olarak ayri konumluyorum

---

## Slayt 21 - Kapanis

**Layout**
Strong close

**Baslik**
Sonuc

**Ana Mesajlar**
- Belge yapisini koruyan ingestion
- Query-aware ve hybrid retrieval
- Multimodal/table destekli evidence toplama
- Guardrail'li cevap uretimi
- Demo dostu ve konfigure edilebilir UI

**Kapanis Cumlesi**
Bu projede hedefim, belgeyi sadece arayan degil; belge yapisini anlayan, farkli extraction ve retrieval stratejilerini birlestiren ve cevabini kaynagi ile savunan bir sistem gelistirmekti.

---

## Gamma Kullanim Notu

- Her `Slayt X` bolumunu Gamma'da tek sayfa olarak kullan.
- `Layout` satirini Gamma prompt'unda sayfa tipi olarak yazabilirsin.
- `Gorsel Onerisi` kisimlarini ilgili sayfaya gorsel referans notu olarak ekleyebilirsin.
- Daha konuskan versiyon gerekiyorsa `SUNUM_ICERIGI.md` icindeki konusma notlarini speaker notes olarak kullan.
