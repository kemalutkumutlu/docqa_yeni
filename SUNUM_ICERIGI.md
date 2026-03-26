# TUSAS DOCQA Teknik Mulakat Sunum Icerigi

Bu dokuman, `/home/kemalutkumutlu/TUSAS_DOCQA/docqa_yeni/TUSAS-DOCQA (4).pdf` ile tutarli olacak sekilde guncellenmistir. Slayt basliklari, sayfa sayisi ve ana mesajlar dogrudan mevcut PDF akisina gore yazilmistir.

## Sunum Akisi

Toplam slayt sayisi: 17

Akis:
- Problem tanimi
- Uctan uca cozum mimarisi
- Dokuman anlama / chunking / retrieval
- Multimodal ve table-layout isleme
- Guardrail, backend, UI ve operasyon
- Test, trade-off, gelecek calismalar ve kapanis

---

## Sayfa 1 - Kapak

**Baslik**
TUSAS DOCQA

**Alt Baslik**
Guvenlik Katmanli Multimodal Dokuman Soru-Cevaplama Sistemi

**Sayfada yer alan metin**
- Kurumsal PDF ve gorsel belgeler icin guvenilir soru-cevap sistemi
- TEKNIK MULAKAT SUNUMU
- KEMAL UTKU MUTLU

**Konusma notu**
Bu projede hedefim yalnizca belgeyi arayan bir sistem kurmak degildi. Belge yapisini koruyan, gerektiginde gorsel extraction kullanan ve cevabini kaynakla savunan guvenilir bir Document QA sistemi gelistirdim.

---

## Sayfa 2 - Problem Tanimi

**Baslik**
Problem Tanimi

**Sayfada yer alan basliklar**
- Kayip Layout ve Gorsel Baglam
- OCR Siniri
- Retrieval Yetersizligi
- Grounding ve Guvenilirlik

**Sayfada yer alan mesajlar**
- Kurumsal belgelerde tablo, form, baslik yapisi ve gorsel yerlesim kritik bilgi tasir
- Salt metin cikarimi bu baglami koruyamaz
- OCR gerekli bir temel katmandir ama tek basina yapisal ve gorsel anlami temsil etmeye yetmez
- Text-only veya sadece dense retrieval liste ve yapisal sorgularda yetersiz kalabilir
- Kaynak gosterimi, guardrail ve grounded answer generation olmadan guvenilirlik saglanamaz

**Konusma notu**
Problemi sadece OCR veya sadece retrieval problemi olarak degil, extraction, yapi ve guvenilirlik problemi olarak ele aldim.

---

## Sayfa 3 - Uctan Uca Cozum Mimarisi

**Baslik**
Uctan Uca Cozum Mimarisi

**Sayfada yer alan bloklar**
- Belge Girdisi: PDF, PNG, JPG ve gorsel/metin tabanli dokumanlar
- Icerik Cikarimi: PDF text, OCR ve opsiyonel VLM extraction
- Yapisal Analiz: Section tree, parent/child chunking ve layout-aware yapi
- Indeksleme: Dense ve sparse index yapisi
- Hibrit Retrieval: Chroma + BM25 + RRF + rerank ile retrieval
- Kaynakli Cevap Uretimi: grounded answer generation, source attribution ve guardrail destekli cikti

**Multimodal yan akis**
- Sayfa/bolge gorselleri
- Layout detection
- OCR / VLM / tablo cikarimi
- Retrieval'a gorsel + metin kanit olarak besleme

**Konusma notu**
Ana mimari text-first cekirdegi koruyor; multimodal akis ise buna yan bir capability olarak ekleniyor.

---

## Sayfa 4 - Dokuman Anlama Katmani

**Baslik**
Dokuman Anlama Katmani

**Sayfada yer alan mesajlar**
- Her sayfa icin metin kalitesi degerlendirilir
- Uygun durumda PDF text kullanilir
- Dusuk kalite durumunda OCR veya VLM tabanli extraction kullanilir
- Aday Secimi: her sayfa icin en yuksek kaliteli extraction adayi belirlenir
- Extraction Stratejisi: PDF text, OCR ve VLM yollari kaliteye gore degerlendirilir; gerekirse daha guclu extraction katmanina gecilir

**Konusma notu**
Bu slaytta anlatilan sey tek bir extractor degil, kaliteye gore secim yapan bir ingestion stratejisi oldugu.

---

## Sayfa 5 - Hiyerarsik Temsil ve Chunking

**Baslik**
Hiyerarsik Temsil ve Chunking

**Alt Baslik**
Hiyerarsik Yapi Nasil Calisir?

**Sayfada yer alan mesajlar**
- Numarali heading yapisi (`2.`, `4.1`, `A.4.1`) tespit edilerek section tree olusturulur
- Her chunk bulundugu bolumle iliskili zengin metadata tasir
- Header/footer temizligi ingestion asamasinda otomatik uygulanir

**Metadata alanlari**
- `section_id`
- `parent_id`
- `heading_path`
- `page_start / end`

**Konusma notu**
Chunking tarafinda fixed-size parcala yerine section-tree mantigini tercih ettim; cunku retrieval'da kapsami korumak istedim.

---

## Sayfa 6 - Retrieval Nasil Calisiyor?

**Baslik**
Retrieval Nasil Calisiyor?

**Sayfada yer alan adimlar**
- Sorgu Siniflandirma: `section_list` ve `normal_qa` ayrimi
- Hibrit Retrieval: Dense (Chroma) + Sparse (BM25) birlesimi
- RRF Birlesimi: Reciprocal Rank Fusion ile yeniden siralama
- Baslik Eslesmeli Section: dogru section ve tum alt agaci getirme

**Konusma notu**
Bu slayt retrieval'i tek asamali bir arama olarak degil, siniflandirma + retrieval + birlestirme + bolum secimi olarak anlatiyor.

---

## Sayfa 7 - Section List Sorgu Mekanizmasi

**Baslik**
Section List Sorgu Mekanizmasi

**Sayfada yer alan mesajlar**
- Liste sorgularinda LLM'e bagimli olmadan deterministik ve kapsam odakli sonuc uretme
- Sorgu Siniflandirma
- Baslik Eslesmesi
- Alt Agac Getirme
- Kapsam Tahmini
- Deterministik Uretim

**Konusma notu**
Bu slaytta ana vurgu, kritik liste sorularinda cevabi tamamen generative moda birakmamak.

---

## Sayfa 8 - Multimodal Katman

**Baslik**
Multimodal Katman

**Alt Baslik**
Gorsel Isleme Mimarisi

**Sayfada yer alan mesajlar**
- Klasik text pipeline'a ek olarak sayfa ve bolge duzeyinde visual chunk uretilir
- Tablo, form ve layout odakli sorgularda region-aware retrieval devreye girer
- Sayfa Chunk: tam sayfa gorsel temsil
- Bolge Chunk: tespit edilen bolge kirpimlari
- Detector Backend: `none`, `sidecar`, `docai`, `docling`

**Konusma notu**
Bu slaytta multimodal katmanin sadece "resim de gonderiyorum" seviyesinde olmadigini, retrieval mantigina gorsel region bilgisinin de girdigini anlatmak gerekiyor.

---

## Sayfa 9 - Tablo ve Layout Isleme

**Baslik**
Tablo ve Layout Isleme

**Sayfada yer alan bloklar**
- Layout Detection: tablo benzeri bolgeler sayfa uzerinde ayri bir asamada tespit edilir
- Extraction: table backend (`docai`, `gemini`, `heuristic`, `auto`) yalnizca ilgili bolgelerde calisir
- Indexing: elde edilen sonuc yapilandirilmis `table` chunk olarak indekslenir
- Retrieval Kalitesi: tablo sorgularinda region odakli evidence daha yuksek dogruluk ve kapsam saglar

**Konusma notu**
Layout detection ile structured table extraction ayni sey degil; bu slaytta bilerek iki asamali bir isleme anlatiyoruz.

---

## Sayfa 10 - Cevap Uretimi ve Guardrail Katmani

**Baslik**
Cevap Uretimi ve Guardrail Katmani

**Sayfada yer alan mesajlar**
- Sistem, belgede bulunmayan bilgiyi uretmemek uzere yapilandirilmistir
- Her cevapta kaynak gosterimi zorunludur
- Cikti Guvencesi: strict system prompt ile LLM davranisi sinirlandirilir
- Streaming ve non-streaming yanit yollari desteklenir
- Cevap uretim dongusu denetlenebilir ve gerektiginde yeniden calistirilabilir
- Baglam Disi Uretim Yasagi: "Belgede bu bilgi bulunamadi."
- Kaynak Zorunlulugu: kaynak gosterilmezse otomatik yeniden deneme
- Kapsam Kontrolu: eksik kapsam tespit edilirse yeniden deneme ve uyari mekanizmasi

**Konusma notu**
Burada odak "daha akici cevap" degil, "daha denetlenebilir ve kaynakli cevap".

---

## Sayfa 11 - Moduler Backend Mimarisi

**Baslik**
Moduler Backend Mimarisi

**Sayfada yer alan mesajlar**
- Her bilesen runtime sirasinda bagimsiz olarak yapilandirilabilir
- Cloud ve local secenekler ayni arayuz uzerinden yonetilir
- LLM Provider: Gemini, OpenAI, local veya extractive
- Embedding: Gemini embedding veya local sentence-transformers
- OCR: `docai`, `paddle_vl`, `tesseract`
- VLM: Gemini veya local vision model
- Layout: `docai`, `docling`, `sidecar`

**Konusma notu**
Bu slayt tek bir servis yoluna kilitlenmeyen, runtime'da degistirilebilir bir backend mimarisi kurdugunuzu gosteriyor.

---

## Sayfa 12 - Arayuz ve Kontrol Edilebilirlik

**Baslik**
Arayuz ve Kontrol Edilebilirlik

**Sayfada yer alan mesajlar**
- Demo sirasinda OCR, VLM, retrieval ve generation ayarlari anlik yonetilebilir
- Runtime Yonetimi: preset ve backend secimi anlik olarak yapilandirilabilir
- Ayar Kontrolu: OCR, VLM ve retrieval ayarlari demo sirasinda yonetilebilir
- Esnek Akis: farkli backend kombinasyonlariyla test edilebilir

**Konusma notu**
Bu slayt UI'yi yalnizca chat arayuzu olarak degil, teknik bir kontrol paneli olarak konumlandiriyor.

---

## Sayfa 13 - Operasyonel Dayaniklilik

**Baslik**
Operasyonel Dayaniklilik

**Sayfada yer alan bloklar**
- Belge Onbellegi: fingerprint tabanli cache invalidation ile gereksiz yeniden isleme onlenir
- Artimli Indeksleme: Persistent Chroma + BM25 yapisinda yalnizca degisen belgeler yeniden islenir
- Guvenli Kapanis: calisan surecler guvenli bicimde sonlandirilir
- Lazy Import: kullanilmayan backend bagimliliklari baslangicta yuklenmez
- Olay Loglama: opsiyonel yapilandirilmis event logging ile gozlemlenebilirlik saglanir

**Konusma notu**
Bu slayt, projenin sadece dogru cevap veren degil, ayni zamanda operasyonel olarak da dayanikli bir uygulama oldugunu gosteriyor.

---

## Sayfa 14 - Test ve Degerlendirme Stratejisi

**Baslik**
Test ve Degerlendirme Stratejisi

**Alt Baslik**
Vaka Calismasi Sonuclari

**Belgeler**
- `Case_Study_20260205.pdf`
- `CV-ornek-muhendis.pdf`

**Tabloda yer alan metrikler**
- Intent Accuracy: `25/25 (100%)`
- Heading Hit: `15/15 (100%)`
- Section Hit: `6/6 (100%)`
- Evidence Met: `25/25 (100%)`
- Manual QA:
  - Online: `71/71`
  - Local: `39/71`

**Degerlendirme metrikleri**
- Intent Accuracy
- Heading Hit
- Section Hit
- Evidence Met
- Hallucination Test
- Manual QA

**Konusma notu**
Bu slaytta script tabanli retrieval degerlendirmesi ile manuel QA sonucunu birlikte anlatiyorsunuz; ozellikle VLM acik/kapali farkinin sonuclara etkisini vurgulamak uygun olur.

---

## Sayfa 15 - Teknik Kararlar ve Trade-off'lar

**Baslik**
Teknik Kararlar ve Trade-off'lar

**Sayfada yer alan mesajlar**
- Text-First Mimari Korundu
- Deterministic Retrieval Onceliklendirildi
- UI'da Apply/Reset Akisi
- Graph RAG / Agentic RAG Bilincli Olarak Eklenmedi

**Detaylar**
- Multimodal katman temel pipeline degistirilmeden additive olarak eklendi
- LLM'e birakilabilecek kararlar mumkun oldugunda heuristic ve kural tabanli mekanizmalarla cozuldu
- Runtime ayarlari kullanici onayina baglandi
- Kapsam, denetlenebilirlik ve sadelik onceliklendirildi

**Konusma notu**
Bu slaytta "neden yapildi?" kadar "neden bilincli olarak yapilmadi?" sorusunu da cevapliyorsunuz.

---

## Sayfa 16 - Gelecek Calismalar

**Baslik**
Gelecek Calismalar

**Uzun Vadeli Vizyon**
- Graph RAG
- Agentic RAG

**Kisa Vadeli Hedefler**
- Reranker / Late Interaction
- Gelismis Bolge ve Tablo Cikarimi
- Genis Benchmark Setleri
- Gozlemlenebilirlik

**Konusma notu**
Bu slayt mevcut sistemi abartmadan, sonraki teknik genisleme alanlarini gosteriyor.

---

## Sayfa 17 - Tesekkurler

**Baslik**
Tesekkurler

**Sayfada yer alan metin**
- Sorularinizi memnuniyetle yanitlayabilirim.
- Kemal Utku Mutlu
- TUSAS DOCQA - Teknik Mulakat Sunumu

---

## Not

Bu markdown artik PDF'deki 17 sayfalik mevcut sunumla hizalidir. Onceki taslakta bulunan su bolumler artik bu dosyada ayrica slayt olarak tutulmamistir:
- Mevcut demo profili
- Canli demo akisi
- Muhtemel mulakat sorulari
- Ayrica ayri kapanis/sonuc slaydi

Istenirse bu basliklar ayri bir "konusma notlari" dokumani olarak yeniden ayrilabilir; ancak mevcut PDF ile birebir tutarlilik icin bu dosyada tutulmadi.
