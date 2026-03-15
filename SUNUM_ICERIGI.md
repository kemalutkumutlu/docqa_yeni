# TUSAS DOCQA Teknik Mulakat Sunum Icerigi

Bu dokuman, mevcut kod tabani taranarak hazirlanmis teknik sunum iskeletidir. Icerik, repo icindeki gercek akis, moduller, testler ve aktif demo konfigurasyonuna gore yazilmistir.

## Sunum Stratejisi

- Sunumu 12-15 dakika icinde bitir.
- Once problemi ve cozum mimarisini anlat, sonra demo yap, en sonda teknik kararlarini savun.
- Kod seviyesine ancak soru gelince in; ana anlatimda sistem davranisini ve nedenlerini one cikar.

---

## Sayfa 1 - Kapak

**Baslik**
TUSAS DOCQA: Multimodal, Hiyerarsik ve Guvenlik Katmanli Belge Soru-Cevap Sistemi

**Alt Baslik**
PDF ve gorsel belgeler icin OCR + layout + hybrid retrieval + kaynakli cevap uretimi

**Sayfada yer almasi gerekenler**
- Proje adi
- Isim
- "Technical Interview Presentation"
- Kisa slogan: "Belgeden cevap, kaynakla birlikte"

**Konusma notu**
Bu projede hedefim, sadece belgeyi arayan bir sistem degil; belgeyi anlayan, yapisini koruyan, gerekirse OCR ve gorsel analiz kullanan ve cevabi kaynaklariyla birlikte ureten bir sistem kurmakti.

---

## Sayfa 2 - Problem Tanimi

**Baslik**
Neyi cozuyorum?

**Sayfada yer almasi gerekenler**
- Kurumsal PDF'lerde metin her zaman temiz degil
- Tarama belgelerde OCR gerekiyor
- Tablolar ve layout kritik bilgi tasiyor
- Sadece vector search ile eksiksiz cevap almak zor
- Halusinasyon ve kaynak gosterimi kritik risk

**Konusma notu**
Klasik RAG yaklasimlari metin katmani temiz olan belgelerde iyi calisiyor. Fakat gercek kurumsal dokumanlarda tarama PDF, tablo, form, layout ve baslik hiyerarsisi devreye giriyor. Bu nedenle sistemi sadece "text chunk + embedding" seviyesinde birakmadim.

---

## Sayfa 3 - Cozum Ozeti

**Baslik**
Sistemin kisa ozeti

**Sayfada yer almasi gerekenler**
- Girdi: PDF, PNG, JPG
- Ingestion: PDF text + OCR + istege bagli VLM extract
- Yapisal analiz: section tree + parent/child chunking
- Retrieval: Chroma + BM25 + RRF
- Multimodal: page/region bazli visual chunk ve table chunk
- Generation: Gemini / OpenAI / Local / Extractive
- Cikti: kaynakli, guvenlik katmanli cevap

**Konusma notu**
Mimariyi modulere kurdum. Boylece ayni sistem hem online hem local, hem classic hem multimodal, hem de farkli OCR ve LLM backend'leri ile calisabiliyor.

---

## Sayfa 4 - Uctan Uca Mimari

**Baslik**
End-to-end pipeline

**Sayfada yer almasi gerekenler**
```text
Belge
-> Ingestion
-> Structure Detection
-> Hierarchical Chunking
-> Dense + Sparse Indexing
-> Hybrid Retrieval
-> Answer Generation
```

**Ek kutu**
Multimodal modda:
```text
Page/Region Visual Assets
-> Layout Regions
-> OCR / VLM / Table Structure
-> Visual + Text Evidence
```

**Konusma notu**
Temel tasarim kararim su oldu: classic text-first akisi bozmadan, multimodal katmani buna ek olarak yerlestirmek. Yani sistem yeni capability eklerken mevcut stabil davranisi kaybetmiyor.

---

## Sayfa 5 - Belge Anlama Katmani

**Baslik**
Ingestion ve extraction tasarimi

**Sayfada yer almasi gerekenler**
- PDF text backend: `pymupdf`, `docling`, `auto`
- OCR backend: `docai`, `paddle_vl`, `paddle`, `tesseract_legacy`
- VLM extract: `gemini` veya local vision model
- Candidate secimi: en iyi yapiyi koruyan metin seciliyor
- Dusuk kalite metinde OCR/VLM devreye giriyor

**Konusma notu**
Burada kritik karar, farkli extract kaynaklarini yaristirip en iyi yapisal sonucu secmek oldu. Sadece en uzun metni almak yerine, baslik korunumunu ve bozulmus satir oranini da dikkate aldim. Bu sayede section detection kalitesi arttı.

---

## Sayfa 6 - Yapisal Temsil ve Chunking

**Baslik**
Neden hiyerarsik chunking kullandim?

**Sayfada yer almasi gerekenler**
- Numbered heading detection: `2.`, `4.1`, `A.4.1`
- Repeating header/footer temizligi
- Section tree olusumu
- Parent chunk: tam bolum
- Child chunk: kucuk ve retrieval dostu parcali metin
- Metadata: `section_id`, `parent_id`, `heading_path`, `page_start`, `page_end`

**Konusma notu**
Duzenli chunking yerine section tree kullandim cunku kullanici genelde paragraf degil, "bolum" soruyor. Parent chunk tam kapsami, child chunk ise retrieval hassasiyetini sagliyor. Bu hibrit yapi eksiksiz liste sorularinda cok fayda sagladi.

---

## Sayfa 7 - Retrieval Nasil Calisiyor?

**Baslik**
Hybrid retrieval ve query routing

**Sayfada yer almasi gerekenler**
- Dense retrieval: embedding tabanli anlamsal arama
- Sparse retrieval: BM25 ile kelime bazli arama
- RRF ile birlestirme
- Query classification:
  - `section_list`
  - `normal_qa`
- Heading-aware section secimi
- Gerekirse tum section + subtree fetch

**Konusma notu**
Sistemde tum sorular ayni sekilde ele alinmiyor. "Teslimatlar nelerdir?" gibi sorularla "Teslim suresi nedir?" ayni retrieval stratejisine gitmiyor. Bu ayrim, hem eksik madde problemini hem de yanlis section secimini azaltıyor.

---

## Sayfa 8 - Section List Sorularini Nasil Guvenceye Aldim?

**Baslik**
Eksiksiz listeleme icin ozel mekanizma

**Sayfada yer almasi gerekenler**
- `section_list` query'leri once siniflandiriliyor
- En uygun bolum heading overlap ile seciliyor
- Sadece top-k degil, tum alt agac getiriliyor
- Coverage heuristic ile beklenen madde sayisi hesaplanıyor
- Uygunsa deterministic liste render ediliyor
- Yetmezse LLM fallback devreye giriyor

**Konusma notu**
Bu kisim benim icin fark yaratan teknik kararlardan biri oldu. Klasik RAG'de LLM bazen listedeki bazi maddeleri atliyor. Ben burada once yapisal olarak kac madde oldugunu tahmin edip, mumkunse listeyi deterministic urettim. Boylece hem halusinasyon hem eksik madde riski dustu.

---

## Sayfa 9 - Multimodal Katman

**Baslik**
Metin yetmediginde ne oluyor?

**Sayfada yer almasi gerekenler**
- Processing mode: `classic` veya `multimodal`
- Visual chunk level: `page` veya `region`
- Region source: `heuristic` veya `detector`
- Detector backend: `none`, `sidecar`, `docai`, `docling`
- Visual evidence generation prompt'una dahil edilebiliyor
- Query'de tablo/form/layout sinyali varsa region skor bonusu uygulanıyor

**Konusma notu**
Multimodal katmani sadece "resim ekledim" seviyesinde kurmadim. Region bazli crop, detector backend secimi ve retrieval tarafinda visual-region aware scoring ekledim. Boylece tablo, form, checkbox veya page layout gibi sorularda ilgili bolge daha yuksek ihtimalle one cikiyor.

---

## Sayfa 10 - Tablo ve Layout Isleme

**Baslik**
Table structure ve layout neden ayri stage?

**Sayfada yer almasi gerekenler**
- Layout detection ayri, OCR ayri, table parsing ayri
- Table backend:
  - `docai`
  - `gemini`
  - `heuristic`
  - `auto`
- Sadece table benzeri region'larda calisiyor
- Sonuc `table` chunk olarak indekse giriyor
- Tablo sorulari icin daha anlamli evidence uretiliyor

**Konusma notu**
Bu ayrimi bilerek yaptim cunku OCR metin verir ama tablo yapisini her zaman korumaz. Layout detector bolgeyi buluyor, table stage ise yapisal satir-sutun bilgisini cikarmaya calisiyor. Bu da tablo sorularinda cevabin kalitesini artiriyor.

---

## Sayfa 11 - Cevap Uretimi ve Guardrail'ler

**Baslik**
LLM kullaniyorum ama kontrolsuz birakmiyorum

**Sayfada yer almasi gerekenler**
- Siki system prompt
- Baglam disi bilgi yasak
- Baglam yoksa sabit fallback:
  - `Belgede bu bilgi bulunamadi.`
- Her bilgi cumlesinde citation zorunlu
- Citation yoksa retry
- Coverage eksikse retry ve warning
- Streaming ve non-streaming varyantlari var

**Konusma notu**
Buradaki hedefim "guzel cevap" degil, "guvenilir cevap" oldu. Bu yuzden cevabin kaynaksiz gelmesini kabul etmiyorum. Liste sorularinda kapsama kontrolu yapiyorum. Gerektiginde extractive yol da kullanabiliyorum.

---

## Sayfa 12 - Neden Bu Kadar Konfigure Edilebilir?

**Baslik**
Moduler backend mimarisi

**Sayfada yer almasi gerekenler**
- LLM provider: `gemini`, `openai`, `local`, `none`
- Embedding: Gemini veya local sentence-transformers
- OCR: cloud ya da local
- VLM: cloud ya da local
- Layout: `docai`, `docling`, `sidecar`
- UI uzerinden runtime degisiklikleri yapilabiliyor

**Konusma notu**
Projenin ana hedeflerinden biri tek bir ortama bagimli kalmamakti. Kalite, maliyet, offline ihtiyaci ve donanim imkanina gore farkli pipeline'lar kurulabiliyor. Bu esneklik teknik olarak maliyetliydi ama urun acisindan cok degerli oldu.

---

## Sayfa 13 - UI ve Demo Deneyimi

**Baslik**
Chainlit tabanli ama standart bir chat arayuzu degil

**Sayfada yer almasi gerekenler**
- Sol ozel sohbet gecmisi paneli
- Sag custom runtime settings paneli
- `Basic` ve `Advanced` sekmeleri
- Ozet kartlari:
  - `Current Draft`
  - `Applied Pipeline`
  - `Fallback Notes`
  - `Document Context`
- Komutlar:
  - `/chat`
  - `/doc`
  - `/use <dosya>`

**Konusma notu**
UI'yi sadece soru-cevap ekranı olarak birakmadim. Teknik mulakatta gostermeyi kolaylastiran bir runtime panel ekledim. Boylece ayni uygulamada hem demo yapabiliyor hem de hangi pipeline'in aktif oldugunu seffaf sekilde gosterebiliyorum.

---

## Sayfa 14 - Operasyonel Dayaniklilik

**Baslik**
Sistemi sadece dogru degil, dayanikli da yapmaya calistim

**Sayfada yer almasi gerekenler**
- Document cache ile ayni dosya tekrar islenmiyor
- Fingerprint ile config degisirse cache invalidation oluyor
- Incremental indexing ile sadece yeni chunk embed ediliyor
- Persistent Chroma + BM25
- Graceful shutdown
- Lazy import
- Event logging opsiyonel

**Konusma notu**
Kod tabaninda sadece algoritma degil, operasyonel davranis da onemliydi. Ayni belgeyi tekrar tekrar islememek, yeni dosya geldiginde tum indexi bastan embed etmemek ve provider hatalarinda sistemi ayakta tutmak bu yuzden oncelikliydi.

---

## Sayfa 15 - Test ve Degerlendirme Yaklasimi

**Baslik**
Kaliteyi nasil dogruladim?

**Sayfada yer almasi gerekenler**
- Preflight: generation + embedding + VLM smoke
- Syntax/import gate
- Baseline gate
- Retrieval eval:
  - intent accuracy
  - heading hit
  - section hit
  - evidence recall
  - latency
- Hallucination test
- Smoke suite
- Folder suite ile coklu PDF denemeleri

**Konusma notu**
Test stratejim sadece unit test degil. Bu tarz bir RAG sistemde ingestion, retrieval ve generation birlikte test edilmeli. Bu yuzden script tabanli kabul kapilari kurdum. Ozellikle retrieval tarafinda LLM-free evaluasyon ayirdim.

---

## Sayfa 16 - Mevcut Demo Profili

**Baslik**
Bugun gosterecegim aktif konfigurasyon

**Sayfada yer almasi gerekenler**
- `LLM_PROVIDER=gemini`
- `GEMINI_MODEL=gemini-3.1-pro-preview`
- `GEMINI_FALLBACK_MODEL=gemini-2.5-pro`
- `EMBEDDING_MODEL=gemini-embedding-2-preview`
- `DOC_PROCESSING_MODE=multimodal`
- `VISUAL_CHUNK_LEVEL=region`
- `VISUAL_REGION_SOURCE=detector`
- `VISUAL_DETECTOR_BACKEND=docai`
- `OCR_BACKEND=paddle_vl`
- `TABLE_STRUCTURE_BACKEND=auto`
- `VLM_PROVIDER=gemini`
- `VLM_MODE=force`

**Konusma notu**
Demo icin kaliteyi one cikan hibrit bir profil kullaniyorum. OCR tarafinda local GPU gucunden, layout ve generation tarafinda ise cloud modellerden yararlaniyorum. Bu kombinasyon kalite-maliyet dengesinde guclu bir nokta sagliyor.

---

## Sayfa 17 - Teknik Kararlar ve Trade-off'lar

**Baslik**
Neleri bilerek sectim, nelerden bilerek vazgectim?

**Sayfada yer almasi gerekenler**
- Text-first mimariyi korudum, multimodal'i additive ekledim
- Deterministic retrieval iyilestirmelerini LLM'den once koydum
- Moduler backend mimarisi secildi, tek yol secilmedi
- UI'da tam reactive config editor yerine `Apply/Reset` modeli secildi
- Region/table capability var, ama Graph RAG veya agentic RAG eklenmedi

**Konusma notu**
Her capability'yi eklemek yerine, belge QA problemini dogrudan cozen ve teknik olarak savunulabilir katmanlara odaklandim. Bu da projeyi gereksizce karmasiklastirmadan guclendirdi.

---

## Sayfa 18 - Canli Demo Akisi

**Baslik**
Demo'yu nasil ilerletecegim?

**Sayfada yer almasi gerekenler**
1. Uygulamayi ac
2. Runtime panelde aktif pipeline'i goster
3. Test PDF'lerinden birini yukle
4. Normal QA sorusu sor
5. Liste/bolum sorusu sor
6. Tablo veya gorsel bolge sorusu sor
7. Citation ve evidence panelini goster
8. Gerekirse `/use <dosya>` ile belge degistir

**Konusma notu**
Demo'yu teknik gucu gosterecek sekilde ilerletecegim. Once basit QA ile temel akis, sonra section-list sorusu ile deterministic coverage mantigi, en sonda da multimodal veya tablo sorusu ile fark yaratan capability'yi gosterecegim.

---

## Sayfa 19 - Muhtemel Mulakat Sorulari

**Baslik**
Bekledigim teknik sorular ve kisa cevaplar

**Sayfada yer almasi gerekenler**
- Neden sadece vector DB yetmiyor?
- Neden BM25 ekledin?
- Neden section tree kurdun?
- Neden layout ve table stage ayri?
- Halusinasyonu nasil azalttin?
- Neden bu kadar cok backend var?
- Online ve local arasinda nasil secim yapiyorsun?

**Konusma notu**
Bu sorularda ana cevabim su olacak: ben sistemi sadece calisan degil, farkli belge tiplerine dayanikli ve kararlarini aciklayabilen bir yapi olarak kurdum.

---

## Sayfa 20 - Gelecek Calismalar

**Baslik**
Bir sonraki teknik adimlar

**Sayfada yer almasi gerekenler**
- Retrieval iyilestirme:
  - late interaction veya reranker katmani eklemek
  - ozellikle benzer section'lar arasinda daha hassas ayristirma yapmak
- Multimodal iyilestirme:
  - region planning ve table extraction kalitesini artirmak
  - daha iyi form/diagram/technical drawing ayrimi yapmak
- Evaluation iyilestirme:
  - daha genis dokuman setleriyle benchmark havuzu kurmak
  - retrieval ve hallucination metriklerini daha sistematik raporlamak
- Operasyonel iyilestirme:
  - profil bazli hazir demo preset'leri
  - daha guclu loglama ve gozlemlenebilirlik

**Konusma notu**
Burada dikkat ettigim nokta su: gelecekte sistemi buyutmek istedigim alanlar var, ama bugun once temel belge QA problemini guvenilir sekilde cozmek istedim. Bir sonraki mantikli adim retrieval tarafinda reranking, multimodal tarafta daha iyi region/table ayrimi ve test tarafinda daha genis benchmark kapsami olur.

---

## Sayfa 21 - Kapanis

**Baslik**
Sonuc

**Sayfada yer almasi gerekenler**
- Belge yapisini koruyan ingestion
- Hybrid ve query-aware retrieval
- Multimodal/table destekli evidence toplama
- Guardrail'li cevap uretimi
- Demo dostu, konfigure edilebilir UI

**Kapanis cumlesi**
Bu projede hedefim, belgeyi sadece arayan degil; belge yapisini anlayan, farkli veri kaynaklarini birlestiren ve cevabini kaynagi ile savunan bir sistem gelistirmekti.

---

## Sunum Sirasinda Dikkat Edilecekler

- `.env` veya credential dosyalarini ekrana acma.
- Demo oncesi `./run.sh` ile preflight gecmis bir ortam kullan.
- Mode veya backend degistirirsen belgeyi yeniden yukle; bunu kendin soyle, soru gelmesini bekleme.
- Demo'da ilk soruyu kolay, ikinci soruyu "section_list", ucuncu soruyu tablo/layout odakli sec.
- Cevap gelince sadece metni degil, citation ve evidence mantigini da goster.

## Onerilen Demo Sorulari

- `Projenin amaci nedir?`
- `Teslimatlar nelerdir?`
- `Bu tabloda hangi sutunlar var?`
- `Bu sayfadaki form alanlari nelerdir?`

## 30 Saniyelik Acilis Metni

Bu proje, PDF ve gorsel belgeler uzerinde calisan multimodal bir Document QA sistemi. Temel farki, sadece metin aramak yerine belge yapisini, baslik hiyerarsisini, tablo ve layout bilgisini de retrieval'e dahil etmesi. Bu sayede cevaplari kaynaklariyla birlikte ve daha kontrollu sekilde uretebiliyor.
