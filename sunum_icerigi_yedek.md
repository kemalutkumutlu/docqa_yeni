# TUSAS DOCQA Teknik Mulakat Sunum Icerigi

Bu dokuman, mevcut kod tabani ile uyumlu olacak sekilde guncellenmis teknik sunum iskeletidir. Icerik; ingestion, section-tree, hybrid retrieval, multimodal capability, guardrail ve runtime ayarlari dahil gercek uygulama davranisina gore yazilmistir.

## Sunum Stratejisi

- Sunumu 12-15 dakika araliginda bitir.
- Once problemi, sonra mimariyi, sonra kritik teknik kararlarini anlat.
- Demo'da sadece "cevap verdi" demek yetmez; evidence, citation ve aktif pipeline'i de goster.
- Metrik veya profil bilgisi vereceksen sunum gunu script ve `.env` ile son kez dogrula.

---

## Sayfa 1 - Kapak

**Baslik**
TUSAS DOCQA: Hiyerarsik, Multimodal ve Guardrail Destekli Belge Soru-Cevap Sistemi

**Alt Baslik**
PDF ve gorsel belgeler icin structure-aware retrieval, OCR/VLM destekli extraction ve kaynakli cevap uretimi

**Sayfada yer almasi gerekenler**
- Proje adi
- Isim
- "Technical Interview Presentation"
- Kisa slogan: "Belgeden cevap, kaynakla birlikte"

**Konusma notu**
Bu projede hedefim sadece belgeyi arayan bir sistem kurmak degildi. Amacim, belge yapisini koruyan, gerektiginde OCR ve gorsel extraction kullanan, retrieval'i sorgu tipine gore degistiren ve cevabi kaynaklariyla birlikte ureten guvenilir bir Document QA sistemi gelistirmekti.

---

## Sayfa 2 - Problem Tanimi

**Baslik**
Neyi cozuyorum?

**Sayfada yer almasi gerekenler**
- Kurumsal PDF'lerde metin katmani her zaman temiz degil
- Scan/image PDF'lerde OCR gerekiyor
- Tablolar, formlar ve layout kritik bilgi tasiyor
- Kullanici cogu zaman paragraf degil, bolum veya liste soruyor
- Sadece vector search ile eksiksiz cevap almak zor
- Halusinasyon ve kaynak gosterimi kritik risk

**Konusma notu**
Klasik RAG yaklasimi "chunk + embedding + LLM" seviyesinde iyi bir baslangic ama gercek belge problemini tam cozmez. Cunku sorun sadece anlamsal arama degil; extraction kalitesi, baslik hiyerarsisi, tablo yapisi, layout ve cevap guvenilirligi de isin icinde.

---

## Sayfa 3 - Cozum Ozeti

**Baslik**
Sistemin kisa ozeti

**Sayfada yer almasi gerekenler**
- Girdi: PDF, PNG, JPG
- Ingestion: PyMuPDF / Docling / OCR / VLM adaylari arasindan en iyi metin secimi
- Yapisal analiz: section tree + parent/child chunking
- Retrieval: Dense + Sparse + RRF + embedding rerank
- Multimodal: page/region visual chunk ve table chunk
- Generation: Gemini / OpenAI / Local / Extractive
- Cikti: citation'li ve guardrail'li cevap

**Konusma notu**
Buradaki temel kararim su oldu: Sistemi moduler kurdum. Boylece ayni uygulama hem online hem local, hem classic hem multimodal, hem de farkli OCR, embedding ve generation backend'leri ile calisabiliyor.

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
-> Guardrailed Answer Generation
```

**Ek kutu**
Multimodal modda:
```text
Visual Assets
-> Page / Region Planning
-> OCR / VLM / Table Structure
-> Visual + Text Evidence
```

**Konusma notu**
Text-first cekirdegi korudum, multimodal katmani buna ek olarak yerlestirdim. Yani sistem yeni capability kazanirken mevcut classic akis bozulmuyor. Bu, projeyi hem daha guvenli hem daha bakimi kolay hale getirdi.

---

## Sayfa 5 - Ingestion ve Extraction Tasarimi

**Baslik**
Belgeyi once dogru okumak zorundayim

**Sayfada yer almasi gerekenler**
- PDF text backend: `pymupdf`, `docling`, `auto`, `smart`
- OCR backend: `docai`, `paddle_vl`, `paddle`, `tesseract_legacy`, `smart`
- VLM mode: `off`, `auto`, `smart`, `force`
- PyMuPDF tarafinda multi-column okuma sirasi duzeltiliyor
- Docling butun PDF icin alternatif text adayi uretiyor
- Dusuk kalite metinde OCR devreye giriyor
- VLM extract-only calisiyor; OCR metni VLM icin grounding olarak verilebiliyor

**Konusma notu**
Burada tek bir extractor'a bagimli kalmadim. Cunku bir sayfa native text olabilir, digeri scan olabilir, baska bir sayfa da layout olarak karmasik olabilir. Bu yuzden her sayfada en iyi adayi secen bir ingestion mantigi kurdum. VLM'i de serbest QA icin degil, extract-only amacla kullandim; layout authority goruntu, karakter dogrulama kaynagi ise OCR.

---

## Sayfa 6 - Yapisal Temsil ve Chunking

**Baslik**
Neden hiyerarsik chunking kullandim?

**Sayfada yer almasi gerekenler**
- Numbered heading detection: `2.`, `4.1`, `A.4.1`
- Repeating header/footer temizligi
- Gerektiginde sinirli unkeyed heading fallback
- Section tree olusumu
- Parent chunk: tam bolum
- Child chunk: paragraf-hizali, token-bazli bolumleme (`tiktoken cl100k_base`)
- Metadata: `section_id`, `parent_id`, `heading_path`, `page_start`, `page_end`

**Konusma notu**
Duzenli fixed-size chunking yerine section tree kullandim cunku kullanici genelde "paragraf" degil "bolum" soruyor. Parent chunk tam kapsami, child chunk ise retrieval hassasiyetini sagliyor. Bu hibrit yapi ozellikle liste sorularinda ve section-based sorularda cok faydali oldu.

---

## Sayfa 7 - Retrieval Nasil Calisiyor?

**Baslik**
Hybrid retrieval ve query routing

**Sayfada yer almasi gerekenler**
- Dense retrieval: embedding tabanli anlamsal arama
- Sparse retrieval: BM25 ile kelime bazli arama
- BM25 tarafi ozellikle child/table chunk'larda calisiyor
- BM25 tarafinda Turkce/Ing. dostu tokenizer ve 3-gram destegi
- Turkce sorguda morphological expansion
- Ingilizce sorguda gerekirse query expansion
- RRF ile birlestirme
- Embedding rerank ile son siralama
- Query classification:
  - `section_list`
  - `multi_section`
  - `normal_qa`

**Konusma notu**
Tum sorulari ayni retrieval yoluna sokmadim. Cunku "Teslimatlar nelerdir?" ile "Teslim suresi nedir?" ayni problem degil. Ayrica sadece dense ya da sadece sparse kullanmak yerine ikisini birlestirdim. Sparse exact terimleri, dense anlamsal benzerligi tasiyor; RRF ise bu iki sinyalin failure mode'larini dengeliyor.

---

## Sayfa 8 - Section List Sorularini Nasil Guvenceye Aldim?

**Baslik**
Eksiksiz listeleme icin ozel mekanizma

**Sayfada yer almasi gerekenler**
- `section_list` sorgulari once siniflandiriliyor
- En uygun bolum heading-aware seciliyor
- Topic-heading confidence guard var
- Sadece top-k degil, tum section + subtree fetch yapiliyor
- TOC false-positive guard var
- Gerekirse visual fallback kullaniliyor
- Coverage heuristic ile beklenen madde sayisi hesaplanabiliyor
- Uygunsa deterministic liste render ediliyor
- Yetmezse LLM fallback devreye giriyor

**Konusma notu**
Bu kisim projede fark yaratan teknik kararlardan biri. Klasik RAG'de LLM listedeki maddelerin bir kismini atlayabiliyor. Ben burada once section'i dogru secmeye, sonra alt agaci eksiksiz getirmeye, sonra da mumkunse cevabi deterministik uretmeye odaklandim. Yani kritik liste sorularinda uretici modeli degil, veriyi otorite yaptim.

---

## Sayfa 9 - Multimodal Katman

**Baslik**
Metin yetmediginde ne oluyor?

**Sayfada yer almasi gerekenler**
- Processing mode: `classic`, `multimodal`, `smart`
- Visual chunk level: `page` veya `region`
- Region source: `heuristic` veya `detector`
- Detector backend: `none`, `sidecar`, `docai`, `docling`
- Query tablo/form/layout sinyali tasiyorsa region-aware skor bonusu uygulanabiliyor
- Gemini tarafinda multimodal answer generation `off/auto/on` olarak kontrol edilebiliyor

**Konusma notu**
Multimodal katmani sadece "goruntu de ekleyelim" seviyesinde kurmadim. Region bazli crop, detector secimi, region metadata'si ve retrieval tarafinda visual-aware scoring ekledim. Boylece tablo, form, checkbox, layout veya sayfadaki belirli bir bolgeyle ilgili sorularda daha anlamli evidence toplanabiliyor.

---

## Sayfa 10 - Tablo ve Layout Neden Ayri Stage?

**Baslik**
Table structure ve layout neden ayri problem?

**Sayfada yer almasi gerekenler**
- Layout detection ayri, OCR ayri, table parsing ayri
- Table stage: `TABLE_STRUCTURE_ENABLED` (acma/kapama) + `smart` flag
- Table backend:
  - `docai`
  - `gemini`
  - `heuristic`
  - `auto`
- Smart modda sadece OCR/VLM sayfalarinda calistirilabiliyor
- Sonuc `table` chunk olarak indekse giriyor

**Konusma notu**
Bu ayrimi bilerek yaptim cunku OCR metin verir ama tabloyu tablo yapan satir-sutun iliskisini her zaman koruyamaz. Layout detector bolgeyi buluyor, table stage ise yapisal satir-sutun bilgisini cikarmaya calisiyor. Ozellikle scan veya gorsel agirlikli belgelerde bu ayrim cevabin kalitesini ciddi artiriyor.

---

## Sayfa 11 - Cevap Uretimi ve Guardrail'ler

**Baslik**
LLM kullaniyorum ama kontrolsuz birakmiyorum

**Sayfada yer almasi gerekenler**
- Siki system prompt
- Baglam disi bilgi yasak
- Baglam yoksa sabit fallback: `Belgede bu bilgi bulunamadi.`
- Yanlis oncullu sorguda: `Hayir, ...` ile duzeltici cevap
- Her bilgi cumlesinde citation zorunlu
- Weak evidence durumunda grounding check ile bos context fallback
- Deterministic section-list path
- Kanit paneli: yalnizca cevapta atifta bulunulan sayfalarin evidence ozeti
- Non-streaming:
  - citation retry
  - coverage retry
  - coverage warning
- Streaming:
  - token geri alma yok
  - silent citation duzeltme passi
  - coverage warning

**Konusma notu**
Buradaki hedefim "guzel cevap" degil, "guvenilir cevap" oldu. Bu yuzden cevabin kaynaksiz gelmesini kabul etmiyorum. Ayrica streaming ve non-streaming akislari ayni gibi anlatmiyorum; non-streaming daha agresif duzeltme yapabilirken, streaming'de yayinlanan token geri alinmadigi icin daha farkli bir guardrail davranisi var.

---

## Sayfa 12 - Neden Bu Kadar Konfigure Edilebilir?

**Baslik**
Moduler backend mimarisi

**Sayfada yer almasi gerekenler**
- LLM provider: `gemini`, `openai`, `local`, `none`
- Embedding: Gemini veya local sentence-transformers
- OCR: cloud ya da local
- VLM: cloud ya da local
- Layout detector: `none`, `sidecar`, `docai`, `docling`
- Detector basarisiz olursa heuristic fallback
- Runtime'da bircok ayar UI uzerinden degistirilebiliyor
- Embedding degisirse indeks yeniden kuruluyor
- OCR/VLM/pdf-text degisiklikleri yeni belge yuklemelerinde etkili oluyor

**Konusma notu**
Projenin ana hedeflerinden biri tek bir ortama bagimli kalmamakti. Kalite, maliyet, offline ihtiyaci ve donanim imkanina gore farkli pipeline'lar kurulabiliyor. Bu esneklik teknik olarak maliyetliydi ama urun ve demo acisindan cok degerli oldu.

---

## Sayfa 13 - UI ve Demo Deneyimi

**Baslik**
Chainlit tabanli ama standart bir chat arayuzu degil

**Sayfada yer almasi gerekenler**
- Sol sohbet/gecmis ve belge baglami paneli
- Sag custom runtime settings paneli
- `Basic` ve `Advanced` sekmeleri
- Runtime preset'ler:
  - `online_best`
  - `hybrid_best`
  - `local_best`
  - `fast`
- `Apply` / `Reset` akisi
- Ozet kartlari:
  - `Current Draft`
  - `Applied Pipeline`
  - `Fallback Notes`
  - `Document Context`
- Multi-doc akista aktif belge secimi ve dosya adindan routing
- Komutlar:
  - `/chat`
  - `/doc`
  - `/use <dosya>`

**Konusma notu**
UI'yi sadece soru-cevap ekrani olarak birakmadim. Teknik mulakatta gostermeyi kolaylastiran bir runtime panel ekledim. Boylece ayni uygulamada hem demo yapabiliyor hem de hangi pipeline'in aktif oldugunu, hangi fallback'in neden devreye girdigini seffaf sekilde gosterebiliyorum.

---

## Sayfa 14 - Coklu Belge ve Operasyonel Dayaniklilik

**Baslik**
Sistemi sadece dogru degil, dayanikli da yapmaya calistim

**Sayfada yer almasi gerekenler**
- Document cache ile ayni dosya tekrar islenmiyor
- Fingerprint ile config degisirse cache invalidation oluyor
- Incremental indexing ile sadece yeni chunk embed ediliyor
- Persistent Chroma + BM25
- Embedding dimension bazli collection isimlendirme
- Multi-document session destegi
- Aktif belge secimi ve partial filename routing
- Optional event logging
- Graceful shutdown / lazy import / fallback-first davranis

**Konusma notu**
Kod tabaninda sadece algoritma degil, operasyonel davranis da onemliydi. Ayni belgeyi tekrar tekrar islememek, yeni dosya geldiginde tum indexi bastan embed etmemek ve coklu belge oturumlarinda yanlis belgeye gitmemek bu yuzden oncelikliydi.

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
  - evidence met
  - latency
- Hallucination test
- Smoke suite
- Folder suite ile coklu PDF denemeleri
- Ek manuel QA ozeti (`test_sonuclari.md`)
  - Belgeler: `Case_Study_20260205.pdf` + `CV-ornek-muhendis.pdf`
  - Online mode: `71/71` soru dogru referans kabul edildi
  - Local mode: ayni soru setinde `39/71` dogru; online moddan belirgin zayif
  - Paraphrase tutarliligi, liste coverage ve belge disi no-answer davranisi ayrica gozlemlendi
  - Gozlenen en belirgin farklardan biri VLM etkisiydi; online testte `vlm_mode=force` iken local testte `vlm_mode=off` oldugu icin ozellikle CV gibi layout-duyarli belgelerde dogruluk belirgin dustu

**Onerilen not**
Bu sayfada sabit rakam yazacaksan, demo oncesi `scripts/eval_retrieval.py` ve ilgili kabul script'lerini son bir kez calistirip tabloyu guncelle. Sabit metrik yazip stale bir sonuc gostermekten kacinin.

**Konusma notu**
Test stratejim sadece unit test degil. Bu tarz bir RAG sistemde ingestion, retrieval ve generation birlikte degerlendirilmelidir. Bu yuzden bir yandan script tabanli kabul kapilari kurdum, bir yandan da manuel soru-cevap akislarini arsivledim. Boylece hem tekrar uretilebilir metrikleri hem de gercek kullaniciya benzeyen QA davranisini ayni slaytta gosterebiliyorum.

---

## Sayfa 16 - Mevcut Demo Profili

**Baslik**
Bugun gosterecegim aktif konfigurasyon

**Sayfada yer almasi gerekenler**
- `LLM_PROVIDER=gemini`
- `VERTEX_ENABLED=1`
- `GEMINI_MODEL=gemini-3.1-pro-preview`
- `GEMINI_FALLBACK_MODEL=gemini-2.5-pro`
- `EMBEDDING_MODEL=gemini-embedding-2-preview`
- `EMBEDDING_VERTEX_ENABLED=0`
- `DOC_PROCESSING_MODE=multimodal`
- `PDF_TEXT_BACKEND=docling`
- `VISUAL_CHUNK_LEVEL=page`
- `VISUAL_REGION_SOURCE=detector`
- `VISUAL_DETECTOR_BACKEND=docai`
- `OCR_ENABLED=0`
- `TABLE_STRUCTURE_ENABLED=0`
- `TABLE_STRUCTURE_BACKEND=auto`
- `VLM_PROVIDER=gemini`
- `VLM_MODE=force`
- Pratik demo odagi: page-level multimodal akisi; region/detector capability sistemde mevcut ama bu profilde ana gosterim noktasi degil

**Konusma notu**
Demo icin kaliteyi one cikan bir profil kullaniyorum. LLM ve VLM tarafinda Vertex yolu acik, embedding tarafinda ise `EMBEDDING_VERTEX_ENABLED=0` ile AI Studio key tabanli yol kullaniliyor. `PDF_TEXT_BACKEND=docling` oldugu icin metin extraction'da Docling tercih ediliyor; OCR kapali ve table stage kapali cunku bu demoda Docling'den gelen yapiyi kullaniyorum. Gorsel tarafta env'de detector ayarlari tanimli olsa da bu profilin pratikteki ana odagi page-level multimodal akis; region-level detector yetenegi sistemde var ama bu sunumda ana mesaj olarak onu one cikarmiyorum.

---

## Sayfa 17 - Teknik Kararlar ve Trade-off'lar

**Baslik**
Neleri bilerek sectim, nelerden bilerek vazgectim?

**Sayfada yer almasi gerekenler**
- Text-first cekirdegi korudum, multimodal'i additive ekledim
- Deterministic retrieval iyilestirmelerini LLM'den once koydum
- Moduler backend mimarisi sectim, tek bir servis yoluna kilitlenmedim
- UI'da tam reactive config editor yerine `Apply/Reset` modeli kullandim
- Kritik listelerde generative degil deterministic yol kullandim
- Bugun Graph RAG veya agentic RAG yok; bilincli olarak eklenmedi

**Konusma notu**
Her capability'yi eklemek yerine, belge QA problemini dogrudan cozen ve teknik olarak savunulabilir katmanlara odaklandim. Bu da projeyi gereksizce karmasiklastirmadan guclendirdi. "Neden yapmadin?" sorusuna cevabim da bu: Onceligim once temel belge QA'yi guvenilir sekilde cozmekti.

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
6. Tablo veya layout odakli bir soru sor
7. Citation ve evidence mantigini goster
8. Gerekirse `/use <dosya>` ile aktif belgeyi degistir

**Konusma notu**
Demo'yu teknik gucu gosterecek sekilde ilerletecegim. Once basit QA ile temel akis, sonra section-list sorusu ile deterministic coverage mantigi, en sonda da tablo/layout odakli soru ile multimodal farki gosterecegim.

---

## Sayfa 19 - Muhtemel Mulakat Sorulari

**Baslik**
Bekledigim teknik sorular ve kisa cevaplar

**Sayfada yer almasi gerekenler**
- Neden sadece vector DB yetmiyor?
- Neden BM25 ekledin?
- Neden section tree kurdun?
- Neden OCR, Docling ve VLM birlikte var?
- Neden layout ve table stage ayri?
- Halusinasyonu nasil azalttin?
- Neden bu kadar cok backend var?

**Konusma notu**
Bu sorularda ana cevabim su olacak: Sistemi sadece calisan degil, farkli belge tiplerine dayanikli, sorgu tipine gore davranan ve kararlarini aciklayabilen bir yapi olarak kurdum. Yani mimari tercihlerin hepsi belirli bir failure mode'u azaltmak icin var.

---

## Sayfa 20 - Gelecek Calismalar

**Baslik**
Bir sonraki teknik adimlar

**Sayfada yer almasi gerekenler**

**Kisa Vadeli Hedefler**
- Retrieval iyilestirme:
  - cross-encoder veya ColBERT benzeri ek reranker
  - benzer section'lar arasinda daha hassas ayristirma
- Multimodal iyilestirme:
  - region planning ve table extraction kalitesini artirma
  - form/diagram/technical drawing ayrimini guclendirme
- Evaluation iyilestirme:
  - daha genis benchmark havuzu
  - retrieval ve hallucination metriklerini sistematik raporlama
- Operasyonel iyilestirme:
  - daha fazla preset/profil
  - yapilandirilmis loglama ve monitoring

**Uzun Vadeli Opsiyonlar**
- Graph RAG
- Agentic RAG

**Konusma notu**
Burada dikkat ettigim nokta su: Bugun sistemde olmayan seyleri "varmis" gibi anlatmiyorum. Uzun vadede Graph RAG veya agentic RAG gibi yonler dusunulebilir; ama bugunku odagim once temel belge QA problemini guvenilir ve savunulabilir sekilde cozmeye yonelikti.

---

## Sayfa 21 - Kapanis

**Baslik**
Sonuc

**Sayfada yer almasi gerekenler**
- Belge yapisini koruyan ingestion
- Query-aware ve hybrid retrieval
- Multimodal/table destekli evidence toplama
- Guardrail'li cevap uretimi
- Demo dostu ve konfigure edilebilir UI

**Kapanis cumlesi**
Bu projede hedefim, belgeyi sadece arayan degil; belge yapisini anlayan, farkli extraction ve retrieval stratejilerini birlestiren ve cevabini kaynagi ile savunan bir sistem gelistirmekti.

---

## Sunum Sirasinda Dikkat Edilecekler

- `.env`, API key veya credential dosyalarini ekrana acma.
- Demo oncesi `./run.sh` ile preflight gecmis bir ortam kullan.
- Runtime'da OCR, VLM, pdf text backend veya processing mode degistirirsen yeni belge yukle; bunu kendin belirt.
- Ilk soruyu kolay, ikinci soruyu `section_list`, ucuncu soruyu tablo/layout odakli sec.
- Cevap geldiginde sadece metni degil, citation ve evidence mantigini da goster.
- Metrik slaydi kullaniyorsan rakamlari demo gunu guncelle.

## Onerilen Demo Sorulari

- `Projenin amaci nedir?`
- `Teslimatlar nelerdir?`
- `Bu tabloda hangi satirlar veya sutunlar var?`
- `Bu sayfadaki form alanlari nelerdir?`
- `X bolumu ile Y bolumu arasindaki fark nedir?`

## 30 Saniyelik Acilis Metni

Bu proje, PDF ve gorsel belgeler uzerinde calisan hiyerarsik ve multimodal bir Document QA sistemi. Temel farki, sadece metin aramak yerine belge yapisini, baslik hiyerarsisini, tablo ve layout bilgisini de retrieval'e dahil etmesi. Bu sayede cevaplari daha kontrollu, daha izlenebilir ve kaynaklariyla birlikte uretebiliyor.
