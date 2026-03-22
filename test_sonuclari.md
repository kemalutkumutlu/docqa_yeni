# TEST SONUCLARI

Bu dokuman, `Case_Study_20260205.pdf` ve `CV-ornek-muhendis.pdf` icin kaydedilen manuel konusma akislarinin sonuc raporudur.

Not: Bu kayitlarda odak UI degil, cevap dogrulugu ve belgeye sadakattir. Online mode cevaplari dogru referans kabul edilerek local mode ayni soru setleriyle ayrica test edilmistir.

## Online Test Baglami

- Test tipi: Manuel sohbet akisi / kabul testi
- Belgeler: `Case_Study_20260205.pdf`, `CV-ornek-muhendis.pdf`
- Toplam soru sayisi: `71`
- Toplam sayfa sayisi: `6`
- Toplam chunk sayisi: `46`
- Toplam indekslenen chunk: `46`
- `Case_Study_20260205.pdf`: `55` soru, `4` sayfa, `23` chunk
- `CV-ornek-muhendis.pdf`: `16` soru, `2` sayfa, `23` chunk
- Degerlendirme odagi: Verilen cevabin dogrulugu

## Online Test Ayarlari

- `processing_mode=multimodal`
- `ocr=off`
- `vlm_mode=force`
- `llm=gemini`
- `pdf_text=docling`
- `chunk_level=page`
- `vlm_provider=gemini`

## Online Genel Sonuc

- Arsivlenen akista toplam `71` soru soruldu.
- Belge ici sorular beklenen bilgilerle dogru cevaplandi.
- Ayni bilginin farkli sekilde soruldugu sorularda sistem tutarli kaldi.
- Belge disi veya belgede bulunmayan bilgi sorularinda sistem uydurma cevap vermedi.
- Belgede acikca serbest birakilan konularda sistem dogru sekilde `zorunlu degil`, `belirtilmemis` veya `size birakilmis` cercevesini korudu.
- Sayisal ozet: `71/71` soru referans dogru cevap olarak kabul edildi.
- Sonuc: Online manuel QA akisi, cevap dogrulugu acisindan `PASS`.

## Online Dogrulanan Alanlar

| Alan | Ornekler | Gozlenen sonuc | Durum |
| --- | --- | --- | --- |
| Pozisyon ve sure bilgileri | Pozisyon, teslim suresi, tahmini calisma suresi | `Mid-Senior Yazilim Gelistirici (AI/ML Odakli)`, `7 gun`, `25-35 saat` bilgileri dogru verildi | PASS |
| Proje ozeti ve fonksiyonel gereksinimler | Proje ozeti, temel islevler, fonksiyonel gereksinimler | Belge yukleme, metin cikarimi, soru-cevap, dogruluk ve kullanilabilirlik eksiksiz aktarildi | PASS |
| Teknik yaklasim | Teknik yaklasim, MVP beklentisi, problemi nasil cozelim | Teknoloji seciminin adaya birakildigi ve islevsel MVP beklentisi dogru anlatildi | PASS |
| Teslimatlar | DEVLOG, TESTING, README, demo video, kaynak kod | Beklenen teslimatlar eksiksiz listelendi | PASS |
| DEVLOG ve TESTING kapsami | DEVLOG'da beklenen sorular, TESTING dosyasinin amaci | Iki teslimatin amaci, kapsam ve beklentileri dogru aciklandi | PASS |
| Teslimat bilgileri ve sonraki adim | Teslim yontemi, demo video suresi, teknik mulakat | GitHub repo linki, 3-5 dakika video ve olumlu degerlendirme sonrasi teknik mulakat bilgileri dogru verildi | PASS |
| Notlar ve teknik serbestlik alani | LLM serbest mi, Docker zorunlu mu, dil zorunlu mu, mimari oneriliyor mu | Belirtilmeyen konularda `zorunlu degil` veya `adaya birakilmis` cevabi korundu | PASS |
| Dosya ve dil beklentileri | Hangi `.md` dosyalari, desteklenen diller, desteklenen dosya formatlari | README, DEVLOG, TESTING; Turkce/Ingilizce; PDF/JPG/PNG bilgileri dogru verildi | PASS |
| Belge disi sorular | `Araba kac beygirdir`, `RTOS nedir`, aday bilgileri, test coverage, bulut saglayicisi | Sistem `Belgede bu bilgi bulunamadi.` cizgisini korudu | PASS |
| CV kisisel bilgiler | Isim, adres, dogum yeri, telefon, ilgi alanlari | Kisisel alanlar dogru cekildi ve listelendi | PASS |
| CV is deneyimi | Son is deneyimi, sirket/rol ayrimi, yanlis oncul sorulari | Is deneyimi sirasi ve rol ayrimi dogru cevaplandi | PASS |
| CV egitim ve beceriler | Egitim bilgisi, mezuniyet tarihi, beceriler, kurslar, referanslar | Egitim, kurs ve referans bilgileri dogru cevaplandi | PASS |

## Online Davranis Notlari

- Paraphrase dayanikliligi iyiydi. Ayni konu farkli sekilde soruldugunda cevap anlami korunarak yeniden uretildi.
- Listeleme sorularinda sistem maddeleri eksiksiz vermeyi basardi. Ozellikle teslimatlar ve fonksiyonel gereksinimler tarafinda kapsam korundu.
- Yanlis oncul veya belgede net olmayan sorularda sistem duzeltici cevap verdi. Ornek: zorunluluk olmayan teknolojiler icin `Hayir` ile baslayan netlestirici cevaplar.
- Belge ici sorularin debug bloklarinda citation sayisi gorunuyordu; belge disi sorularda citation uretmeyip no-answer vermesi beklenen davranisti.

## Local Test Baglami

- Test tipi: Manuel sohbet akisi / kabul testi
- Belgeler: `Case_Study_20260205.pdf`, `CV-ornek-muhendis.pdf`
- Toplam soru sayisi: `71`
- Toplam sayfa sayisi: `6`
- Toplam chunk sayisi: `92`
- Toplam indekslenen chunk: `92`
- `Case_Study_20260205.pdf`: `55` soru, `4` sayfa, `27` chunk
- `CV-ornek-muhendis.pdf`: `16` soru, `2` sayfa, `65` chunk
- Degerlendirme temeli: Online mode cevaplari dogru referans kabul edildi

## Local Test Ayarlari

- `preset=custom`
- `processing_mode=multimodal`
- `ocr=off`
- `ocr_backend=docai`
- `multimodal_answer_gen=auto`
- `visual_chunk_level=page`
- `visual_region_source=detector`
- `visual_detector_backend=docai`
- `table_structure=off`
- `table_backend=auto`
- `llm=local`
- `llm_model=qwen2.5:7b`
- `embedding_model=intfloat/multilingual-e5-base`
- `embedding_device=auto`
- `vlm_provider=gemini`
- `vlm_mode=off`
- `vlm_max_pages=25`

## Local Genel Sonuc

- Online mode ile ayni soru setleri local mode uzerinde tekrar soruldu.
- `Case_Study_20260205.pdf` sorularinda genel belge kapsami korunmakla birlikte bazi sorularda online moda gore dogruluk dususu gozlendi.
- `CV-ornek-muhendis.pdf` sorularinda isim, is deneyimi, telefon ve referans gibi alanlarda online moda gore belirgin hata artisi gozlendi.
- Local mode, dogru cevap verebildigi sorularda belge referansi uretebildi; ancak ayni soru setinde online moda gore daha fazla `bulunamadi`, eksik veya yanlis cevap olustu.
- Sayisal ozet:
  - `Case_Study_20260205.pdf`: `35/55` dogru, `20/55` eksik veya yanlis
  - `CV-ornek-muhendis.pdf`: `4/16` dogru, `12/16` eksik veya yanlis
  - Toplam: `39/71` dogru, `32/71` eksik veya yanlis
- Sonuc: Local mode ayni test setinde `ONLINE MODEDAN DAHA ZAYIF`.

## Local Davranis Notlari

- `Case_Study_20260205.pdf` tarafinda bazi temel sorular dogru cevaplandi, ancak `7 adet fonksiyonel gereksinim mi var` gibi sorularda yanlis sayim uretildi.
- `LLM kullanmak yasak midir`, `README.md yeterli midir` gibi sorularda belge icerigine aykiri veya eksik yorumlar olustu.
- `CV-ornek-muhendis.pdf` tarafinda adres ve dogum yeri gibi bazi alanlar bulunabildi, ancak isim, is deneyimi, telefon ve referans gibi kritik alanlarda hatalar olustu.
- Local mode sonuclarinda ayni belge icin daha fazla chunk olusmasina ragmen cevap kalitesi online mode seviyesine ulasmadi.
- Bu farkin en bariz olasi nedenlerinden biri VLM etkisidir. Online testte `vlm_mode=force` ve `vlm_provider=gemini` kullanilirken, local testte `vlm_mode=off` durumundaydi.
- Sayisal sonuclar da bu etkiyi desteklemektedir: `Case_Study_20260205.pdf` tarafinda local dogruluk `35/55`, `CV-ornek-muhendis.pdf` tarafinda ise `4/16` seviyesinde kalmistir.
- Ozellikle CV gibi alan bazli, gorsel yerlesim ve bolgesel ayristirma gerektiren belgelerde VLM'in kapali olmasi; isim, telefon, is deneyimi ve referans gibi kritik alanlarin dogru cekilmesini belirgin bicimde zayiflatmis olabilir.

## Son Karar

Online mode, bu manuel kabul testlerinde referans ve dogru sonuc veren mod olarak degerlendirilmistir. Local mode ise ayni soru seti uzerinde calismis, ancak ozellikle yorumlama, alan cikarimi ve yanlis oncul duzeltme basliklarinda online mode performansinin gerisinde kalmistir.
