# TUSAS DOCQA Sayfa Sayfa Konusma Metni

Bu dokuman, `TUSAS-DOCQA (4).pdf` ile hizali 17 sayfalik sunum icin hazirlanmis akici konusma metnidir. Dil, teknik mulakat sunumuna uygun olacak sekilde dogal ve savunulabilir tutulmustur.

---

## Sayfa 1 - Kapak

Merhaba, ben Kemal Utku Mutlu. Bu sunumda TUSAS DOCQA projemi anlatacagim. Bu proje, kurumsal PDF ve gorsel belgeler uzerinde calisan, kaynakli ve daha guvenilir cevap uretebilen bir dokuman soru-cevap sistemi.

Buradaki temel amacim sadece belgeyi aramak degildi. Belge yapisini koruyan, gerektiginde OCR ve multimodal extraction kullanan ve cevabini kaynakla destekleyen bir sistem kurmakti.

---

## Sayfa 2 - Problem Tanimi

Problemi dort ana baslikta ele aliyorum. Birincisi, kurumsal belgelerde layout ve gorsel baglam cok kritik. Tablolar, formlar, baslik yapisi ve sayfa yerlesimi kayboldugunda yalnizca metin cikarimi yeterli olmuyor.

Ikinci olarak OCR tek basina yeterli degil. OCR gerekli bir temel katman ama sadece karakter tanimak, belgenin yapisal ve gorsel anlamini korumaya yetmiyor. Ucuncu problem retrieval tarafinda ortaya cikiyor; sadece text-only veya sadece dense retrieval kullandigimizda liste ve yapisal sorgularda eksik cevap riski olusuyor. Son olarak da grounding ve guvenilirlik problemi var. Kaynak gosterimi ve guardrail olmadan LLM cevabinin dogrulugunu savunmak zor.

---

## Sayfa 3 - Uctan Uca Cozum Mimarisi

Bu slaytta sistemin uctan uca mimarisini ozetliyorum. Belge girdisi PDF, PNG veya JPG olabiliyor. Sonra icerik cikarimi asamasinda PDF text, OCR ve opsiyonel VLM extraction kullaniliyor. Bunun uzerine yapisal analiz geliyor; burada section tree ve parent-child chunking var.

Daha sonra dense ve sparse index birlikte kuruluyor. Retrieval tarafinda Chroma, BM25, RRF ve rerank kullaniyorum. En sonda ise kaynakli cevap uretimi var. Alt tarafta da multimodal yan akis bulunuyor. Yani sayfa veya bolge gorselleri uzerinden layout detection, OCR, VLM ve tablo cikarimi yapilip retrieval tarafina ek kanit olarak besleniyor.

---

## Sayfa 4 - Dokuman Anlama Katmani

Burada odak noktasi, her sayfayi ayni sekilde islememek. Sistem once sayfadaki metin kalitesini degerlendiriyor. Eger sayfa temiz bir text-layer tasiyorsa PDF text kullaniliyor. Eger kalite dusukse OCR veya VLM tabanli extraction devreye giriyor.

Yani burada tek bir extractor yok; bir aday secim mantigi var. Her sayfa icin en yuksek kaliteli extraction adayini secmeye calisiyorum. Bu da farkli belge tiplerinde daha dayanikli bir ingestion davranisi sagliyor.

---

## Sayfa 5 - Hiyerarsik Temsil ve Chunking

Bu projede klasik fixed-size chunking yerine hiyerarsik bir temsil kullandim. Numarali heading yapisini tespit ederek section tree olusturuyorum. Boylece her parca sadece metin degil, bulundugu bolumle birlikte anlam kazaniyor.

Her chunk `section_id`, `parent_id`, `heading_path` ve sayfa araligi gibi metadata tasiyor. Bunun faydasi su: kullanici genelde paragraf sormuyor, bir bolum ya da bir liste soruyor. Bu nedenle belgeyi bolum yapisiyla temsil etmek retrieval kalitesini artiriyor.

---

## Sayfa 6 - Retrieval Nasil Calisiyor?

Retrieval tarafini tek bir arama adimi olarak kurmadim. Once sorguyu siniflandiriyorum. Ornegin `section_list` tipindeki bir sorguyla normal soru-cevap sorgusunu ayni sekilde ele almiyorum.

Ardindan hibrit retrieval geliyor. Dense tarafta anlamsal benzerlik, sparse tarafta ise BM25 ile kelime tabanli sinyal aliyorum. Bu sonuclar RRF ile birlestiriliyor. Son asamada da baslik eslesmeli section secimi ile dogru bolumu ve gerekiyorsa alt agaci getiriyorum.

---

## Sayfa 7 - Section List Sorgu Mekanizmasi

Bu slayt projenin ayirt edici kisimlarindan birini anlatiyor. Liste sorgularinda cevabi tamamen LLM'e birakmiyorum. Once sorguyu `section_list` olarak siniflandiriyorum, sonra baslik eslesmesiyle dogru section'i buluyorum.

Ardindan sadece top-k retrieval yapmak yerine ilgili alt agaci getiriyorum. Beklenen madde sayisini tahmin ederek kapsam kontrolu yapiyorum. Eger yeterli kapsami saglayabiliyorsam sonucu deterministik sekilde uretiyorum; yetmiyorsa ancak o zaman LLM fallback devreye giriyor.

---

## Sayfa 8 - Multimodal Katman

Bu slaytta text pipeline'in uzerine ekledigim multimodal katmani anlatiyorum. Sistem sadece metin chunk'lariyla calismiyor; sayfa ve bolge duzeyinde visual chunk'lar da uretebiliyor. Bu, ozellikle tablo, form veya layout odakli sorularda onemli.

Sayfa chunk tam sayfa gorsel temsilini tutuyor. Bolge chunk ise detector veya heuristic ile bulunan kirpimlari temsil ediyor. Boylece retrieval tarafi yalnizca metin benzerligine degil, gorsel baglama da bakabiliyor.

---

## Sayfa 9 - Tablo ve Layout Isleme

Burada layout detection ile table extraction arasindaki ayrimi vurguluyorum. Once tablo benzeri bolgeler bulunuyor. Yani sistem sayfa uzerinde hangi bolgelerin tablo gibi gorundugunu tespit ediyor.

Sonra sadece ilgili bolgelerde table backend calistiriliyor. Bu backend `docai`, `gemini`, `heuristic` veya `auto` olabiliyor. Elde edilen sonuc yapilandirilmis `table` chunk olarak indeksleniyor. Bu ayrim sayesinde tablo sorularinda daha dogru ve kapsamli evidence toplanabiliyor.

---

## Sayfa 10 - Cevap Uretimi ve Guardrail Katmani

Bu projede odagim sadece guzel cevap uretmek degil, guvenilir cevap uretmek. Bu nedenle sistem, belgede olmayan bilgiyi uretmemek uzere yapilandirildi. Her cevapta kaynak gosterimi zorunlu.

Strict system prompt, kaynak kontrolu, yeniden deneme mantigi ve kapsam kontrolleri bu katmanda yer aliyor. Eger baglamda bilgi yoksa sistem sabit bir fallback veriyor: "Belgede bu bilgi bulunamadi." Yani burada LLM tamamen serbest bir uretici gibi degil, guardrail ile sinirlanmis bir cevaplayici gibi davranıyor.

---

## Sayfa 11 - Moduler Backend Mimarisi

Sistemin bir diger onemli yani moduler backend mimarisi. LLM, embedding, OCR, VLM ve layout bilesenleri runtime sirasinda bagimsiz olarak yapilandirilabiliyor. Cloud ve local secenekleri ayni arayuz altinda yonetilebiliyor.

Bu tasarim sayesinde tek bir servis saglayicisina bagli kalmiyorum. Kalite, maliyet, offline ihtiyaci veya donanim kosullarina gore farkli kombinasyonlar kullanmak mumkun oluyor.

---

## Sayfa 12 - Arayuz ve Kontrol Edilebilirlik

UI tarafini da sadece standart bir chat arayuzu olarak birakmadim. Demo sirasinda OCR, VLM, retrieval ve generation ayarlarini anlik olarak yonetebilecegim bir kontrol yapisi ekledim.

Bu bana su avantaji sagliyor: sistemin hangi modda calistigini gosterebiliyorum, farkli backend kombinasyonlarini deneyebiliyorum ve pipeline davranisini daha seffaf sekilde anlatabiliyorum. Teknik mulakat icin bu kontrol edilebilirlik bence cok degerli.

---

## Sayfa 13 - Operasyonel Dayaniklilik

Bu slaytta algoritmadan cok sistem davranisina odaklaniyorum. Ayni belgeyi tekrar tekrar islememek icin fingerprint tabanli cache invalidation kullaniyorum. Incremental indexing ile sadece degisen belgeler veya yeni chunk'lar yeniden isleniyor.

Ayrica guvenli kapanis, lazy import ve opsiyonel event logging gibi kararlarla sistemi daha dayanikli hale getirdim. Yani proje sadece teknik olarak dogru sonuc vermiyor, operasyonel olarak da daha saglam davranmaya calisiyor.

---

## Sayfa 14 - Test ve Degerlendirme Stratejisi

Burada kaliteyi nasil dogruladigimi gosteriyorum. Hem retrieval odakli metrikler kullandim hem de manuel QA sonucunu takip ettim. Intent Accuracy, Heading Hit, Section Hit ve Evidence Met gibi metrikler retrieval tarafinin ne kadar tutarli oldugunu gosteriyor.

Manuel QA tarafinda da online mod ile local mod arasindaki farki gozlemledim. Ozellikle VLM acik oldugunda, layout-duyarli belgelerde belirgin kalite artisi gordum. Bu nedenle bu slayt, sistemin sadece teorik olarak degil pratikte de nasil davrandigini gostermesi acisindan onemli.

---

## Sayfa 15 - Teknik Kararlar ve Trade-off'lar

Bu projede bazi seyleri bilerek sectim, bazi seyleri de bilerek eklemedim. Text-first mimariyi korudum ve multimodal katmani bunun uzerine additive olarak ekledim. Ayrica deterministic retrieval iyilestirmelerini LLM'in onune koydum.

UI tarafinda Apply/Reset akisi kullandim; cunku runtime degisikliklerinin kontrollu olmasini istedim. Graph RAG veya agentic RAG gibi daha buyuk genislemeleri ise bilincli olarak bu surume eklemedim. Cunku once temel belge QA problemini guvenilir sekilde cozmek daha oncelikliydi.

---

## Sayfa 16 - Gelecek Calismalar

Gelecekte iki farkli ufuk goruyorum. Uzun vadede Graph RAG ve Agentic RAG gibi daha iliskisel ve cok adimli yapilara genisleme potansiyeli var. Ancak bunlar bugunku kapsamdan daha buyuk genislemeler.

Kisa vadede ise daha somut hedefler var: daha guclu reranker'lar, daha iyi bolge ve tablo cikarimi, daha genis benchmark setleri ve daha guclu gozlemlenebilirlik. Yani once mevcut sistemin kalitesini ve olgunlugunu artirmayi hedefliyorum.

---

## Sayfa 17 - Tesekkurler

Sunumun sonuna geldim. Ozetle bu projede belgeyi sadece arayan degil, belge yapisini anlayan, farkli extraction ve retrieval stratejilerini birlestiren ve cevabini kaynaklariyla savunan bir sistem gelistirmeye calistim.

Tesekkur ederim. Sorularinizi memnuniyetle yanitlayabilirim.
