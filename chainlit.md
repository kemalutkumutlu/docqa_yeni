# Chainlit UI Notlari

Bu dosya, uygulamanin kullanici arayuzu davranisini ozetler. Guncel onerilen calistirma sekli `./run.sh` uzerindendir; dogrudan `python -m chainlit run ...` komutunu rutin kullanim icin onerilmez.

## Hizli Kullanim

```bash
./run.sh
```

Varsayilan adres:

```text
http://127.0.0.1:8000
```

## Temel Akis

- Belge olmadan acilir
- PDF/PNG/JPG yuklenebilir
- Belge geldikten sonra ingestion -> indexing -> retrieval -> generation akisi kurulur
- Ayni session icinde birden fazla belge tutulabilir
- `/use <dosya>` ile aktif belge secilebilir
- Sohbet modunda provider cevabi yarim keserse uygulama devami otomatik tamamlamayi dener

Komutlar:

- `/chat`: belge disi sohbet
- `/doc`: belge modu
- `/use <dosya>`: aktif belge sec

## Profil ve Runtime Ayarlari

UI tarafinda provider ve runtime ayarlari vardir:

- Profil secimi:
  - `Gemini`
  - `OpenAI`
  - `Local`
  - `Extractive`
- Runtime ayarlari:
  - `Embedding Model`
  - `Embedding Device`
  - `VLM Mode`
  - `VLM Provider`
  - `VLM Max Pages`

Guncel embedding secenekleri:

- `gemini-embedding-001`
- `auto`
- `intfloat/multilingual-e5-small`
- `intfloat/multilingual-e5-base`

## Demo Icin Oneri

UI ayarlari olsa da demo sirasinda bunlari degistirmemen tavsiye edilir.

Sebep:

- embedding modeli degisince index yeniden olusur
- provider degisince davranis farkli auth yoluna kayabilir
- VLM ayarlari yeni yuklemelerde fark yaratir

Mülakat profili icin en guvenli yol:

- `.env` icinde sabit Vertex/Gemini profilini tut
- `./run.sh` ile baslat
- UI ayarlarini degistirme

## Belge Durumu Paneli

Sidebar'da tipik olarak su bilgiler gorunur:

- aktif mod
- LLM provider
- embedding model/device
- VLM provider/mode/max pages
- aktif belge
- yuklu belgeler

## Gecmis Sohbetler

UI thread bazli hafiza tutar.

- Gecmis thread mesajlari varsayilan olarak `DATA_DIR/thread_history/*.json` altina yazilir
- Sidebar'dan eski bir sohbet secildiginde, icerik RAM'de yoksa diskten geri yuklenir
- Istersen dizini `THREAD_HISTORY_DIR` ile override edebilirsin

## Not

Bu UI, case-study ve demo odakli tasarlanmistir. Operational/production admin paneli degildir; runtime override'larin varligi daha cok gelistirme kolayligi icindir.
