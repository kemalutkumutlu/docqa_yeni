# Chainlit UI Notlari

Bu dosya, uygulamanin bugunku UI davranisini ozetler. Amaç "tasarladigimiz ideal panel" degil, su an gercekten calisan davranisi anlatmaktir.

## Baslatma

Tek onerilen yol:

```bash
./run.sh
```

Varsayilan adres:

```text
http://127.0.0.1:8000
```

## Temel Akis

- Uygulama belge olmadan acilir
- PDF / PNG / JPG yuklenebilir
- Belge geldikten sonra ingestion -> indexing -> retrieval -> generation kurulur
- Ayni session icinde birden fazla belge tutulabilir
- `/use <dosya>` ile aktif belge secilebilir

Komutlar:

- `/chat`: belge disi sohbet
- `/doc`: belge modu
- `/use <dosya>`: aktif belge sec

## Sol Sidebar

Projede varsayilan Chainlit gecmisi yerine ek bir ozel sidebar mantigi vardir.

- `public/history_sidebar.js` desktop'ta mini sohbet gecmisi paneli ekler
- thread secimi tarayici `localStorage` ile tutulur
- native Chainlit thread history acik degilse bu fallback davranis calisir
- panel ustunde TUSAS logo + kucuk marka alani bulunur

## Settings Panel

UI'da iki ayar sekmesi vardir:

- `Basic`
- `Advanced`

### Basic

Gunluk kullanim icin ust seviye alanlar:

- `Runtime Preset`
- `Processing Mode`
- `OCR`
- `LLM Provider`
- gerekiyorsa `Generation Model`

### Advanced

Detayli backend ayarlari:

- `Embedding Model`
- `Embedding Device`
- `VLM Mode`
- `OCR Backend`
- `Visual Chunk Level`
- `Table Structure`
- `Multimodal Answer Generation`
- `Visual Region Source`
- `Visual Detector Backend`
- `Table Structure Backend`
- `VLM Provider`
- `VLM Max Pages`

Panelin ust kisminda ozet kartlari bulunur:

- `Current Draft`
- `Applied Pipeline`
- `Fallback Notes`
- `Document Context`

## Ayar Bagimliliklari

Advanced alanda alanlar kaybolmaz; desteklenmeyen durumda kilitlenir.

Ornekler:

- `OCR=off` ise `OCR Backend` disabled olur
- `VLM Mode=off` ise `VLM Provider` ve `VLM Max Pages` disabled olur
- `classic` modda visual/table zinciri disabled olur
- `region` degilse detector kaynaklari disabled olur

Bu tercih bilincli:

- kullanici hangi alanin var oldugunu gorsun
- ama desteklenmeyen kombinasyonda secim yapamasin

## Onemli UI Davranisi

Mevcut runtime ayarlari artik `Chainlit ChatSettings` yerine custom sag panelden yonetilir:

- alan degisimi panelin draft state'ine hemen yansir
- backend pipeline degisikligi sadece `Apply` ile olur
- `Reset` aktif draft'i son uygulanmis duruma dondurur
- belge baglami ve fallback ozeti ayni panelde gorulur

Bu tasarimla onceki ayri `Belge Durumu` sag sidebar'i kaldirilmistir; sag tarafta tek panel kalir.

## Chat Profile / Provider Davranisi

UI provider secimi ile runtime ayarlar birlikte calisir:

- `Gemini`
- `OpenAI`
- `Local`
- `Extractive`

Ancak gercek calisan model ve backend kombinasyonu:

- profil secimi
- runtime settings
- `.env`

birlikte degerlendirilerek belirlenir.

## Demo Tavsiyesi

Canli demo sirasinda:

- once `.env` ile temel profili sabitle
- sonra UI'dan sadece gerekli runtime farklarini yap
- mode/backend degisirse belgeyi yeniden yukle

Ozellikle:

- embedding model degisirse index davranisi degisir
- processing mode degisirse yeni ingestion farkli olur
- OCR / detector / table backend degisirse mevcut belge geriye donuk donusmez

## UI Debug Ozeti

Panelde dort bilgilendirme alani vardir:

- `Current Draft`
- `Applied Pipeline`
- `Fallback Notes`
- `Document Context`

Bunlar salt okunurdur ve secili kombinasyonun neye donustugunu anlatir.

## Kanit Paneli (Evidence)

Soru cevaplandiktan sonra gosterilen kanit panel yalnizca **cevaptta atifta bulunulan sayfalara** ait chunk'lari icerir.

- Cevap `[DosyaAdi - Sayfa X]` formatinda citation iceriyorsa, yalnizca o sayfalardaki chunk'lar gosterilir.
- Retrieval sonucunda gelen ama cevapla ilgisi olmayan chunk'lar kanit panelinde gizlenir.
- Cevap "Belgede bu bilgi bulunamadi." ise kanit paneli bos olabilir.

Bu davranis bilinclidir: kullaniciya "neden bu cevap geldi" sorusuna dogrudan yanit verir.

## Tasarim Notu

Bu UI bir operasyon paneli degil, gelistirme ve demo panelidir. Bu yuzden:

- backend secenekleri gozukur
- kombinasyonlar kilitlenerek korunur
- ama tam reactive config editor hedeflenmemistir
