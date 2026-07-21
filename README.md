# Drone Person Tracker (Prototype)

Referans bir fotoğraftan kişiyi tanıyıp canlı kamera akışında bulan ve takip eden bir
görüntü işleme sistemi. Arama-kurtarma senaryosu için tasarlandı: "kayıp kişinin
fotoğrafını yükle, sistem onu havadan bulup takip etsin."

## Mimari

```
Referans Fotoğraf ──► Person Detection ──► Re-ID Embedding ──► Referans Vektörü
                                                                       │
Canlı Kare ──► Ego-Motion Compensation ──► Person Detection ──► Re-ID Match ──► Tracker
```

### Neden Ego-Motion Compensation?

Drone hiçbir zaman tam sabit durmuyor (rüzgar, titreşim, uçuş hareketi). Bu yüzden ham
görüntüde arka plan da "hareket ediyormuş" gibi görünüyor ve tracker'ı şaşırtıyor.
`src/motion_compensation.py`, ardışık kareler arasında sparse optical flow + RANSAC ile
kameranın kendi hareketini (global affine transform) tahmin edip görüntüyü buna göre
hizalıyor. Bu adımdan sonra kalan hareket gerçek nesne hareketidir -- detection ve
tracking bu "temizlenmiş" görüntü üzerinde çalışır.

## Kurulum

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

İlk çalıştırmada `ultralytics` otomatik olarak `yolov8n.pt` ağırlıklarını indirecek.

## Çalıştırma

```bash
# Referans fotoğrafla başlat
python app/main.py --reference data/reference/person.jpg

# Ya da webcam açıkken 'r' tuşuna basarak referansı canlı yakala
python app/main.py
```

Tuşlar: `n` = yeni hedef ekle, `a` = seçili hedefe ek açı ekle, `1`-`9` = hedef seç,
`x` = seçili hedefi kaldır, `p` = duraklat/devam (video dosyasında), `q` = çıkış.

## Çoklu Hedef Takibi

Sistem artık aynı anda birden fazla kişiyi (varsayılan üst sınır 5, `MAX_CONCURRENT_TARGETS`)
bağımsız olarak takip edebiliyor. Her hedefin kendi galerisi, kendi tracker'ı, kendi
aktivite/duruş durumu var -- birbirine karışmıyorlar. Aynı fiziksel kişinin iki farklı
hedef olarak sayılmasını önlemek için, yeni bir hedef eklerken (`n`) sistem önce "bu kişi
zaten takip ettiğim birine benziyor mu" diye kontrol ediyor (`DUPLICATE_TARGET_SIMILARITY_THRESHOLD`);
öyleyse yeni ID vermeyi reddedip mevcut hedefin ID'sini söylüyor, o zaman `a` ile açı eklemen gerekiyor.

Her karede bir kişi bir hedef tarafından "sahiplenildiyse" (claim edildiyse), aynı kare
içinde başka bir hedef aynı kutuyu tekrar sahiplenemiyor -- bu da çift sayımı engelliyor.

## Modül Yapısı

| Dosya | Sorumluluk |
|---|---|
| `src/detection.py` | YOLO ile insan tespiti |
| `src/reid.py` | Kişi crop'undan embedding çıkarma (OSNet gerçek Re-ID / ResNet18 fallback) + galeri karşılaştırma |
| `src/motion_compensation.py` | Kamera hareketi tahmini ve kompanzasyonu (kapatılabilir, bkz. `ENABLE_MOTION_COMPENSATION`) |
| `src/tracker.py` | IOU tabanlı kare-kare takip (hafif, Re-ID'yi her karede çalıştırmamak için) |
| `src/target.py` | Her hedef için bağımsız durum (galeri, tracker, aktivite/duruş) |
| `src/activity.py` | Göreceli hıza dayalı hareket durumu (duruyor/yürüyor/koşuyor) |
| `src/posture.py` | Oturma/kalkma geçişini zaman içinde takip etme |
| `src/pose_analysis.py` | Uzuv görünürlüğü, el yanında nesne, oturma tespiti, vücut bölümü kutuları |
| `src/emotion.py` | Yüz ifadesi analizi (DeepFace, FER2013 tabanlı) |
| `src/pipeline.py` | Çoklu hedef orkestrasyonu -- paylaşımlı tespit, hedef başına Re-ID/tracking/analiz |
| `app/main.py` | Webcam/video üzerinden çalışan canlı demo, çoklu hedef kontrolleri |

### Re-ID Backend Seçimi

`config.py` içinde `REID_BACKEND = "osnet"` (varsayılan) gerçek bir Re-ID modeli kullanır
(`torchreid` paketi üzerinden OSNet, Market-1501 gibi kişi-özel veri setlerinde eğitilmiş).
`torchreid` kurulu değilse ya da ağırlık indirme başarısız olursa sistem otomatik olarak
`"resnet18"` fallback'ine düşer (genel amaçlı, daha zayıf ama bağımlılıksız).

### Aktivite ve Poz Analizi Sınırlamaları

- **Hareket durumu**: Piksel hızına dayanıyor, gerçek dünya hızı DEĞİL (monoküler kamerada
  derinlik/ölçek referansı yok). Kameraya uzaklık değiştikçe eşiklerin yeniden ayarlanması
  gerekebilir (`ACTIVITY_WALK_THRESHOLD_PX_S`, `ACTIVITY_RUN_THRESHOLD_PX_S`).
- **Uzuv görünürlüğü**: Bir eklem noktasının tespit edilememesi açı, gölge, veya örtülmeden
  kaynaklanabilir -- bu bir tıbbi yaralanma teşhisi değildir, sadece "şu an kamerada görünmüyor"
  bilgisidir. Yorumlamayı her zaman bir insan operatöre bırakın.
- **El yanında nesne**: Bilek etrafında küçük bir bölgede genel nesne tespiti çalıştırılıyor,
  gerçek bir "tutma" doğrulaması değil, kaba bir yakınlık sinyali. Ayrıca COCO sınıflarında
  olmayan uzun/ince cisimler (sopa, bayrak sırığı gibi) için kontur tabanlı bir şekil sezgiseli
  var -- bu da nesne KİMLİĞİNİ değil sadece "uzun/ince bir şekil var" bilgisini veriyor.
- **Oturma/kalkma tespiti**: Kalça-diz-ayak bileği açısına bakıyor (dar açı = oturuyor). Aşırı
  kamera açılarında (çok yandan/tepeden çekim gibi) yanılabilir, kalibre edilmiş bir ölçüm değil.
- **Vücut bölümü kutuları**: Kafa/gövde/kollar/bacaklar için ayrı kutular, eklem noktalarının
  (keypoint) konumlarından hesaplanıyor. Bir eklem noktası güvenle görünmüyorsa o bölge için
  kutu çizilmiyor (yani her zaman 6 kutunun hepsini görmeyebilirsin, bu normal).
- **Yüz ifadesi (emotion) analizi**: DeepFace'in FER2013 tabanlı hazır modelini kullanıyor,
  7 kategori var (angry/disgust/fear/happy/sad/surprise/neutral). Bu bilinen ölçüde kusurlu,
  genel amaçlı bir sınıflandırıcı -- kesin bir psikolojik durum tespiti değil, insan
  operatörün yorumlaması gereken kaba bir sinyal olarak düşünülmeli. `deepface` kurulu
  değilse ya da yüklenemezse özellik sessizce devre dışı kalır, uygulama çökmez.

## Küçük/Uzak Nesne Desteği (Tiling)

Drone yüksekten çekim yaptığında insan çok az piksel kaplayabiliyor, YOLO'nun görüntüyü
tek seferde küçültmesi bu durumda tespiti kaçırabiliyor. `config.py` içinde
`ENABLE_TILED_DETECTION = True` yaparsan sistem her kareyi örtüşen parçalara bölüp her
parçayı ayrı ayrı tarıyor, sonra sonuçları birleştiriyor (NMS ile tekrarları temizleyerek).
Bu, tespit kalitesini artırıyor ama karede birden fazla YOLO çağrısı yapıldığı için
**belirgin şekilde yavaşlatıyor**. Varsayılan olarak kapalı -- webcam gibi kişinin zaten
büyük göründüğü durumlarda faydası yok, sadece gecikme ekliyor. Yüksekten drone
görüntüsü/VisDrone gibi veri setleriyle test ederken açman önerilir.

İlgili ayarlar:
- `TILE_SIZE_PX`: her parçanın piksel boyutu (varsayılan 640, YOLO'nun doğal girdi boyutu)
- `TILE_OVERLAP_RATIO`: parçalar arası örtüşme oranı (kenardaki nesnelerin bölünmemesi için)
- `TILING_NMS_IOU_THRESHOLD`: örtüşen bölgelerdeki tekrar tespitleri birleştirme eşiği

## Bilinen Sınırlamalar / Sonraki Adımlar

- **Gerçek drone feed'i**: `cv2.VideoCapture(CAMERA_INDEX)` yerine bir RTSP stream URL'i
  verilerek gerçek drone video akışına bağlanabilir. `app/main.py --video` parametresi
  hem yerel video dosyalarını hem de RTSP/HTTP stream URL'lerini destekliyor.
- **Çoklu benzer kişi**: Aynı renk kıyafeti giyen başka biri varsa yanlış eşleşme riski
  var; `REID_MATCH_THRESHOLD` ayarı ve/veya ek özellik (yürüyüş biçimi vb.) gerekebilir.
