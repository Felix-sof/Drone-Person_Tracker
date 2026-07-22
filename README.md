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
| `src/distance.py` | Pinhole kamera yaklaşımıyla hedefe kabaca mesafe tahmini |
| `src/config_validation.py` | Uygulama başlamadan önce `config.py` değerlerinin mantıklı aralıkta olup olmadığını kontrol eder |
| `src/pipeline.py` | Çoklu hedef orkestrasyonu -- paylaşımlı tespit, hedef başına Re-ID/tracking/analiz |
| `app/main.py` | Webcam/video üzerinden çalışan canlı demo, çoklu hedef kontrolleri |
| `tests/test_distance.py` | `src/distance.py` için birim testler |

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

## Mesafe Tahmini

`config.py` içinde `ENABLE_DISTANCE_ESTIMATION = True` (varsayılan) ile her hedef için kabaca
bir mesafe tahmini ("~15.3 M") HUD panelinde gösteriliyor. Yöntem: klasik pinhole kamera
yaklaşımı -- hedefin ekrandaki piksel yüksekliği ile kameranın dikey görüş açısı (FOV) ve
varsayılan bir insan boyu (1.7m) kullanılarak mesafe hesaplanıyor.

**Önemli:** `CAMERA_VERTICAL_FOV_DEG` değerini **gerçek kameranıza göre ayarlaman gerekiyor**
-- yanlış FOV, gerçekçi görünen ama yanlış bir mesafe üretir, bunu sessizce yapar. Ayrıca bu
yöntem hedefin **dik durduğunu ve tam göründüğünü** varsayıyor; çömelmiş/oturan/kısmen
görünen biri için mesafe olduğundan uzak görünür (çünkü görünen boy küçülür, ama sebep
mesafe değil poz).

## Isı Görüşlü (Termal) Kamera Desteği

Video giriş katmanı (`app/main.py --video`) `cv2.VideoCapture`'ın anladığı her kaynağı kabul
ediyor -- bu, UVC üzerinden standart bir webcam gibi görünen bir termal kamerayı da kapsıyor,
yani I/O seviyesinde ekstra kod değişikliği gerekmeden bağlanabilir. **Ancak** çalışmayan kısım
şu: bu projedeki her tespit/pose/Re-ID modeli sıradan RGB görüntülerle eğitilmiş. Termal
karelerde insanlar ayırt edici özelliği olmayan parlak lekeler gibi görünüyor (kıyafet
rengi/dokusu yok), bu da hazır RGB modellerin doğruluğunu ölçülebilir şekilde düşürüyor
(Teledyne FLIR ADAS gibi veri setlerinde yayınlanmış sonuçlar bunu gösteriyor). Gerçek termal
desteği eklemek bir config ayarı değil, bir **model eğitimi projesi** (örn. YOLO'yu termal bir
veri setinde fine-tune etmek) -- şu an için uygulanmadı, gelecekteki bir yön olarak not
düşülüyor.

## Config Validasyonu ve Testler

Uygulama başlamadan önce `src/config_validation.py` içindeki `validate_config()`
fonksiyonu çalışıyor ve `config.py`'deki kritik ayarların (FOV açısı, Re-ID eşikleri,
tiling ayarları vb.) mantıklı aralıkta olup olmadığını kontrol ediyor. Örneğin
`CAMERA_VERTICAL_FOV_DEG` fiziksel olarak imkansız bir değere (0 veya 180+) ayarlanmışsa,
uygulama sessizce yanlış mesafe üretmek yerine açık bir `[CONFIG ERROR]` mesajıyla
başlamayı reddediyor.

Ayrıca `tests/` klasöründe `pytest` ile çalışan birim testler var (şu an
`src/distance.py` için). Çalıştırmak için:

```bash
pip install pytest
pytest tests/ -v
```

Proje kökündeki `pytest.ini` dosyası (`pythonpath = .`), testlerin `src/` ve `app/`
paketlerini doğru şekilde import edebilmesini sağlıyor.

## Bilinen Sınırlamalar / Sonraki Adımlar

- **Gerçek drone feed'i**: `cv2.VideoCapture(CAMERA_INDEX)` yerine bir RTSP stream URL'i
  verilerek gerçek drone video akışına bağlanabilir. `app/main.py --video` parametresi
  hem yerel video dosyalarını hem de RTSP/HTTP stream URL'lerini destekliyor.
- **Çoklu benzer kişi**: Aynı renk kıyafeti giyen başka biri varsa yanlış eşleşme riski
  var; `REID_MATCH_THRESHOLD` ayarı ve/veya ek özellik (yürüyüş biçimi vb.) gerekebilir.
- **Termal kamera**: Yukarıda açıklandığı gibi, I/O seviyesinde destekleniyor ama model
  doğruluğu için termal-özel eğitim gerekiyor.
- **Test kapsamı**: Şu an sadece `src/distance.py` için birim test var; diğer
  modüller (tracker, Re-ID, pose analysis) henüz test edilmiyor.