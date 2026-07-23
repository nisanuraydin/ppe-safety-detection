# İş Güvenliği Görüntü Analizi

Fabrika ve saha görüntülerinde kişisel koruyucu ekipman (PPE) kullanımını analiz eden Streamlit uygulaması. İki ayrı YOLO modeli ile baret, yelek, insan ve baretsiz kafa tespiti yapar.

## Özellikler

- Fotoğraf ve video analizi
- Yerel webcam, video dosyası, RTSP/HLS/MP4 bağlantısı ve YouTube canlı yayın kaynağı
- Kişi başına baret/yelek uygunluğu kontrolü
- Tehlikeli bölge tanımı, süreye bağlı kritik ihlal uyarısı ve snapshot kaydı
- Canlı ihlal günlüğü, Dashboard ve Excel raporu
- NVIDIA GPU mevcutsa CUDA ile hızlandırılmış analiz

## Model ve veri yapısı

- `runs/detect/train/weights/best.pt`: baret modeli (`head`, `helmet`, `person`)
- `runs/detect/train-2/weights/best.pt`: PPE modeli (`boots`, `gloves`, `helmet`, `human`, `vest`)
- `data.yaml`: baret modeli veri kümesi yapılandırması
- `dataset_vest/data.yaml`: PPE modeli veri kümesi yapılandırması

Veri kümesi kaynak bilgileri için [README.dataset.txt](README.dataset.txt) ve [README.roboflow.txt](README.roboflow.txt) dosyalarına bakın.

## Kurulum

Windows ve Python 3.11 önerilir.

```powershell
cd C:\Users\casper\OneDrive\Desktop\ppe-detection
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

NVIDIA GPU için temel kurulumdan sonra CUDA PyTorch paketlerini yükleyin:

```powershell
pip install -r requirements-gpu.txt
```

## Çalıştırma

VS Code'da yorumlayıcı olarak aşağıdaki dosyayı seçin:

```text
.venv311\Scripts\python.exe
```

Ardından uygulamayı başlatın:

```powershell
.\.venv311\Scripts\python.exe -m streamlit run app.py
```

## Canlı kaynaklar

Kamera sekmesinden aşağıdaki kaynaklar seçilebilir:

- Bilgisayarın yerel webcam'i
- Yüklenen bir video dosyası
- Doğrudan `rtsp://`, HLS (`.m3u8`) veya MP4 bağlantısı
- Herkese açık YouTube video ya da canlı yayın bağlantısı

YouTube ve üçüncü taraf yayınlar için yalnızca kullanım izniniz olan veya herkese açık kaynakları kullanın. Web sayfası bağlantısı yerine doğrudan video/yayın bağlantısı gerekebilir.

## Notlar

- GPU kullanılabildiğinde Kamera sekmesinde `İşlem birimi: NVIDIA GPU` görünür.
- `.venv311`, ihlal snapshot'ları ve geçici dosyalar Git takibine dahil edilmez.
- İhlal istatistikleri mevcut uygulama oturumu boyunca tutulur; sayfa/oturum sıfırlandığında Dashboard sayaçları da sıfırlanır.

## Test

Arayüzden bağımsız tehlikeli bölge hesaplarını çalıştırmak için:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv311\Scripts\python.exe -m pytest -q
```

## Bilinen sınırlamalar

- Bu çalışma eğitim ve portföy amaçlı bir prototiptir; gerçek iş güvenliği kararları için insan denetiminin yerine geçmez.
- Tespit başarısı, kullanılan iki modelin eğitim verisi, kamera açısı, ışık ve görüntü kalitesinden etkilenir.
- Çevrim içi yayınlarda gecikme ve bağlantı kararlılığı yayın sağlayıcısına bağlıdır.
- Dashboard verileri yalnızca açık Streamlit oturumu boyunca saklanır.

## Katkı ve lisans

Geri bildirim ve geliştirme önerilerine açıktır. Proje [MIT Lisansı](LICENSE) ile yayımlanmıştır.
