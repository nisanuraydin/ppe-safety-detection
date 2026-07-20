# İş Güvenliği Görüntü Analizi

Bu proje, fabrika ortamlarında iş güvenliği kurallarının yapay zekâ ile otomatik olarak takip edilmesini sağlayan bir görsel analiz sistemidir. Kamera görüntüleri ve videolar YOLO tabanlı bir model ile analiz edilerek iş ekipmanları (baret, yelek vb.) tespit edilebilir.

## Özellikler

- Fotoğraf yükleyerek iş güvenliği tespiti
- Video dosyası yükleyerek video analizi
- Canlı kamera üzerinden tek kare görüntü analizi
- Varsayılan model ile otomatik yükleme
- Hata yönetimi ve kullanıcıya açıklayıcı mesajlar

## Dosya yapısı

- `app.py` - Streamlit tabanlı uygulama kodu
- `requirements.txt` - projenin Python bağımlılıkları
- `runs/detect/train/weights/best.pt` - eğitimli model ağırlığı (varsa)
- `yolov8n.pt`, `yolo26n.pt` - örnek ön yüklü model ağırlıkları
- `README.dataset.txt` - kullanılan veri kümesi bilgisi
- `README.roboflow.txt` - Roboflow veri kümesi açıklaması

## Gereksinimler

- Python 3.10 veya üzeri (tercihen 3.11)
- `pip` ile aşağıdaki paketler

## Kurulum

1. Proje dizinine gidin:

```powershell
cd C:\Users\casper\OneDrive\Desktop\ppe-detection
```

2. Sanal ortamı etkinleştirin (varsa):

```powershell
.\venv\Scripts\activate
```

3. Gerekli paketleri yükleyin:

```powershell
pip install -r requirements.txt
```

## Uygulamayı Çalıştırma

Aşağıdaki komutu çalıştırın:

```powershell
python -m streamlit run app.py
```

Tarayıcınız açılmazsa terminalde görünen `Local URL` adresini kopyalayıp tarayıcıya yapıştırın.

## Kullanım

1. Uygulama varsayılan modeli otomatik olarak yükler.
2. `Güven eşiği`ni ayarlayın.
3. Fotoğraf yükleyip `Analiz Et` ile görsel tespiti çalıştırın.
4. Video yükleyip `Videoyu Analiz Et` ile video analizi yapın.
5. `Kameradan bir kare al` ile bağlı web kameranızdan tek kare görüntü tespiti yapın.

## Notlar

- Video analizi birden fazla kare üzerinde hesaplama yapar; bu nedenle işlem süresi videonun uzunluğuna göre uzayabilir.
- Yüklenen modelin uyumlu bir YOLO `.pt` ağırlığı olması gerekir.
- Uygulama önce `runs/detect/train/weights/best.pt` dosyasını arar. Bu dosya yoksa `yolov8n.pt` veya `yolo26n.pt` otomatik olarak fallback olarak kullanılır.
- `README.dataset.txt` ve `README.roboflow.txt` dosyaları, bu projede kullanılan veri kümesiyle ilgili ek bilgileri içerir.
