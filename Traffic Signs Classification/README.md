# Trafik İşaretleri Sınıflandırması

Bu proje, Evrişimli Sinir Ağları (CNN) kullanarak trafik işaretlerini sınıflandırmayı amaçlamaktadır. German Traffic Sign Recognition Benchmark (GTSRB) veri seti kullanılarak eğitilmiştir.

## Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Veri Seti

[German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html) veri setini kullanmaktadır. Veri setini indirmek için:

1. [GTSRB web sitesini](https://benchmark.ini.rub.de/gtsrb_dataset.html) ziyaret edin.
2. "Download the dataset" bölümünden veri setini indirin.
3. İndirdiğiniz veri setini `data` klasörü içine çıkartın.

## Kullanım

Model eğitimi için:

```bash
python train.py
```

Test görüntüleri üzerinde tahmin yapmak için:

```bash
python predict.py --image_path path/to/your/image.jpg
```

## Proje Yapısı

- `data/`: Veri seti dosyaları
- `models/`: Eğitilmiş model dosyaları
- `src/`: Kaynak kodlar
  - `data_loader.py`: Veri yükleme ve ön işleme 
  - `model.py`: CNN model tanımı
  - `train.py`: Model eğitimi
  - `predict.py`: Tahmin yapma
- `notebooks/`: Jupyter not defterleri (veri keşfi ve analiz için) 