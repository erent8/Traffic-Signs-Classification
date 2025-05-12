import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

from data_loader import preprocess_image
from model import load_trained_model

def load_class_names(data_dir):
    """
    Sınıf isimlerini içeren CSV dosyasını yükler
    
    Args:
        data_dir: Veri seti dizini (CSV dosyasının bulunduğu yer)
    
    Returns:
        Sınıf ID'leri ve isimlerini içeren sözlük
    """
    class_names = {}
    
    # İlk olarak meta CSV dosyasını kontrol et
    csv_path = os.path.join(data_dir, 'signnames.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Başlığı atla
            for row in reader:
                class_id = int(row[0])
                class_name = row[1]
                class_names[class_id] = class_name
    else:
        # CSV yoksa varsayılan olarak sınıf ID'lerini kullan
        print("Uyarı: signnames.csv dosyası bulunamadı. Sayısal ID'ler kullanılacak.")
        # GTSRB için 43 sınıf kullanılır
        for i in range(43):
            class_names[i] = f"Sınıf {i}"
    
    return class_names

def predict_image(model_path, image_path, data_dir='data', image_size=(32, 32)):
    """
    Görüntüdeki trafik işaretini tahmin eder
    
    Args:
        model_path: Eğitilmiş model dosya yolu
        image_path: Tahmin edilecek görüntü dosya yolu
        data_dir: Veri seti dizini (sınıf isimleri için gerekli)
        image_size: Yeniden boyutlandırma için hedef boyut
    
    Returns:
        Tahmin edilen sınıf ID'si ve adı
    """
    # Modeli yükle
    print(f"Model yükleniyor: {model_path}")
    model = load_trained_model(model_path)
    
    # Görüntüyü ön işle
    print(f"Görüntü işleniyor: {image_path}")
    img = preprocess_image(image_path, image_size)
    
    # Tahmin yap
    prediction = model.predict(img)
    class_id = np.argmax(prediction[0])
    confidence = prediction[0][class_id] * 100
    
    # Sınıf isimlerini yükle
    class_names = load_class_names(data_dir)
    class_name = class_names.get(class_id, f"Sınıf {class_id}")
    
    print(f"Tahmin: {class_name} (ID: {class_id}), Güven: {confidence:.2f}%")
    
    # Görüntüyü ve tahmini göster
    plt.figure(figsize=(6, 6))
    img_display = plt.imread(image_path)
    plt.imshow(img_display)
    plt.title(f"Tahmin: {class_name}\nGüven: {confidence:.2f}%")
    plt.axis('off')
    plt.show()
    
    return class_id, class_name, confidence

def predict_multiple(model_path, image_dir, data_dir='data', image_size=(32, 32), max_images=5):
    """
    Bir dizindeki birden fazla görüntü üzerinde tahmin yapar
    
    Args:
        model_path: Eğitilmiş model dosya yolu
        image_dir: Görüntülerin bulunduğu dizin
        data_dir: Veri seti dizini (sınıf isimleri için gerekli)
        image_size: Yeniden boyutlandırma için hedef boyut
        max_images: İşlenecek maksimum görüntü sayısı
    """
    # Modeli yükle
    print(f"Model yükleniyor: {model_path}")
    model = load_trained_model(model_path)
    
    # Sınıf isimlerini yükle
    class_names = load_class_names(data_dir)
    
    # Görüntüleri bul
    image_files = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):
            image_files.append(os.path.join(image_dir, filename))
    
    # Maksimum görüntü sayısını sınırla
    image_files = image_files[:max_images]
    
    if not image_files:
        print(f"Dizinde görüntü bulunamadı: {image_dir}")
        return
    
    # Görüntü sayısına göre subplot düzeni belirleme
    n_images = len(image_files)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, image_path in enumerate(image_files):
        try:
            # Görüntüyü ön işle
            img = preprocess_image(image_path, image_size)
            
            # Tahmin yap
            prediction = model.predict(img, verbose=0)
            class_id = np.argmax(prediction[0])
            confidence = prediction[0][class_id] * 100
            
            class_name = class_names.get(class_id, f"Sınıf {class_id}")
            
            # Subplot'a görüntüyü ve tahmini çiz
            plt.subplot(n_rows, n_cols, i + 1)
            img_display = plt.imread(image_path)
            plt.imshow(img_display)
            plt.title(f"Tahmin: {class_name}\nGüven: {confidence:.2f}%")
            plt.axis('off')
            
            print(f"Görüntü {i+1}/{n_images}: {os.path.basename(image_path)}")
            print(f"  Tahmin: {class_name} (ID: {class_id}), Güven: {confidence:.2f}%")
            
        except Exception as e:
            print(f"Görüntü işlenirken hata: {image_path}")
            print(f"  Hata: {e}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trafik İşaretleri Sınıflandırması Tahmini')
    parser.add_argument('--model_path', type=str, default='models/traffic_sign_model.h5', help='Eğitilmiş model dosya yolu')
    parser.add_argument('--image_path', type=str, help='Tahmin edilecek görüntü dosya yolu')
    parser.add_argument('--image_dir', type=str, help='Tahmin edilecek görüntülerin bulunduğu dizin')
    parser.add_argument('--data_dir', type=str, default='data', help='Veri seti dizini')
    parser.add_argument('--width', type=int, default=32, help='Görüntü genişliği')
    parser.add_argument('--height', type=int, default=32, help='Görüntü yüksekliği')
    parser.add_argument('--max_images', type=int, default=5, help='İşlenecek maksimum görüntü sayısı (dizin modunda)')
    
    args = parser.parse_args()
    image_size = (args.height, args.width)
    
    if args.image_dir:
        # Birden fazla görüntü tahmini
        predict_multiple(args.model_path, args.image_dir, args.data_dir, image_size, args.max_images)
    elif args.image_path:
        # Tek görüntü tahmini
        predict_image(args.model_path, args.image_path, args.data_dir, image_size)
    else:
        parser.error("--image_path veya --image_dir parametrelerinden birini belirtmelisiniz.")