import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_gtsrb_data(data_dir, image_size=(32, 32)):
    """
    GTSRB veri setini yükler ve ön işler
    
    Args:
        data_dir: Veri seti dizini
        image_size: Yeniden boyutlandırma için hedef boyut
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes
    """
    # Eğitim veri setini yükle
    train_path = os.path.join(data_dir, 'Train')
    test_path = os.path.join(data_dir, 'Test')
    
    # Kullanılabilecek alternatif bir yükleme işlevi
    try:
        # Meta veri dosyası yöntemini deneyin
        print("Meta veri yaklaşımıyla yüklemeyi deniyorum...")
        return load_using_meta_files(data_dir, image_size)
    except Exception as e:
        print(f"Meta veri kullanarak yükleme başarısız: {e}")
        print("Alternatif klasör yapısı yükleme yöntemini deniyorum...")
        return load_from_directory_structure(data_dir, image_size)

def load_from_directory_structure(data_dir, image_size=(32, 32)):
    """
    Veri setini klasör yapısından yükler (her sınıf için ayrı klasör)
    """
    train_path = os.path.join(data_dir, 'Train')
    test_path = os.path.join(data_dir, 'Test')
    
    images = []
    labels = []
    
    # Eğitim klasörlerini oku
    for class_id in os.listdir(train_path):
        class_path = os.path.join(train_path, class_id)
        if os.path.isdir(class_path):
            class_id_num = int(class_id)
            for img_file in os.listdir(class_path):
                if img_file.endswith('.png') or img_file.endswith('.jpg') or img_file.endswith('.ppm'):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, image_size)
                        images.append(img)
                        labels.append(class_id_num)
                    except Exception as e:
                        print(f"Görüntü yüklenirken hata: {img_path}, Hata: {e}")
    
    # Numpy dizilerine dönüştür
    X = np.array(images, dtype=np.float32) / 255.0  # Normalleştirme
    y = np.array(labels)
    
    # Eğitim ve doğrulama setlerine ayır
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test verilerini yükle (varsa)
    X_test = []
    y_test = []
    
    if os.path.exists(test_path):
        # Test görüntüleri ve etiketleri için meta veri dosyasını kontrol et
        test_csv_path = os.path.join(data_dir, 'Test.csv')
        if os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
            
            for index, row in test_df.iterrows():
                img_path = os.path.join(test_path, row['Filename'])
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, image_size)
                    X_test.append(img)
                    y_test.append(row['ClassId'])
                except Exception as e:
                    print(f"Test görüntüsü yüklenirken hata: {img_path}, Hata: {e}")
        else:
            # Klasör yapısından test verilerini yükle
            for class_id in os.listdir(test_path):
                class_path = os.path.join(test_path, class_id)
                if os.path.isdir(class_path):
                    class_id_num = int(class_id)
                    for img_file in os.listdir(class_path):
                        if img_file.endswith('.png') or img_file.endswith('.jpg') or img_file.endswith('.ppm'):
                            img_path = os.path.join(class_path, img_file)
                            try:
                                img = cv2.imread(img_path)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, image_size)
                                X_test.append(img)
                                y_test.append(class_id_num)
                            except Exception as e:
                                print(f"Test görüntüsü yüklenirken hata: {img_path}, Hata: {e}")
    
    if X_test:
        X_test = np.array(X_test, dtype=np.float32) / 255.0
        y_test = np.array(y_test)
    else:
        # Test verisi yoksa değerlendirme için doğrulama verilerini kullan
        X_test = X_val
        y_test = y_val
    
    # Sınıf sayısını hesapla
    num_classes = len(np.unique(y))
    
    # One-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"Yüklenen veri: Eğitim: {X_train.shape}, Doğrulama: {X_val.shape}, Test: {X_test.shape}")
    print(f"Sınıf sayısı: {num_classes}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes

def load_using_meta_files(data_dir, image_size=(32, 32)):
    """
    Meta veri dosyalarını (CSV) kullanarak veri setini yükler
    """
    train_csv = os.path.join(data_dir, 'Train.csv')
    test_csv = os.path.join(data_dir, 'Test.csv')
    
    train_df = pd.read_csv(train_csv)
    
    images = []
    labels = []
    
    # Eğitim görüntülerini yükle
    for index, row in train_df.iterrows():
        img_path = os.path.join(data_dir, row['Path'])
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(row['ClassId'])
        except Exception as e:
            print(f"Görüntü yüklenirken hata: {img_path}, Hata: {e}")
    
    # Numpy dizilerine dönüştür
    X = np.array(images, dtype=np.float32) / 255.0  # Normalleştirme
    y = np.array(labels)
    
    # Eğitim ve doğrulama setlerine ayır
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test verilerini yükle (varsa)
    X_test = []
    y_test = []
    
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        
        for index, row in test_df.iterrows():
            img_path = os.path.join(data_dir, row['Path'])
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, image_size)
                X_test.append(img)
                y_test.append(row['ClassId'])
            except Exception as e:
                print(f"Test görüntüsü yüklenirken hata: {img_path}, Hata: {e}")
    
    if X_test:
        X_test = np.array(X_test, dtype=np.float32) / 255.0
        y_test = np.array(y_test)
    else:
        # Test verisi yoksa değerlendirme için doğrulama verilerini kullan
        X_test = X_val
        y_test = y_val
    
    # Sınıf sayısını hesapla
    num_classes = len(np.unique(y))
    
    # One-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"Yüklenen veri: Eğitim: {X_train.shape}, Doğrulama: {X_val.shape}, Test: {X_test.shape}")
    print(f"Sınıf sayısı: {num_classes}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes

def preprocess_image(image_path, image_size=(32, 32)):
    """
    Tahmin için tek bir görüntüyü ön işler
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)  # Batch boyutu ekle 