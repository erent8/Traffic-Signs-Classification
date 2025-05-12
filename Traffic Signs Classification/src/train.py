import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_gtsrb_data
from model import create_traffic_sign_model, create_simple_model, create_callbacks

def plot_training_history(history, save_path=None):
    """
    Eğitim geçmişini görselleştirir
    
    Args:
        history: Keras model.fit'ten dönen geçmiş nesnesi
        save_path: Grafiği kaydetme dizini (isteğe bağlı)
    """
    # Doğruluk (Accuracy) grafiği
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.title('Model Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend(loc='lower right')
    
    # Kayıp (Loss) grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Eğitim grafiği kaydedildi: {save_path}")
    
    plt.show()

def train_model(data_dir, model_path, epochs=50, batch_size=32, image_size=(32, 32), use_simple_model=False):
    """
    Trafik işareti sınıflandırma modelini eğitir
    
    Args:
        data_dir: Veri seti dizini
        model_path: Eğitilen model için kayıt yolu
        epochs: Eğitim turları sayısı
        batch_size: Toplu iş boyutu
        image_size: Yeniden boyutlandırma için hedef görüntü boyutu
        use_simple_model: Basit modeli kullanmak için bayrak
    """
    print("Veri yükleniyor...")
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = load_gtsrb_data(data_dir, image_size)
    
    print("Model oluşturuluyor...")
    input_shape = X_train.shape[1:]  # (height, width, channels)
    
    if use_simple_model:
        model = create_simple_model(input_shape, num_classes)
        print("Basit model kullanılıyor...")
    else:
        model = create_traffic_sign_model(input_shape, num_classes)
        print("Tam kapsamlı model kullanılıyor...")
    
    model.summary()
    
    # Model kaydetme yolu için dizin oluştur
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Geri çağrıları oluştur
    callbacks = create_callbacks(model_path)
    
    # Veri artırma (Data Augmentation)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    # Sınıf ağırlıklarını hesapla (sınıf dengesizliği için)
    y_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    print("Modeli eğitme...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )
    
    # En iyi modeli yükle (callbacks ile kaydedildi)
    model.load_weights(model_path)
    
    # Test seti değerlendirmesi
    print("Test setinde değerlendirme yapılıyor...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Doğruluğu: {test_acc*100:.2f}%")
    
    # Eğitim geçmişini görselleştir
    history_plot_path = os.path.join(os.path.dirname(model_path), 'training_history.png')
    plot_training_history(history, history_plot_path)
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trafik İşaretleri Sınıflandırma Modeli Eğitimi')
    parser.add_argument('--data_dir', type=str, default='data', help='Veri seti dizini')
    parser.add_argument('--model_path', type=str, default='models/traffic_sign_model.h5', help='Model kayıt yolu')
    parser.add_argument('--epochs', type=int, default=50, help='Eğitim turları sayısı')
    parser.add_argument('--batch_size', type=int, default=32, help='Toplu iş boyutu')
    parser.add_argument('--simple_model', action='store_true', help='Basit modeli kullan (daha hızlı eğitim)')
    parser.add_argument('--width', type=int, default=32, help='Görüntü genişliği')
    parser.add_argument('--height', type=int, default=32, help='Görüntü yüksekliği')
    
    args = parser.parse_args()
    image_size = (args.height, args.width)
    
    train_model(
        args.data_dir, 
        args.model_path, 
        args.epochs, 
        args.batch_size, 
        image_size, 
        args.simple_model
    ) 