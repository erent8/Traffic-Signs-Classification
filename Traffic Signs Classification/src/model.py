import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def create_traffic_sign_model(input_shape, num_classes):
    """
    Trafik işaretleri sınıflandırması için bir CNN modeli oluşturur
    
    Args:
        input_shape: Giriş görüntüsü boyutu (height, width, channels)
        num_classes: Sınıf sayısı
    
    Returns:
        Derlenen Keras modeli
    """
    model = Sequential()
    
    # İlk Evrişim Bloğu
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    # İkinci Evrişim Bloğu
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Üçüncü Evrişim Bloğu
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    # Düzleştirme ve Tam Bağlantılı Katmanlar
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Modeli derle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_callbacks(save_path):
    """
    Eğitim sırasında kullanılacak geri çağrılar (callbacks) oluşturur
    
    Args:
        save_path: Model kaydetme dizini
        
    Returns:
        Geri çağrılar listesi
    """
    checkpoint = ModelCheckpoint(
        filepath=save_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    return [checkpoint, early_stopping]

def create_simple_model(input_shape, num_classes):
    """
    Daha basit bir CNN modeli (daha hızlı eğitim için)
    
    Args:
        input_shape: Giriş görüntüsü boyutu (height, width, channels)
        num_classes: Sınıf sayısı
    
    Returns:
        Derlenen Keras modeli
    """
    model = Sequential()
    
    # İlk Evrişim Bloğu
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # İkinci Evrişim Bloğu
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Düzleştirme ve Tam Bağlantılı Katmanlar
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Modeli derle
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_trained_model(model_path):
    """
    Önceden eğitilmiş modeli yükler
    
    Args:
        model_path: Model dosya yolu
        
    Returns:
        Yüklenmiş model
    """
    return load_model(model_path) 