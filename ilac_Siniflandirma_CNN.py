# Gerekli kütüphaneleri yüklüyoruz
import numpy as np  # Matematiksel işlemler için
import pandas as pd  # Veri manipülasyonu için
import matplotlib.pyplot as plt  # Grafikler çizmek için
import tensorflow as tf  # Derin öğrenme modelleri oluşturmak için
from tensorflow.keras import layers, models  # Keras katmanları ve modelleri için
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Görüntü verisi oluşturma ve artırma
from tensorflow.keras.layers import Dense, Dropout, Resizing, Rescaling  # Temel katmanlar (tam bağlantı, dropout, boyutlandırma vb.)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Erken durdurma ve model kaydetme
from tensorflow.keras.optimizers import Adam  # Optimizasyon algoritması
from tensorflow.keras.applications import MobileNetV2  # Önceden eğitilmiş model

from pathlib import Path  # Dosya yollarını işlemek için
from sklearn.metrics import classification_report  # Modelin değerlendirme metrikleri
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test için bölmek

import os.path

# Uyarı mesajlarını kapatıyoruz
import warnings
warnings.filterwarnings("ignore")

# Veri yolunu tanımlıyoruz
dataset = r"C:/Users/mhmmt/OneDrive/Masaüstü/python/veri_setleri/Drug_vision"  # Görüntü veri setinin bulunduğu yol
image_dir = Path(dataset)  # Dosya yolu işlemlerini kolaylaştırıyoruz

# Veri setindeki tüm görüntü dosyalarını alıyoruz
filepaths = list(image_dir.glob(r"**/*.JPG")) + list(image_dir.glob(r"**/*.png"))  # .JPG ve .png dosyalarını bul
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))  # Dosya yolundaki klasör isimlerini etiket olarak ayır

# Veri çerçevesi oluşturuyoruz
filepaths = pd.Series(filepaths, name="filepath").astype(str)  # Görüntü dosya yollarını bir pandas serisine dönüştür
labels = pd.Series(labels, name="Label")  # Etiketleri bir pandas serisine dönüştür
image_df = pd.concat([filepaths, labels], axis=1)  # Dosya yolları ve etiketleri birleştir

# Eğitim ve test veri çerçevelerini oluşturuyoruz
train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True)  # Veriyi %80 eğitim, %20 test olacak şekilde ayır

# Eğitim ve test için görüntü artırma (data augmentation) ayarlarını yapıyoruz
train_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,  # MobileNetV2'ye uygun ön işleme
    validation_split=0.2  # Eğitim verisinin %20'si doğrulama (validation) için ayrılacak
)
test_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input  # Sadece test verisini ölçeklendirme
)

# Eğitim ve doğrulama veri jeneratörleri oluşturuyoruz
train_images = train_gen.flow_from_dataframe(
    dataframe=train_df,  # Eğitim verisi çerçevesi
    x_col="filepath",  # Görüntülerin dosya yollarını temsil eden sütun
    y_col="Label",  # Görüntülerin etiketlerini temsil eden sütun
    target_size=(224, 224),  # Görüntüleri 224x224 boyutuna yeniden boyutlandır
    color_mode="rgb",  # Görüntülerin renk modunu RGB olarak ayarla
    class_mode="categorical",  # Etiketlerin kategorik olduğunu belirt
    batch_size=64,  # Her bir iterasyonda kullanılacak görüntü sayısı
    shuffle=True,  # Görüntülerin sırasını karıştır
    seed=42,  # Rastgeleliği kontrol etmek için sabit bir tohum değeri
    subset="training"  # Bu jeneratör sadece eğitim verisini döndürecek
)
validation_images = train_gen.flow_from_dataframe(
    dataframe=train_df,  # Eğitim verisi çerçevesi
    x_col="filepath", y_col="Label", target_size=(224, 224), color_mode="rgb",
    class_mode="categorical", batch_size=64, shuffle=True, seed=42, subset="validation"
)

# Test veri jeneratörünü oluşturuyoruz
test_images = test_gen.flow_from_dataframe(
    dataframe=test_df,  # Test verisi çerçevesi
    x_col="filepath", y_col="Label", target_size=(224, 224), color_mode="rgb",
    class_mode="categorical", batch_size=64, shuffle=False
)

# Görüntü ön işleme için boyutlandırma ve yeniden ölçekleme katmanları
preprocessing_layer = tf.keras.Sequential([
    Resizing(224, 224),  # Görüntüleri 224x224 boyutuna yeniden boyutlandır
    Rescaling(1.0 / 255)  # Görüntü piksel değerlerini 0-1 aralığına ölçeklendir
])

# Önceden eğitilmiş MobileNetV2 modelini yükleyip, son katmanlarını çıkarıyoruz
pretrained_model = MobileNetV2(
    input_shape=(224, 224, 3),  # Giriş boyutu 224x224 ve RGB kanallarına uygun
    include_top=False,  # Sınıflandırma katmanını dahil etme (özellik çıkarıcı olarak kullan)
    weights="imagenet",  # ImageNet üzerinde eğitilmiş ağırlıkları yükle
    pooling="avg"  # Son katman için ortalama havuzlama uygula
)
pretrained_model.trainable = False  # Modelin ağırlıklarını dondur (eğitim sırasında değişmeyecek)

# Eğitim sırasında modelin kaydedilmesi ve erken durdurma için callback'ler
checkpoint_path = "checkpoint.weights.h5"  # Model ağırlıklarının kaydedileceği dosya yolu
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor="val_accuracy", save_best_only=True)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)  # Val_loss 5 epoch boyunca iyileşmezse dur

# Modelin katmanlarını tanımlıyoruz
inputs = pretrained_model.input  # Modelin giriş katmanını al
x = preprocessing_layer(inputs)  # Ön işleme katmanını ekle
x = pretrained_model(x)  # Önceden eğitilmiş modelin çıktısını al
x = Dense(256, activation="relu")(x)  # Tam bağlantılı katman, 256 nöron, ReLU aktivasyonu
x = Dropout(0.2)(x)  # Aşırı öğrenmeyi önlemek için %20 dropout uygula
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(10, activation="softmax")(x)  # Son katman, 10 sınıf, softmax aktivasyonu
model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Modeli tanımla

# Modeli derliyoruz
model.compile(
    optimizer=Adam(0.0001),  # Adam optimizasyon algoritması, öğrenme hızı 0.0001
    loss="categorical_crossentropy",  # Çok sınıflı sınıflandırma kaybı
    metrics=["accuracy"]  # Doğruluk metriği
)

# Modeli eğitiyoruz
history = model.fit(
    train_images,  # Eğitim verisi
    steps_per_epoch=len(train_images),  # Her epoch'ta kullanılacak adım sayısı
    validation_data=validation_images,  # Doğrulama verisi
    validation_steps=len(validation_images),  # Her epoch'ta doğrulamada kullanılacak adım sayısı
    epochs=10,  # Eğitim süreci için toplam epoch sayısı
    callbacks=[early_stopping, checkpoint_callback]  # Callback fonksiyonlarını dahil et
)
