import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")
# 1. Veri Oluşturma
def create_sine_wave_data(seq_length, num_samples):
    """
    Sinüs dalgası tabanlı veri seti oluşturur.
    seq_length: Giriş dizisinin uzunluğu (örneğin, 50 zaman adımı)
    num_samples: Kaç veri noktası oluşturulacak
    """
    x = np.linspace(0, 50, num_samples)
    sine_wave = np.sin(x)  # Sinüs dalgası
    X, y = [], []
    for i in range(len(sine_wave) - seq_length):
        X.append(sine_wave[i:i + seq_length])
        y.append(sine_wave[i + seq_length])  # Sıradaki değeri tahmin etmek
    return np.array(X), np.array(y)

# Parametreler
seq_length = 50  # Girdi dizisi uzunluğu
num_samples = 1000  # Toplam örnek sayısı

# Veri oluşturma
X, y = create_sine_wave_data(seq_length, num_samples)

# 2. Veriyi 3 boyutlu hale getirme (LSTM için gerekli)
X = X.reshape((X.shape[0], X.shape[1], 1))  # (örnek_sayısı, zaman_adımları, özellik_sayısı)

# 3. Eğitim ve test setine ayırma
train_size = int(len(X) * 0.8)  # %80 eğitim, %20 test
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Model Oluşturma
model = Sequential([
    # İlk LSTM Katmanı
    LSTM(
        units=128, 
        activation='tanh', 
        recurrent_activation='sigmoid', #bir önceki unutup unutmiyacağını belirler
        return_sequences=True, 
        input_shape=(seq_length, 1), 
        dropout=0.2, 
        recurrent_dropout=0.2
    ),
    # İkinci LSTM Katmanı
    LSTM(
        units=64, 
        activation='tanh', 
        recurrent_activation='sigmoid', #bir önceki unutup unutmiyacağını belirler
        return_sequences=True, 
        dropout=0.2, 
        recurrent_dropout=0.2
    ),
    # Üçüncü LSTM Katmanı
    LSTM(
        units=32, 
        activation='tanh', 
        recurrent_activation='sigmoid', #bir önceki unutup unutmiyacağını belirler
        return_sequences=False,  # Son zaman adımını döndür
        dropout=0.2, 
        recurrent_dropout=0.2
    ),
    # Tam Bağlantılı (Dense) Çıkış Katmanı
    Dense(units=1, activation=None)  # Çıkış aktivasyonu yok, regresyon problemi
])

# 5. EarlyStopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',        # Doğrulama kaybını takip et
    patience=10,                # 5 epok boyunca iyileşme olmazsa eğitim durdurulur
    restore_best_weights=True,  # Eğitim durdurulduğunda en iyi ağırlıklar geri yüklenir
    verbose = 1
)

# Modeli Derleme
model.compile(
    optimizer='adam',          # Adam optimizasyon algoritması
    loss='mse'# Ortalama kare hata (MSE)
)

# Model özeti
model.summary()

# 6. Modeli Eğitme
history = model.fit(
    X_train, y_train,                   # Eğitim verisi ve hedefler
    epochs=50,                          # Maksimum epok sayısı
    batch_size=32,                      # Mini-batch boyutu
    validation_data=(X_test, y_test),   # Doğrulama verisi
    callbacks=[early_stopping],         # EarlyStopping callback'i kullanılıyor
    verbose=1                           # Eğitim sürecini konsola yazdır
)

# 7. Tahmin
predictions = model.predict(X_test)

# 8. Performans Değerlendirme ve Görselleştirme

# Eğitim ve doğrulama kayıpları
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı', color='blue')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', color='orange')
plt.title("Model Kaybı")
plt.xlabel("Epok Sayısı")
plt.ylabel("Kayıp (Loss)")
plt.legend()
plt.show()

# Gerçek değerler vs Tahminler
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Gerçek Değerler', color='green')
plt.plot(predictions, label='Tahminler', color='red', linestyle='dashed')
plt.title("LSTM ile Sinüs Dalgası Tahmini")
plt.xlabel("Zaman Adımları")
plt.ylabel("Değer")
plt.legend()
plt.show()

