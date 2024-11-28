import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv("../veri_setleri/heart_failure_clinical_records_dataset.csv")


df.info()
print(df.describe())
print(df.columns)


X = df.drop(["DEATH_EVENT"], axis=1)
y = df["DEATH_EVENT"]



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(16, activation='relu'),                                   
    Dense(1, activation='sigmoid')                                 
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=12, batch_size=32 ,validation_data=(X_test, y_test))

# Model Performansı
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title("Model Doğruluk Değişimi")
plt.show()

print(f"model validation loss : {history.history["val_loss"][-1]}")
print(f"modelvalidation  accuracy : {history.history["val_accuracy"][-1]}")
print(f"model loss : {history.history["loss"][-1]}")
print(f"model accuracy : {history.history["accuracy"][-1]}")



