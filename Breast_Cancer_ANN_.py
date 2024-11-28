import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dropout,Dense

df = pd.read_csv("../veri_setleri/breast_cancer.csv")
df.drop(["Unnamed: 32"],axis=1,inplace=True)
#df.info()
#df.describe().T

X = df.drop(["id","diagnosis"],axis=1)
y = df["diagnosis"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train ,X_test ,y_train,y_test = train_test_split(X,y,test_size=0.15)

model = Sequential([
    Dense(units=32, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.1),
    Dense(units=64, activation='relu'),
    Dropout(0.1),
    Dense(units=32, activation='relu'),
    Dropout(0.1),
    Dense(units=16,activation='relu'),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=32, epochs=20,validation_split=0.2)

loss ,accuracy = model.evaluate(X_test,y_test)
print(f"test loss {loss} and test accuracy : {accuracy}")
model.summary()



plt.figure(figsize=(12, 6))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


