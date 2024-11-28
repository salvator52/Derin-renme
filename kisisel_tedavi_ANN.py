import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#data create 
num_samples =1000
"""
1000 adet hasta 
age : 20-80
gender K-E
disease  0(hasta değil),1(hipertansiyon),2(diyabet),3(her ikisi)
ates  0(yok),1(hafifi),2(şiddetli)
öksürük  0,1,2  aynı
baş ağrısı  0,1,2
kan basıncı 90-180
kan şekeri 70-200
onceki tedavi yanıtı  0(kötü),1(orta),2(iyi)
"""

"""
0 (tedavi1) (basit tedavi)  kan basıncı < 120 kan şekeri < 100 ,semptomlar <=1
1 (tedavi2)  (orta tedavi)  kan basinci >120 , kan sekeri <140  semptomlpar ==1 
2 (tedavi3) (ileri seviye)   //         >140 ,  //        >=150    //   ==2
"""


def assign_treatment_1(blood_pressure,blood_sugar,symptom):
    return (blood_pressure <120) & (blood_sugar < 100) & (symptom <=1)
def assign_treatment_2(blood_pressure,blood_sugar,symptom):
    return (blood_pressure >120) & (blood_sugar < 140) & (symptom ==1)
def assign_treatment_3(blood_pressure,blood_sugar,symptom):
    return (blood_pressure >120) & (blood_sugar >= 100) & (symptom ==2 )


age = np.random.randint(20,80,num_samples)
gender = np.random.randint(0,2,num_samples)
disease = np.random.randint(0,4,num_samples)
symptom_fever = np.random.randint(0,3,num_samples)
symptom_cough = np.random.randint(0,3,num_samples)
symptom_headache = np.random.randint(0,3,num_samples)
blood_pressure = np.random.randint(90,180,num_samples)
blood_sugar = np.random.randint(70,200,num_samples)
previous_treatmen_responce = np.random.randint(0,3,num_samples)

symptom = symptom_cough + symptom_fever + symptom_headache

treatment_plan = np.zeros(num_samples)
    
for i in range(num_samples):
    if assign_treatment_1(blood_pressure[i], blood_sugar[i], symptom[i]):
        treatment_plan[i] = 0 #tedavi 1
    elif assign_treatment_1(blood_pressure[i], blood_sugar[i], symptom[i]):  
        treatment_plan[i] = 1 #tedavi 1
    else :
        treatment_plan[i] = 2
        
    
data = pd.DataFrame({
    "age":age,
    "gender":gender,
    "disease":disease,
    "symptom_fever":symptom_fever,
    "symptom_cough":symptom_cough,
    "symptom_headache":symptom_headache,
    "previous_treatmen_responce":previous_treatmen_responce,
    "symptom":symptom,
    "treatment_plan":treatment_plan})

X = data.drop(["treatment_plan"],axis=1)
y = to_categorical(data.treatment_plan,num_classes=3)

X_train ,X_test ,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = Sequential([
    Dense(32,activation ="relu",input_shape=(X_train.shape[1],)),
    Dense(64,activation="relu"),
    Dense(3,activation="softmax")])
model.compile(optimizer = "adam",loss="categorical_crossentropy",metrics=["accuracy"])

history = model.fit(X_train,y_train,epochs=20,validation_data = (X_test,y_test),batch_size = 32)


val_loss , val_accuracy = model.evaluate(X_test,y_test)
print(f"val accuracy : {val_accuracy}, val loss {val_loss}")

#Görselleştirme
plt.figure()

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],"r^-",label="trainin acc")
plt.plot(history.history["val_accuracy"],"bo-",label="validation acc")
plt.title("training and validation accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["loss"],"r^-",label="trainin loss")
plt.plot(history.history["val_loss"],"bo-",label="validation loss")
plt.title("training and validation loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.grid(True)

