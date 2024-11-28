import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import cv2
import os 

from tqdm import tqdm

# %%   load data 

labels = ["PNEUMONIA","NORMAL"]
img_size = 150

def get_training_data(data_dir):
    data = []
    
    for label in labels:
        path = os.path.join(data_dir,label)
        class_num = labels.index(label)
        
        for img in tqdm(os.listdir(path)):
            
            try:
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                
                if img_arr is None:
                    print("Read image error")
                    continue
                
                resized_arr = cv2.resize(img_arr, (img_size,img_size))
                
                data.append([resized_arr,class_num])
                           
            except Exception as e:
                print("Error",e)
                
    return np.array(data,dtype=object)


#../veri_setleri/akciger_kanseri/chest_xray/test   
data_dir = r"..\veri_setleri\akciger_kanseri\chest_xray\train"
train = get_training_data(data_dir)
test = get_training_data(r"..\veri_setleri\akciger_kanseri\chest_xray\test")
validation = get_training_data(r"..\veri_setleri\akciger_kanseri\chest_xray\val")

# %% data visualization and preprocessing

l = []

for i in train:
    if(i[1]==0):
        l.append("PNEUMONIA")
    else:
        l.append("NORMAL")

sns.countplot(x=l)

X_train = []
y_train=[]

X_test = []
y_test = []

X_val = []
y_val = []

def splitting(data):
    features = []
    target = []
    for x in data:
        features.append(x[0])
        target.append(x[1])
    return features,target
X_train,y_train = splitting(train)
X_test,y_test = splitting(test)
X_val ,y_val = splitting(validation)

plt.figure()
plt.imshow(train[0][0],cmap="gray")
plt.title(labels[0][1])
plt.axis("off")

#normalization

X_train =np.array(X_train)/255.0
X_test =np.array(X_test)/255.0
X_val =np.array(X_val)/255.0


#Noral network kanal sayısınıda ister

X_train = X_train.reshape(-1,img_size,img_size,1)
X_test = X_test.reshape(-1,img_size,img_size,1)
X_test = X_test.reshape(-1,img_size,img_size,1)

#liste kabul etmediği için numpy array'e çevirdik
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


# %% data augmentation


datagen = ImageDataGenerator(   #true iken geçerli aşağıdakiler
    featurewise_center = False, # verisetinin genel ortalamasını 0 yapar
    samplewise_center = False, # her bir örneğin ortalamasını sıfır yapar              
    featurewise_std_normalization = False, #veriyi verinin std sine bölme
    samplewise_std_normalization = False, #her bir örneği kendi std sine böler
    zca_whitening = False, #zca beyazlatma yöntemi verilerin korelasyonunu azaltır
    
    rotation_range = 30, #resimleri -30 ,+30 arası rastgele döndürür
    zoom_range = 0.2, # %80 ile %120 arası yakınlaştırır
    width_shift_range = 0.1, #resimleri yatayda rastgele kaydırır
    height_shift_range = 0.1, #/   /   dike       /       /
    horizontal_flip = True, #resimleri rastgele yatay olarak çevirir
    vertical_flip = True, #   /           /     dikey   /       /
    )

datagen.fit(X_train)

# %% create dl model

# conv2d - noramlization -maxpooling

model = Sequential()
model.add(Conv2D(128,(7,7),strides=(1,1),padding='same',activation="relu",input_shape=(img_size,img_size,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2),strides=(2,2)))

model.add(Conv2D(64,(5,5),strides=(2,2),padding='same',activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2),strides=(2,2)))

model.add(Conv2D(32,(3,3),strides=(1,1),padding='same',activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=["accuracy"])

#Learning rate'i kontrol edecek ve ayrıca kötü bir durum olursa durdurma işlemini yapcak
learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy",patience=2,verbose=1,factor=0.3,min_lr=0.000001)
epoch_number = 3

history = model.fit(datagen.flow(X_train,y_train,batch_size = 16),epochs= epoch_number,validation_data = datagen.flow(X_test,y_test),callbacks = [learning_rate_reduction],verbose=1)

ef=pd.DataFrame(history.history)
ef[['loss','val_loss']].plot()
ef[['accuracy','val_accuracy']].plot()


num_images = 36  # Number of images to display
y_pred = model.predict(X_test[:num_images])
y_pred_classes = np.argmax(y_pred, axis=1)

classes = ["PNEUMONIA", "NORMAL"]  # Class labels

plt.figure(figsize=(10, 10))
for i in range(num_images):
    plt.subplot(6, 6, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    true_label = classes[int(y_test[i])]
    y_pred_p = classes[int(y_pred_classes[i])]
    color = 'green' if true_label == y_pred_p else 'red'
    plt.title(f"True: {true_label}\nPred: {y_pred_p}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()