import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
print(train.shape)

test = pd.read_csv("test.csv")
print(test.shape)

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis=1)

print(Y_train.value_counts())

plt.figure(figsize=(15,7))
g = sns.countplot(x = Y_train,palette="icefire")
plt.title("Number of Digit Classes")
plt.show()

img = X_train.iloc[0].to_numpy()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()

img = X_train.iloc[3].to_numpy()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train.iloc[3,0])
plt.axis("off")
plt.show()

X_train = X_train / 255.0
test = test / 255.0
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)

plt.imshow(X_train[2][:,:,0],cmap='gray')
plt.axis('off')
plt.show()

from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau


model = Sequential()

model.add(Input(shape=(28,28,1)))

model.add(Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()

optimizer = Adam(learning_rate=0.001,beta_1 = 0.9,beta_2=0.999)

model.compile(optimizer= optimizer, loss ='categorical_crossentropy',metrics =['accuracy'])




















