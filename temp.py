from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import math
import numpy as np
import model
from sklearn.model_selection import train_test_split
import random
batch_size = 5000
num_classes = 51
epochs = 5
data_augmentation = False
label_counter = 1

all_images = []
all_labels = []

def preprocess_input(B):
    for i in range(len(B)):
        img = (load_img(B[i])).resize((299,299))
        B[i] = img_to_array(img)
        B[i] = np.divide(B[i], 255.0)
        B[i]= np.subtract(B[i], 0.5)
        B[i] = np.multiply(B[i], 2.0)
    return B


for subdir, dirs, files in os.walk('newdata'):
    for folder in dirs:
        for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
            for file in folder_files:
                all_images.append(os.path.join(folder_subdir, file))
                all_labels.append(label_counter)

        label_counter = label_counter + 1


print(len(all_images))
print(len(all_labels))

perm = list(range(len(all_images)))
random.shuffle(perm)
all_images = [all_images[index] for index in perm]
all_labels = [all_labels[index] for index in perm]


training_images, test_images, training_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.3, random_state=1)

test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=1)

val_images = preprocess_input(val_images)
val_images=np.array(val_images)
val_labels = (keras.utils.to_categorical(val_labels, num_classes)).astype(int)
nice_n = math.floor(len(training_images) / batch_size) * batch_size

print(nice_n)
print("Data is ready...")

def get_batch(in_images,in_labels):
    while True:
        index = 1
        global current_index
        B = np.zeros(shape=(batch_size, 299, 299, 3))
        L = np.zeros(shape=(batch_size))
        while index < batch_size:
            try:
                img = (load_img(in_images[current_index])).resize((299,299))
                B[index] = img_to_array(img)
                B[index] = np.divide(B[index], 255.0)
                B[index]= np.subtract(B[index], 0.5)
                B[index] = np.multiply(B[index], 2.0)
    
                L[index] = in_labels[current_index]
    
                index +=1
                current_index += 1
            except:
                print("Ignoring {}".format(in_images[current_index]))
                current_index = current_index + 1
        if(current_index>nice_n-6):
            current_index =0
        yield B, (keras.utils.to_categorical(L, num_classes)).astype(int)

modl = model.create_model(num_classes=51, dropout_prob=0.2, weights=None, include_top=True)

modl.compile(loss = keras.losses.categorical_crossentropy,optimizer=keras.optimizers.RMSprop(lr=0.045,rho=0.94,epsilon=1,decay=0.9))

current_index =0
loss = modl.fit_generator(get_batch(training_images,training_labels),validation_data=(val_images,val_labels), steps_per_epoch=9, epochs = 5)
#val_loss = modl.fit_generator(get_batch(training_images,training_labels),steps_per_epoch=9, epochs = 1)