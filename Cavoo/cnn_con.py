import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks

os.chdir('C:\\Users\\royal\\Downloads\\Compressed\\dataset52bd6ce')

epochs = 2
train_data_path = 'dataset\\train'
validation_data_path = 'dataset\\val'

"""Parameters"""
img_width, img_height = 224, 224
batch_size = 32
#samples_per_epoch = 1000
samples_per_epoch = 2
validation_steps = 1
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 15
lr = 0.0004

model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=lr),metrics=['accuracy'])

#train_datagen = ImageDataGenerator(
#    rescale=1. / 255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True)

train_datagen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    width_shift_range = 0.2,
    height_shift_range = 0.2)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    class_mode='categorical')


model.fit_generator(
    train_generator,verbose=1,
    steps_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=2)

model.save('model_cnn.h5')
model.save_weights('model_cnn_weights.h5')
