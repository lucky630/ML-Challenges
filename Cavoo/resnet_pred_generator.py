#from keras.models import load_model
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
import os
import numpy as np
import cv2
import csv

os.chdir('C:\\Users\\royal\\Downloads\\Compressed\\dataset52bd6ce')

model = load_model('resnet_model.h5')

print('loaded')

dd={}
dd1 = {0: '0', 1: '1', 2: '10', 3: '11', 4: '12', 5: '13', 6: '14', 7: '2', 8: '3', 9: '4',
       10: '5', 11: '6', 12: '7', 13: '8', 14: '9'}

data_generator = ImageDataGenerator(rescale=1./255)

image_size = 224

test_generator = data_generator.flow_from_directory(
        'dataset\\aa',
        target_size=(image_size, image_size),
        class_mode='categorical')

filenames = test_generator.filenames
nb_samples = len(filenames)
print(nb_samples)

predict = model.predict_generator(test_generator,steps = nb_samples / 22)

print(predict)
print(predict.shape)
print(dd)

print(predict.argmax(axis=1))
