#from tensorflow.python.keras.applications import ResNet50,InceptionV3,VGG16
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout

#from tensorflow.python.keras.applications.resnet50 import preprocess_input
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, ResNet50

from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras import optimizers, Input

import os

os.chdir('C:\\Users\\royal\\Downloads\\Compressed\\dataset52bd6ce')

num_classes = 15

inception_resnet_weight = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_weights_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

main_model = InceptionResNetV2(include_top=False, weights=inception_resnet_weight, input_tensor=Input(shape=(224, 224, 3)))
#main_model = ResNet50(include_top=False, weights=resnet_weights_path, input_tensor=Input(shape=(224, 224, 3)))
main_model.trainable = False
for layer in main_model.layers[:-2]:
    layer.trainable = False

top_model = Sequential()
top_model.add(Flatten(input_shape=main_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.3))
top_model.add(Dense(num_classes, activation='sigmoid'))
my_new_model = Model(inputs=main_model.input,outputs=top_model(main_model.output))
opt = optimizers.Adam(lr=0.001, decay=0.1)
my_new_model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

#my_new_model = Sequential()
#my_new_model.add(InceptionV3(include_top=False, weights=inception_weights_path))
#my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
#my_new_model.add(Flatten())
#my_new_model.add(Dropout(0.1))
#my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
#for layer in my_new_model.layers[0].layers[:-4]:
#    layer.trainable = False

#my_new_model.layers[0].trainable = False

#categorical_crossentropy
#my_new_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#my_new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

image_size = 224

data_generator_with_aug = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        width_shift_range = 0.2,
        rescale=1./255,
        height_shift_range = 0.2)

data_generator_with_aug = ImageDataGenerator(rescale=1./255)

train_generator = data_generator_with_aug.flow_from_directory(
        'dataset\\train',
        target_size=(image_size, image_size),
        batch_size=24, class_mode='categorical')

label_map = (train_generator.class_indices)

print(label_map)

data_generator = ImageDataGenerator(rescale=1./255)

validation_generator = data_generator.flow_from_directory(
        'dataset\\val',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator, verbose=1,
        steps_per_epoch=1,
        epochs=15,
        validation_data=validation_generator,validation_steps=1)

# serialize weights to HDF5
mod_weight = 'incep_resnet_model_weight.h5'
mod_only = 'incep_resnet_model.h5'
#mod_weight = 'resnet_model_weight.h5'
#mod_only = 'resnet_model.h5'
#mod_weight = 'inception_model_weight.h5'
#mod_only = 'inception_model.h5'

my_new_model.save_weights(mod_weight)
print("Saved weight to disk")
my_new_model.save(mod_only)
print("Saved model to disk")

data_generator = ImageDataGenerator(rescale=1./255)
image_size = 224
test_generator = data_generator.flow_from_directory(
        'dataset\\aa',
        target_size=(image_size, image_size),
        batch_size=1,
        class_mode='categorical')

filenames = test_generator.filenames
nb_samples = len(filenames)
print(nb_samples)

print('test prediction')
predict = my_new_model.predict_generator(test_generator,steps = nb_samples)

#print(predict)
print(predict.shape)

pred = predict.argmax(axis=1)
print(pred.shape)
print(pred)

dd={}
dd1 = {0: '0', 1: '1', 2: '10', 3: '11', 4: '12', 5: '13', 6: '14', 7: '2', 8: '3', 9: '4',
       10: '5', 11: '6', 12: '7', 13: '8', 14: '9'}

writ = open('sub.csv','w')
writ.writelines('image_name,category'+'\n')

for j,i in enumerate(filenames):
    writ.writelines(i+','+ dd1[pred[j]] + '\n')
writ.close()
print('closed sub file..')


writ_prob = open('sub_prob.csv','w')
writ_prob.writelines('image_name,category'+'\n')

for j,i in enumerate(filenames):
    writ_prob.writelines(i+',')
    for k in predict[j]:
        writ_prob.writelines(str(k)+',')
    writ_prob.writelines('\n')
writ_prob.close()
print('closed prob file')
