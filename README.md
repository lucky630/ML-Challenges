# ML-Challenges

Clip art of DeepFashion dataset
Let’s quickly give more information about the DeepFashion dataset. DeepFashion is an open-source (commercial use not allowed by release agreement) dataset which is created for IEEE Conference on Computer Vision and Pattern Recognition (CVPR) in 2016 by Ziwei Liu, Ping Luo and their colleagues in The Chinese University of Hong-Kong and Shenzhen Institutes of Advanced Technology.

This dataset consists of more than 800.000 different RGB-colored images -ranging from well-posed shopping images to unstructured customer images & cross-posed and cross-domain images-. The size of images are not same as in the other well-known datasets and each image in the dataset is labeled with one of ~50 categories, ~1000 attributes, bounding box and clothing landmarks. In this work, I just focused on category classification and bounding box detection tasks using such a large subset (~290.000 images) of this well-prepared dataset which contains clothing categories and attributes in wild.

After downloading the dataset from here, we need to prepare the category labels by adding images to folders as images with same label in the same folder. Besides, we need to split the data into train, validation and test sets as annotated in the paper. But first, importing the libraries -please-.

import shutil
import os
import re
import cv2
# will use them for creating custom directory iterator
import numpy as np
from six.moves import range
# regular expression for splitting by whitespace
splitter = re.compile("\s+")
base_path = '<FOLDER_OF_IMAGES_THAT_YOU_DOWNLOADED>'
Then,

##image_name evaluation_status
##img/Sheer_Pleated-Front_Blouse/img_00000001.jpg train
def process_folders():
    # Read the relevant annotation file and preprocess it
    # Assumed that the annotation files are under '<project folder>/data/anno' path
    with open('./data/anno/list_eval_partition.txt', 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]
		
	##list_all have [(name after img/ , last_name_like (Blouse) , (train,test,val) )]
	
    # Put each image into the relevant folder in train/test/validation folder
    for element in list_all:
	
		## if train,test,val folder not there then it will create that.
        if not os.path.exists(os.path.join(base_path, element[2])):
            os.mkdir(os.path.join(base_path, element[2]))
		
		## will create folder with element last name,Blouse.
        if not os.path.exists(os.path.join(os.path.join(base_path, element[2]), element[1])):
            os.mkdir(os.path.join(os.path.join(base_path, element[2]), element[1]))
		
		## will create folder like blouse category Sheer_pleeted-Front_Blouse.
        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                              element[0].split('/')[0])):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                     element[0].split('/')[0]))
        shutil.move(os.path.join(base_path, element[0]),
                    os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1]), element[0])
process_folders()

##Folder Structure After process_folder:-
Train:
	Blouse:
		Sheer_pleeted-Front_Blouse
Val:
	Blouse:
		Sheer_pleeted-Front_Blouse
Test:
	Blouse:
		Sheer_pleeted-Front_Blouse

We need to extract the bounding box information from the annotation file and to normalize the values of bounding box information by the shape of the related image.

def create_dict_bboxes(list_all, split='train'):
	##only taking element which equal split,that is train,test,val. total 4 elements are in each tuple.
	##list_all - [(image_name, name last part like(Blouse), eval_type like(train,test,val), (xmin,ymin,xmax,ymax)) ,...]
    lst = [(line[0], line[1], line[3], line[2]) for line in list_all if line[2] == split]
	##lst - [(image_name, name last part like(Blouse), (xmin,ymin,xmax,ymax), eval_type like(train,test,val)) ,...]
	
	##[('img' + '/' + 'test,train,val' + '/' + (name_last_part + name[3:])) , name_last_part like Blouse , (xmin,ymin,xmax,ymax)]
    lst = [("".join(line[0].split('/')[0] + '/' + line[3] + '/' + line[1] + line[0][3:]), line[1], line[2]) for line in lst]
	##read image using first element and get Shape.
    lst_shape = [cv2.imread('./data/' + line[0]).shape for line in lst]
	##normalize the xmin & xmax with width,ymin & ymax with height.
    lst = [(line[0], line[1], (round(line[2][0] / shape[1], 2), round(line[2][1] / shape[0], 2), round(line[2][2] / shape[1], 2), round(line[2][3] / shape[0], 2))) for line, shape in zip(lst, lst_shape)]
	##get dictionary {image_name : {'x1':100,'y1':100,'x2':100,'y2':100,'shape':} }
    dict_ = {"/".join(line[0].split('/')[2:]): {'x1': line[2][0], 'y1': line[2][1], 'x2': line[2][2], 'y2': line[2][3], 'shape': line[2][4]} for line in lst}
    return dict_

def get_dict_bboxes():
    with open('./data/anno/list_category_img.txt', 'r') as category_img_file, \
            open('./data/anno/list_eval_partition.txt', 'r') as eval_partition_file, \
            open('./data/anno/list_bbox.txt', 'r') as bbox_file:
		##first two lines in the file are count and header so skip those.
        list_category_img = [line.rstrip('\n') for line in category_img_file][2:]
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_bbox = [line.rstrip('\n') for line in bbox_file][2:]

        list_category_img = [splitter.split(line) for line in list_category_img]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_bbox = [splitter.split(line) for line in list_bbox]

        list_all = [(k[0], k[0].split('/')[1].split('_')[-1], v[1], (int(b[1]), int(b[2]), int(b[3]), int(b[4])))
                    for k, v, b in zip(list_category_img, list_eval_partition, list_bbox)]
		##list_all is in this format.
		##list_all - [(image_name, name last part like(Blouse), eval_type like(train,test,val), (xmin,ymin,xmax,ymax)) , ...]
        list_all.sort(key=lambda x: x[1])
		##sort by second element in tuple that is blouse.

        dict_train = create_dict_bboxes(list_all)
        dict_val = create_dict_bboxes(list_all, split='val')
        dict_test = create_dict_bboxes(list_all, split='test')

        return dict_train, dict_val, dict_test
Too much work for preprocessing, huh?…

And now, we can import Keras-things.

from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

In our case, we will use 50-layer residual network (ResNet50) model pre-trained with ImageNet, but we will not train all layers in this model from scratch. After freezing the earlier layers which represent low-level features as weights such as line detector and pattern detector, we will train the layers which represent higher level features -more specific to data- by optimizing the loss function with low learning rate.

Less parameters to train
Less time for training
Preserving the lower level feature weights while fine-tuning the data-specific feature weights
Eliminating the possibility of getting stucked on local minima for the loss function during the early stage of the training
Just write this code snippet to get pre-trained ResNet50 model in Keras.

model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
Not including at top?? What does that mean?
For ImageNet dataset, there are different 1000 labels to categorize the images. Thus, when you want to train a model with ImageNet dataset, you need to specify the number of neurons in the output (softmax) layer as 1000. However, we have such a dataset with ~50 -actually, 46- labels to categorize the images. We should not include the top (output, softmax, last, whatever you would like to call) layer of ResNet50 for our model, so we can add a new layer and specify the number of neurons as what the dataset needs.

As I mentioned before, we need to freeze some layers in the very first part of the model. Freezing a layer means that -simply- making it not trainable in the model.

for layer in model_resnet.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    layer.trainable = False
Now, let’s build the category classification branch in the model.

x = model_resnet.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)
Then, we will build the bounding box detection branch in the model.

x_bbox = model_resnet.output
x_bbox = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
x_bbox = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)
Finally, we will create our final model by specifying the input and outputs for the branches.

final_model = Model(inputs=model_resnet.input,
                    outputs=[y, bbox])
The summary of our transfer learning model could be seen as:

print(final_model.summary())
The summary of trainable part of our transfer learning model
It could be seen that the number of trainable parameters in our custom ResNet50-like model are almost 25% percent of total number of parameters in original ResNet50 since we have already frozen the bunch of layers that contains low-level feature information and we will be training just last 12 layers.

To train a transfer learning model is hard to optimize. I am -still- working on how the optimization methods effect the training process and loss function for transfer learning approach. I will use Stochastic Gradient Descent (SGD) algorithm to optimize the weights in the backpropagation in order to make sure that I am on the safe side. Set the momentum parameter as 0.9 and the nesterov parameter as True. I strongly recommend you to read an article, this one, to get more information about SGD algorithm.

opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)
Why do we keep the learning rate too low?
The answer is simple. We want to not change the weights by destroying the information coming from the ImageNet and to learn something from the data. If you use default learning value, for example, the loss function will converge too fast and start to over-fit the training set.

We are ready now to compile our model. While categorical crossentropy method has been picked as loss function for category classification task, mean squared error method has been picked as loss function for bounding box detection task -you can pick either mean squarred logarithmic error-. Likewise, we will measure our performance on the validation set with top-1 and top-5 accuracies for category classification, and mean squarred error for bounding box detection.

final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy',
                          'bbox': 'mean_squared_error'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                             'bbox': ['mse']})
We have some problems here. How do we load our data without being out of bounds for memory? Also, how do we give such an input which contains an image, category label and bounding box together? Let’s figure it out!

Loading the data:
If you try to load all images with at least 100x100 size to your less than 64 GB memory, it will be out of bounds for memory. The solution is flowing the images as a batch from the directory -ImageDataGenerator class in Keras-, it will load the data and give it to the model as batch-to-batch. Besides, we can augment the data in real time by helping of this method. At the end, we need to create ImageDataGenerator objects for training and test sets.

train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()
Note that if you use normalization methods (feature-wise/sample-wise std normalization) to augment the data, you have to fit and transform the data before starting to train the data.

Manipulating the batch iterator:
This part could be seen as the most challenging part of this episode, but actually it is not. In ImageDataGenerator class, the method of flowing the data from the directory uses a DirectoryIterator object to iterate the data over the directory. We have to extend a custom object from the original DirectorIterator object. For the original one, GO.

class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes: dict = None, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = 'categorical', batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix: str = '', save_format: str = 'jpeg', follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes

    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'], bounding_box['y2']],
                    dtype=K.floatx())
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 46), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y
Bold parts of the code above have been added to DirectoryIterator object to reach category labels and bounding box information at the same time.
It is mentioned that the size of the images in the dataset are not same, so we need to set a target size for the images in the iterator objects.

dict_train, dict_val, dict_test = get_dict_bboxes()
train_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/train", train_datagen, bounding_boxes=dict_train, target_size=(200, 200))
test_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/val", test_datagen, bounding_boxes=dict_val,target_size=(200, 200))

Add some helpful features to our model. First, we will define a learning rate reducer in order to get rid of the plateaus in the loss function. Also, we will record what our model has done during the training process. Next, we will make sure that the training will be stopped if there is no change in the value of the loss function on the validation set for a certain epoch. Finally, we will save our trained model in each epoch that has better result than previous one.

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir='./logs')
early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
checkpoint = ModelCheckpoint('./models/model.h5')
Edit: Thank you to Killian, we have to create a custom generator object which yields the batches of images to the model.

def custom_generator(iterator):
    while True:
        batch_x, batch_y = iterator.next()
        yield (batch_x, batch_y)

final_model.fit_generator(custom_generator(train_iterator),
                          steps_per_epoch=2000,
                          epochs=200, validation_data=custom_generator(test_iterator),
                          validation_steps=200,
                          verbose=2,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          workers=12)

Well, after early stopping at between 140 and 145th epochs, we can measure the performance of our model on the test set.

test_datagen = ImageDataGenerator()

test_iterator = DirectoryIteratorWithBoundingBoxes("./data/img/test", test_datagen, bounding_boxes=dict_test, target_size=(200, 200))
scores = final_model.evaluate_generator(custom_generator(test_iterator), steps=2000)

print('Multi target loss: ' + str(scores[0]))
print('Image loss: ' + str(scores[1]))
print('Bounding boxes loss: ' + str(scores[2]))
print('Image accuracy: ' + str(scores[3]))
print('Top-5 image accuracy: ' + str(scores[4]))
print('Bounding boxes error: ' + str(scores[5]))
Results
~85% accuracy on top-5 predictions. WOW!

~<0.05 error on bounding box regression. WOW!

Of course, the results could be improved by increasing the number of augmentation methods and hyperparameter optimization in a certain range, but we are still very close to the results in the paper. It is -definitely- the time to celebrate.
