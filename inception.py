from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import os
import numpy as np
from sklearn.model_selection import train_test_split
import math
import cv2
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

batch_size = 32

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))

# data_folder = 'test_for_classification/stegnet_ucf_240_320_0.75_test/'

# cover_container_dirs = []
# for name in os.listdir(data_folder):
#   splits = name.split('_')
    
#   dir_cover = None
#   if splits[1] == 'test':
#       dir_cover = 'ucf/test_all/'+splits[3]+'/'
#   else:
#       dir_cover = 'ucf/train/'+splits[2]+'/'

#   dir_container = data_folder+name+'/container/'

#   cover_container_dirs.append([dir_cover, dir_container])

# # cover_container_dirs = [['ucf/train/' + name.split('_')[1] + '/', data_folder+name+'/container/'] for name in os.listdir(data_folder)]
# class_cover = np.array([1, 0])
# class_container = np.array([0, 1])
# all_samples = []
# for i in range(len(cover_container_dirs)):
#   for file in os.listdir(cover_container_dirs[i][1]):
#       all_samples.append((cover_container_dirs[i][1]+file, class_container))
#       name = str(int(file.split('.')[0]) + 1)
#       if len(name) == 1:
#           file = '00'+name+'.jpg'
#       elif len(name) == 2:
#           file = '0'+name+'.jpg'
#       else:
#           file = name+'.jpg'
#       all_samples.append((cover_container_dirs[i][0]+file, class_cover))

# np.random.shuffle(all_samples)
# X = []
# y = []
# for x_i, y_i in all_samples:
#   X.append(x_i)
#   y.append(y_i)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train = [(X_train[i], y_train[i]) for i in range(len(X_train))]
# num_training_samples = len(train)
# test = [(X_test[i], y_test[i]) for i in range(len(X_test))]
# num_test_samples = len(test)
train = pickle.load(open('train_classification_set.pkl', 'rb'))
np.random.shuffle(train)
num_training_samples = len(train)
test = pickle.load(open('test_classification_set.pkl', 'rb'))
num_test_samples = len(test)

def generator(data):
    ind = 0
    while True:
        image_batch = []
        label_batch = []
        for i in range(batch_size):
            if ind == len(data):
                ind = 0
                np.random.shuffle(data)
            image_batch.append(cv2.imread(data[ind][0]) / 255.0)
            label_batch.append(data[ind][1])
            ind += 1
        yield np.array(image_batch), np.array(label_batch)


input_tensor = Input(shape=(240, 320, 3))

# create the base pre-trained model
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dense(256, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='adam', loss='categorical_crossentropy')
# print(model.summary())

# train the model on the new data for a few epochs

# model = load_model('models_classification/inception.h5')
# for i, layer in enumerate(model.layers):
#    print(i, layer.name)

# for layer in model.layers:
#     layer.trainable = True

model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

# filepath = 'models_classification/inception_full_trained.h5'
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

history = LossHistory()
# callbacks_list = [checkpoint, history]
model.fit_generator(generator(train), epochs=1, steps_per_epoch=math.ceil(num_training_samples / batch_size), 
                    # validation_data=generator(test), validation_steps=math.ceil(num_test_samples / batch_size),
                    callbacks=[history])

pickle.dump(history.accs, open('models_classification/inception_full_history_accs.pkl', 'wb'))
pickle.dump(history.losses, open('models_classification/inception_full_history_losses.pkl', 'wb'))

model.save('models_classification/inception_full_trained.h5')
results = model.evaluate_generator(generator(test), steps=math.ceil(num_test_samples / batch_size), verbose=1)
pickle.dump(results, open('inception_full_res.pkl', 'wb'))
print (results)
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:

# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# model.fit_generator(...)