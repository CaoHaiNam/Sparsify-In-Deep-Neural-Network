import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical

from tfkerassurgeon import identify
from tfkerassurgeon.operations import delete_channels
import tensorflow.compat.v1 as tf
from tensorflow.python.keras import callbacks
from tfkerassurgeon.identify import get_apoz
from tfkerassurgeon import Surgeon

tf.disable_v2_behavior()
tf.reset_default_graph()
# Set some static values that can be tweaked to experiment
keras_verbosity = 2 # limits the printed output but still gets the Epoch stats
epochs=100 # we'd never reach 200 because we have early stopping
batch_size=1024 # tweak this depending on your hardware and Model


# Load the dataset (it will automatically download it if needed), they provided a nice helper that does all the network and downloading for you
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)


# input image dimensions
nb_filters = 64
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
nb_classes = 10 # OR len(set(train_labels))

# Simple reusable shorthand to compile the model, so that we can be sure to use the same optomizer, loss, and metrics
def compile_model(model):    
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
def get_callbacks(use_early_stopping = True, use_reduce_lr = False):
    callback_list = []
    if(use_early_stopping):
        callback_list.append(callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0,
                                             patience=10,
                                             verbose=keras_verbosity,
                                             mode='auto'))
    if(use_reduce_lr):
        callback_list.append(callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            patience=5,
                                            verbose=keras_verbosity,
                                            mode='auto',
                                            epsilon=0.0001,
                                            cooldown=0,
                                            min_lr=0))

    return callback_list

# and get the callbacks
callback_list = get_callbacks()


# method that encapsulates the Models archeteture and construction
def build_model():
    # Create LeNet model
    model = Sequential()
    model.add(Convolution2D(input_shape=(32,32,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Convolution2D(filters=64,kernel_size=(3,3),padding="same", activation="relu",name='conv_1'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",name='conv_2'))
    model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same", activation="relu",name='conv_3'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name='conv_4'))
    model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu",name='conv_5'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='conv_6'))
    model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu",name='conv_7'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(512,name='dense_1'))
    model.add(Activation('relu'))
    model.add(Dense(512,name='dense_2'))
    model.add(Activation('relu'))
    # FC-30 Last layer
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    compile_model(model)
    return model

def fit_model(model):
    return model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=keras_verbosity,
        validation_data=(X_test, Y_test),
        callbacks=callback_list)


# Simple reusable shorthand for evaluating the model on the Validation set 
def eval_model(model):
    return model.evaluate(
                        X_test, 
                        Y_test, 
                        batch_size=batch_size, 
                        verbose=keras_verbosity)


# THIS IS WHERE THE MAGIC HAPPENS!
# This method uses the Keras Surgeon to identify which parts od a layer can be pruned and then deletes them
# Note: it returns the new, pruned model, that was recompiled


def prune_layer(model, layer):
    # Get the APOZ (Average Percentage of Zeros) that should identify where we can prune
    apoz = identify.get_apoz(model, layer, X_test)
    print(apoz)
    # Get the Channel Ids that have a high APOZ, which indicates they can be pruned
    # high_apoz_channels = identify.high_apoz(apoz,cutoff_std=0.8)
    high_apoz_channels = identify.high_apoz(apoz,method="absolute", cutoff_absolute=0.999)

    # print("============", high_apoz_channels)
    # Run the pruning on the Model and get the Pruned (uncompiled) model as a result
    model = delete_channels(model, layer, high_apoz_channels)
    # Recompile the model
    compile_model(model)
    return model

# A helper that gets the layer by it's name 
def prune_layer_by_name(model, layer_names):
    for layer_name in layer_names:
       layer = model.get_layer(name=layer_name)
       model = prune_layer(model, layer)
    # Then prune is and return the pruned model
    return model

# def prune_layer_by_name(model, layer_name):

#     # First we get the layer we are working on
#     layer = model.get_layer(name=layer_name)
#     # Then prune is and return the pruned model
#     return prune_layer(model, layer)

# the main function, that runs the training
def main(): 
    # build the model
    model = build_model()
    print(model.summary())
    # Initial Train on dataset
    results = fit_model(model)
    # eval and print the results of the training
    loss = eval_model(model)
    print('original model loss:', loss, '\n')
    for i in range(2):
        layer_names = ['conv_7','dense_1']
        # layer_name = 'dense_1'
        model = prune_layer_by_name(model, layer_names)
        print(model.summary())
        # eval and print the results of the pruning
        loss = eval_model(model)
        print('model loss after pruning: ', loss, '\n')
        # Retrain the model to accomodate for the changes
        results = fit_model(model)
        # eval and print the results of the retraining
        loss = eval_model(model)
        print('model loss after retraining: ', loss, '\n')
        # While TRUE will repeat until an ERROR occurs

# Run the main Method
if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    main()




