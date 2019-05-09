from tkinter import Text

import keras

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import UpSampling3D,Convolution3D,MaxPooling3D
from keras.layers import Input, Concatenate, Activation,Dropout,ZeroPadding3D
from keras.models import Model, model_from_json
from keras.optimizers import *
import numpy as np
import json



class SparseVox:
    def __init__(self):
        super().__init__()
        self.model = None
        self.batch_size=1
        self.last_best_epoch=0
        self.last_epoch=0

    # Initialize the model
    def init_model(self):

        Input_1 = Input(shape=(50, 50, 50, 1), name='Input_1')
        Input_2 = Input(shape=(50, 50, 50, 1), name='Input_2')

        Convolution3D_8 = Convolution3D(name='Convolution3D_8', kernel_dim3=5, nb_filter=16, kernel_dim1=5,
                                        kernel_dim2=5, border_mode='same')(Input_2)
        merge_1 = Concatenate(axis=4)([Input_1, Input_2])
        Convolution3D_1 = Convolution3D(name='Convolution3D_1', kernel_dim3=5, nb_filter=32, kernel_dim1=5,
                                        kernel_dim2=5, border_mode='same')(merge_1)
        MaxPooling3D_1 = MaxPooling3D(name='MaxPooling3D_1', pool_size=(8, 8, 8))(Convolution3D_1)
        Convolution3D_20 = Convolution3D(name='Convolution3D_20', kernel_dim3=5, nb_filter=1, kernel_dim1=5,
                                         kernel_dim2=5, border_mode='same')(Input_2)
        BatchNormalization_1 = BatchNormalization(name='BatchNormalization_1')(MaxPooling3D_1)
        Activation_1 = Activation(name='Activation_1', activation='relu')(BatchNormalization_1)
        Convolution3D_7 = Convolution3D(name='Convolution3D_7', kernel_dim3=5, nb_filter=32, kernel_dim1=5,
                                        kernel_dim2=5, border_mode='same')(Activation_1)
        Convolution3D_6 = Convolution3D(name='Convolution3D_6', kernel_dim3=5, nb_filter=32, kernel_dim1=5,
                                        kernel_dim2=5, border_mode='same')(Activation_1)
        Dropout_3 = Dropout(name='Dropout_3', p=0.2)(Convolution3D_6)
        Convolution3D_4 = Convolution3D(name='Convolution3D_4', kernel_dim3=5, nb_filter=32, kernel_dim1=5,
                                        kernel_dim2=5, border_mode='same')(Activation_1)
        Dropout_1 = Dropout(name='Dropout_1', p=0.2)(Convolution3D_4)
        Dropout_4 = Dropout(name='Dropout_4', p=0.2)(Convolution3D_7)
        MaxPooling3D_3 = MaxPooling3D(name='MaxPooling3D_3', pool_size=(8, 8, 8))(Convolution3D_8)
        Convolution3D_5 = Convolution3D(name='Convolution3D_5', kernel_dim3=5, nb_filter=32, kernel_dim1=5,
                                        kernel_dim2=5, border_mode='same')(Activation_1)
        Dropout_2 = Dropout(name='Dropout_2', p=0.2)(Convolution3D_5)
        merge_2 = Concatenate(axis=4)([Dropout_3, Dropout_2, Dropout_1, Activation_1, Dropout_4])
        merge_5 = Concatenate(axis=4)([MaxPooling3D_3, merge_2])
        BatchNormalization_3 = BatchNormalization(name='BatchNormalization_3')(merge_5)
        Activation_3 = Activation(name='Activation_3', activation='relu')(BatchNormalization_3)
        Convolution3D_9 = Convolution3D(name='Convolution3D_9', kernel_dim3=1, nb_filter=64, kernel_dim1=1,
                                        kernel_dim2=1, border_mode='same')(Activation_3)
        BatchNormalization_4 = BatchNormalization(name='BatchNormalization_4')(Convolution3D_9)
        Activation_4 = Activation(name='Activation_4', activation='relu')(BatchNormalization_4)
        Convolution3D_10 = Convolution3D(name='Convolution3D_10', kernel_dim3=5, nb_filter=16, kernel_dim1=5,
                                         kernel_dim2=5, border_mode='same')(Activation_4)
        Dropout_9 = Dropout(name='Dropout_9', p=0.2)(Convolution3D_10)
        merge_6 = Concatenate(axis=4)([Dropout_9, merge_5])
        BatchNormalization_5 = BatchNormalization(name='BatchNormalization_5')(merge_6)
        Activation_5 = Activation(name='Activation_5', activation='relu')(BatchNormalization_5)
        Convolution3D_11 = Convolution3D(name='Convolution3D_11', kernel_dim3=1, nb_filter=64, kernel_dim1=1,
                                         kernel_dim2=1, border_mode='same')(Activation_5)
        BatchNormalization_6 = BatchNormalization(name='BatchNormalization_6')(Convolution3D_11)
        Activation_6 = Activation(name='Activation_6', activation='relu')(BatchNormalization_6)
        Convolution3D_12 = Convolution3D(name='Convolution3D_12', kernel_dim3=5, nb_filter=16, kernel_dim1=5,
                                         kernel_dim2=5, border_mode='same')(Activation_6)
        Dropout_10 = Dropout(name='Dropout_10', p=0.2)(Convolution3D_12)
        merge_7 = Concatenate(axis=4)([Dropout_10, merge_6])
        BatchNormalization_7 = BatchNormalization(name='BatchNormalization_7')(merge_7)
        Activation_7 = Activation(name='Activation_7', activation='relu')(BatchNormalization_7)
        Convolution3D_13 = Convolution3D(name='Convolution3D_13', kernel_dim3=1, nb_filter=64, kernel_dim1=1,
                                         kernel_dim2=1, border_mode='same')(Activation_7)
        BatchNormalization_8 = BatchNormalization(name='BatchNormalization_8')(Convolution3D_13)
        Activation_8 = Activation(name='Activation_8', activation='relu')(BatchNormalization_8)
        Convolution3D_14 = Convolution3D(name='Convolution3D_14', kernel_dim3=5, nb_filter=16, kernel_dim1=5,
                                         kernel_dim2=5, border_mode='same')(Activation_8)
        Dropout_11 = Dropout(name='Dropout_11', p=0.2)(Convolution3D_14)
        merge_8 = Concatenate(axis=4)([Dropout_11, merge_7])
        BatchNormalization_9 = BatchNormalization(name='BatchNormalization_9')(merge_8)
        Activation_9 = Activation(name='Activation_9', activation='relu')(BatchNormalization_9)
        Convolution3D_15 = Convolution3D(name='Convolution3D_15', kernel_dim3=1, nb_filter=64, kernel_dim1=1,
                                         kernel_dim2=1, border_mode='same')(Activation_9)
        BatchNormalization_10 = BatchNormalization(name='BatchNormalization_10')(Convolution3D_15)
        Activation_10 = Activation(name='Activation_10', activation='relu')(BatchNormalization_10)
        Convolution3D_16 = Convolution3D(name='Convolution3D_16', kernel_dim3=5, nb_filter=16, kernel_dim1=5,
                                         kernel_dim2=5, border_mode='same')(Activation_10)
        Dropout_12 = Dropout(name='Dropout_12', p=0.2)(Convolution3D_16)
        merge_9 = Concatenate(axis=4)([Dropout_12, merge_8])
        BatchNormalization_11 = BatchNormalization(name='BatchNormalization_11')(merge_9)
        Activation_11 = Activation(name='Activation_11', activation='relu')(BatchNormalization_11)
        Convolution3D_17 = Convolution3D(name='Convolution3D_17', kernel_dim3=1, nb_filter=32, kernel_dim1=1,
                                         kernel_dim2=1, border_mode='same')(Activation_11)
        UpSampling3D_1 = UpSampling3D(name='UpSampling3D_1', size=(8, 8, 8))(Convolution3D_17)
        ZeroPadding3D_1 = ZeroPadding3D(name='ZeroPadding3D_1')(UpSampling3D_1)
        merge_10 = Concatenate(axis=4)([ZeroPadding3D_1, Convolution3D_20])
        BatchNormalization_12 = BatchNormalization(name='BatchNormalization_12')(merge_10)
        Activation_12 = Activation(name='Activation_12', activation='relu')(BatchNormalization_12)
        Convolution3D_18 = Convolution3D(name='Convolution3D_18', kernel_dim3=5, nb_filter=16, kernel_dim1=5,
                                         kernel_dim2=5, border_mode='same')(Activation_12)
        merge_11 = Concatenate(axis=4)([Convolution3D_18, merge_10])
        BatchNormalization_13 = BatchNormalization(name='BatchNormalization_13')(merge_11)
        Activation_13 = Activation(name='Activation_13', activation='relu')(BatchNormalization_13)
        Convolution3D_19 = Convolution3D(name='Convolution3D_19', kernel_dim3=9, nb_filter=1, kernel_dim1=9,
                                         kernel_dim2=9, border_mode='same')(Activation_13)
        Activation_14 = Activation(name='Activation_14', activation='sigmoid')(Convolution3D_19)

        model = Model([Input_1,Input_2], [Activation_14])

        print(model.summary())
        self.model = model
        return model

    # Return the model optimizer
    def get_optimizer(self):
        return RMSprop(lr=1e-3)#Adam()

    # Return the model Loss function
    def get_loss_function(self):
        return 'mean_squared_error'

    # Return the Batch size
    def get_batch_size(self):
        return 1

    # Return the default number of epochs
    def get_num_epoch(self):
        return 100

    # Load model and weights from disk
    def load_model_and_weight(self, model_name):
        # load model
        json_file = open(model_name + '.json', 'r')
        model = json_file.read()
        json_file.close()
        model = model_from_json(model)
        # load weights into model
        model.load_weights(model_name + ".h5")
        print("Loaded model from disk")
        self.model = model
        fp = open(model_name+'_settings.txt', "r")
        data = json.load(fp)
        self.last_epoch = data['last_epoch']
        self.last_best_epoch = data['last_best_epoch']
        print(model.summary())


    # Save model and weights into model directory
    def save_model_and_weight(self, model_name,last_epoch,last_best_epoch):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_name + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_name + '.h5')
        print("Saved model to disk")

        data = {'last_epoch': last_epoch, 'last_best_epoch': last_best_epoch}
        with open(model_name+'_settings.txt', 'w') as fp:
            json.dump(data, fp)

    # Compile the model
    def compile(self):
        self.model.compile(optimizer=self.get_optimizer(), loss=self.get_loss_function(), metrics=['accuracy'])
        return self.model

    # Train the model
    def train(self, x, y, n_epoch=20, batch_size=1):
        self.batch_size=batch_size
        self.model.fit(x, y, epochs=n_epoch, batch_size=batch_size, verbose=1,shuffle=True)

    # Check the error rate on its input test data (x_test & y_test) and print the result in consule
    def get_error_rate(self, x_ts, y_ts,batch_size=8):
        p = self.model.predict(x_ts, batch_size=batch_size, verbose=0)
        mse = np.mean(np.square(y_ts - p))
        print("Error rate is " + str(mse))
        return mse
