import gdal
import os
import numpy as np
import tensorflow.keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
#import unet
import SwinUnet
#import unet_test
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def  decay_schedule(epoch, lr):
    if (epoch % 5 == 0) and (epoch != 0):
        lr = lr * 0.1
    return lr

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, path_dataset_img, path_dataset_label, patch_fmt_file='.tif', batch_size=64, dim=(128, 128), n_channels=3,
                 n_classes=5, scale_factor=10000, shuffle=True):
        '''

        :param list_IDs: lista of name of files
        :param batch_size: Como vou ler o meu banco de dados: batch 64
        :param dim: 16 x 16 pixels
        :param n_channels: rgb 3 bands
        :param n_classes: land covers
        :param path_dataset: where are the 1 million images? path
        :param shuffle:
        '''

        'Initialization'
        self.dim = dim
        self.path_dataset_img = path_dataset_img
        self.path_dataset_label = path_dataset_label
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.patch_fmt_file = patch_fmt_file
        self.scale_factor = scale_factor
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        #print('data_gene: index=', index, 'batch_size=', self.batch_size)

        #print('data_gene:self.indexes = ', len(self.indexes))
        #print('--------------------------------')



        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        datagen = ImageDataGenerator(\
                horizontal_flip=True,
                rotation_range=20,
                vertical_flip=True)

        results= datagen.flow(X, y)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        global array_label
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        # Generate data
        #
        for i, patch_name in enumerate(list_IDs_temp):
            # Store sample
            patch_name = str(patch_name)

            # X INPUT
            if self.patch_fmt_file == '.tif':
                path_img = os.path.join(self.path_dataset_img, 'img_'+patch_name + self.patch_fmt_file)

                ds1 = gdal.Open(path_img)
                loaded = ds1.ReadAsArray().astype(np.float) / float(self.scale_factor)

                #good_index=[1,2,3,8,10,11,12]

                #loaded=loaded[good_index]

                #loaded=loaded[:,0:256,0:256]

                X[i,] = np.moveaxis(loaded, 0, 2)

            # Store class - Y INPUT
            if self.patch_fmt_file == '.tif':
                path_label = os.path.join(self.path_dataset_label, 'burn_'+patch_name + self.patch_fmt_file)
                ds2 = gdal.Open(path_label)
                array_label = ds2.ReadAsArray().astype(np.uint8)

                #print(array_label.shape)

                #array_label=array_label[0:256,0:256]

            array_multi = np.zeros((self.dim[0], self.dim[1], self.n_classes), 'uint8')

            for x in range(0, self.n_classes):
            #for x in range(1,self.n_classes+1):

                a_mask = np.where(array_label == x, 1, 0)   
                array_multi[:, :, x] = a_mask

            y[i,] = np.array(array_multi)

        #X=tf.data.Dataset.from_tensor_slices(X)
        #y=tf.data.Dataset.from_tensor_slices(y)

        return X, y




