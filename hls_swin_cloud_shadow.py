import glob,random, rasterio
import matplotlib.pyplot as plt
import scipy.ndimage
import pandas as pd
#from predict_unet_pair_small import call_predictor
import shutil
import cv2
import gdal
import glob
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")
import  matplotlib
#matplotlib.use('Agg')
import multiprocessing
from multiprocessing import set_start_method
set_start_method('spawn', force=True)
from tensorflow.keras.models import load_model

def borders(arraybands, bordersize=128):
    border = cv2.copyMakeBorder(arraybands, top=bordersize, bottom=bordersize,\
    left=bordersize, right=bordersize,\
        borderType=cv2.BORDER_REFLECT)
    return border


def recall_m(y_true, y_pred):
    # Clip predictions and convert to booleans (1 if >= 0.5, 0 otherwise)
    y_pred = K.round(K.clip(y_pred, 0, 1))
    # Initialize variable to keep sum of recalls
    recall_sum = 0
    # Number of classes
    num_classes = K.int_shape(y_pred)[-1]

    for i in range(num_classes):
        # Per class true positives
        true_positives = K.sum(K.cast(K.equal(y_true[:, i] + y_pred[:, i], 2), 'float32'))
        # Per class possible positives (actual positives)
        possible_positives = K.sum(K.cast(K.equal(y_true[:, i], 1), 'float32'))
        
        # Compute recall for the current class and add it to recall_sum
        recall_class = true_positives / (possible_positives + K.epsilon())
        recall_sum += recall_class
    
    # Calculate average recall across all classes
    recall = recall_sum / num_classes
    return recall


def precision_m(y_true, y_pred):
    # Clip predictions and convert to booleans (1 if >= 0.5, 0 otherwise)
    y_pred = K.round(K.clip(y_pred, 0, 1))
    # Initialize variable to keep sum of precisions
    precision_sum = 0
    num_classes = K.int_shape(y_pred)[-1]

    for i in range(num_classes):
        # Per class true positives
        true_positives = K.sum(K.cast(K.equal(y_true[:, i] + y_pred[:, i], 2), 'float32'))
        # Per class predicted positives
        predicted_positives = K.sum(K.cast(K.equal(y_pred[:, i], 1), 'float32'))
        
        # Compute precision for the current class and add it to precision_sum
        precision_class = true_positives / (predicted_positives + K.epsilon())
        precision_sum += precision_class
    
    # Calculate average precision across all classes
    precision = precision_sum / num_classes
    return precision



def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    x=2*((precision*recall)/(precision+recall+K.epsilon()))

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



STRIDE=64

PATCH_SZ=128
BORDERSIZE=64

SR_SCALE = 10000.0

model_path='trained_model/trained_model'

model = load_model(model_path,custom_objects={'f1_m': f1_m})

N_BANDS=13


def call_code(filename):

    print(filename)

    fileroot=filename.split('/')[-1].split('_')

    fileroot=('_').join(fileroot[1:len(fileroot)-2])

    lab_file=glob.glob(inputdir+'label/*'+fileroot+'*tif')

    with rasterio.open(lab_file[0]) as f:
        ref_data=f.read()[0]


    with rasterio.open(filename) as f:
        hls_data=f.read()


    to_be_predict=hls_data[0:N_BANDS]

    fill=np.where(hls_data[0]==-9999)

    nband, height, width=to_be_predict.shape               
    
    to_be_predict=np.moveaxis(to_be_predict,0,2)

    im_arr=borders(to_be_predict,BORDERSIZE)
    
    im_arr=im_arr/SR_SCALE
    
    
    img_height, img_width, _ = im_arr.shape  # com borda
    prediction_img = np.zeros((img_height, img_width), 'int')-9999
    prediction_soft=np.zeros((img_height, img_width,4), 'float')-9999



    prediction_prob=np.zeros((img_height, img_width,8), 'float')
    prediction_img_1 = np.zeros((img_height, img_width), 'uint8')
    prediction_max = np.zeros((img_height, img_width), 'float')

    t1=time.time()

    for row in range(0, img_height - PATCH_SZ, STRIDE): 
        for col in range(0, img_width - PATCH_SZ, STRIDE):  
            patches_array = im_arr[row:row + PATCH_SZ, col:col + PATCH_SZ].copy()
            
            for iband in range(N_BANDS):
                tmp_data=patches_array[:,:,iband]
                good=np.where(tmp_data>-9999)
                bad=np.where(tmp_data==-9999)
                if len(bad[0])>0 and len(good[0])>0:
                    tmp_data[bad]=random.choices(tmp_data[good],k=len(bad[0]))
                    patches_array[:,:,iband]=tmp_data
    
            patches_array = patches_array.reshape(1, PATCH_SZ, PATCH_SZ, N_BANDS)

            patches_predict = \
                    np.array(model(patches_array, training=False))[0]

            classesi = np.argmax(patches_predict,axis=2).astype(np.uint8)
            
            tmp_soft=prediction_soft[row:row + PATCH_SZ, col:col + PATCH_SZ,:]

            tmp_classesi=prediction_img[row:row + PATCH_SZ, col:col + PATCH_SZ]

            tmp_non_fill=np.where(tmp_classesi!=-9999)
            tmp_fill    =np.where(tmp_classesi==-9999)

            #plot_without_border(cloudsen_palette[classesi+1])

            if len(tmp_fill[0])>0:
                tmp_classesi[tmp_fill]=-1

            if len(tmp_non_fill[0])>0:
                tmp_1=tmp_classesi[tmp_non_fill]
                tmp_2=classesi[tmp_non_fill]

                tmp_ori=tmp_1.copy()

                x3=np.where(np.logical_or(tmp_1==2, tmp_2==2))
                x2=np.where(np.logical_or(tmp_1==3, tmp_2==3))

                x1=np.where(np.logical_or(tmp_1==1, tmp_2==1))

                if len(x3[0])>0: tmp_ori[x3]=2
                if len(x2[0])>0: tmp_ori[x2]=3
                if len(x1[0])>0: tmp_ori[x1]=1

                classesi[tmp_non_fill]=tmp_ori
                                
            
            #plot_rgb_log(np.moveaxis(patches_array[0][:,:,0:3]*10000,2,0),-9999)
            #plot_without_border(cloudsen_palette[classesi+1])
            #plot_without_border(cloudsen_palette[tmp_classesi+1])
            #plt.show(block=False)
            #input()
            #plt.close('all')
            
            prediction_img[row:row + PATCH_SZ, col:col + PATCH_SZ] = classesi

            prediction_soft[row:row + PATCH_SZ, col:col + PATCH_SZ,:] = patches_predict

            prediction_max[row:row + PATCH_SZ, col:col + PATCH_SZ] =np.max(patches_predict,axis=2)

            #prediction_prob[row:row + PATCH_SZ, col:col + PATCH_SZ,:] =combine_prob

    pred_array = prediction_img[BORDERSIZE:img_height - BORDERSIZE, BORDERSIZE:img_width - BORDERSIZE]
    pred_soft =  prediction_soft[BORDERSIZE:img_height - BORDERSIZE, BORDERSIZE:img_width - BORDERSIZE]
    pred_max=prediction_max[BORDERSIZE:img_height - BORDERSIZE, BORDERSIZE:img_width - BORDERSIZE]

    pred_array=pred_array+1

    if len(fill[0])>0:
        pred_array[fill]=0

    ref_data=ref_data+1


    plt.figure()
    plt.imshow(pred_array)
    plt.figure()
    plt.imshow(ref_data)
    plt.show(block=False)

    input()

    plt.close('all')

    
if __name__=="__main__":


    inputdir='test_data/'

    filenames=glob.glob(inputdir+'img/*tif')

    filenames=sorted(filenames)

    for filename in filenames:
        call_code(filename)







