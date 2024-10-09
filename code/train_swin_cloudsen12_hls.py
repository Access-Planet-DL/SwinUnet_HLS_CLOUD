import boto3
import pickle
import logging
import gdal, rasterio,os, glob,string
import random
import matplotlib.pyplot as plt
import numpy as np
from rasterio.transform import Affine
import tensorflow as tf
from  tensorflow.keras.callbacks import ReduceLROnPlateau
tf.config.optimizer.set_jit(False) # Start with XLA disabled.
tf.config.threading.set_inter_op_parallelism_threads(4)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import Precision
from keras import backend as K
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from tensorflow.keras.callbacks import CSVLogger
from keras import backend as K


from data_gene_cloudsen_hls  import DataGenerator
from warmup_cosine_decay_scheduler import *
import SwinUnet


csv_log_path = 'train_swin_cloudsen12_hls.csv'

csv_logger = CSVLogger(csv_log_path, append=True)

TMP_IMG_SIZE=128


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
    # Number of classes
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

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        # return optimizer.lr
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr
        return current_lr


def lr_warmup_cosine_decay(global_step, warmup_steps, hold=0, total_steps=0, target_lr=1e-3, start_lr=0.0):
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    global_step_tf = tf.cast(global_step, tf.float32)
    learning_rate = 0.5 * target_lr * (1 + tf.cos(tf.constant(np.pi) * (global_step_tf - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))
    
    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = tf.cast(target_lr * (global_step / warmup_steps), tf.float32)
    
    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold, learning_rate, target_lr)
    
    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    #print('learning_rate=', learning_rate)
    return tf.cast(learning_rate, tf.float32)

## https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold
    
    def __call__(self, step):
        lr = lr_warmup_cosine_decay(global_step=step, total_steps=self.total_steps, warmup_steps=self.warmup_steps, start_lr=self.start_lr, 
            target_lr=self.target_lr, hold=self.hold)
        
        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")
    
    def get_config(self):
        # config=super().get_config().copy()
        # config.update({
                  # 'start_lr':self.start_lr,
                  # 'target_lr':self.target_lr,
                  # 'warmup_steps':self.warmup_steps,
                  # 'total_steps':self.total_steps,
                  # 'hold':self.hold})
        config = {
                  'start_lr':self.start_lr,
                  'target_lr':self.target_lr,
                  'warmup_steps':self.warmup_steps,
                  'total_steps':self.total_steps,
                  'hold':self.hold}
        return config


def get_compiled_model(path_model, continue_training, nb_classes, window, nb_channels):

    # parallel my GPU training
    
    filter_num_begin = 128
    depth = 4
    stack_num_down = 2
    stack_num_up = 2
    patch_size = (2, 2)
    num_heads = [4, 8, 16, 32]
    
    filter_nums=[64,128,256,512]
    #filter_nums=[128,256,512,1024]
    filter_nums=[8,16,32,64]
    filter_nums=[32,64,128,256]


    window_size = [16, 16, 16, 16]
    #window_size = [2 , 4 , 8 , 16]

    window_size=[8,8,8,8]
    #window_size=[4,4,4,4]

    num_mlp = 512              # number of MLP nodes within the Transformer
    shift_window=True          # Apply window shifting, i.e., Swin-MSA

    n_class = 4

    IMG_HEIGHT = TMP_IMG_SIZE
    IMG_WIDTH  = TMP_IMG_SIZE
    IMG_BANDS=13
    IMG_BANDS_N=IMG_BANDS

    build_model = SwinUnet.swin_unet_2d_model(IMG_HEIGHT, IMG_WIDTH, IMG_BANDS_N, filter_num_begin, depth, stack_num_down, stack_num_up,
                                                          patch_size, num_heads, window_size, num_mlp, n_class, \
                                                                  filter_nums=filter_nums, shift_window=shift_window, name='swin_unet')
    nth_round=0
    nth_epoch=-1

    if continue_training:       

        tmpvar=path_model[0:len(path_model)-len('swin')]
        checkfile=sorted(glob.glob(tmpvar+'**checkpoint'))[-1]

        nth_epoch=int(checkfile.split('_')[-2])
        nth_round=int(checkfile.split('_')[-4])

        print('path_model=', path_model)
        print('nth_epoch=', nth_epoch, 'nth_round=', nth_round)

        print('check_point=', checkfile)

#        if os.path.exists(path_model):
#            build_model.load_weights(path_model)
#        else:
         
        build_model.load_weights(checkfile)
        print('load the existing weights')
        #build_model=tf.keras.models.load_model(checkfile,custom_objects={'f1_m': f1_m})

        
    return build_model, nth_epoch, nth_round


def train_vs_val(train_fraction, img_ids, lab_ids,test_fraction=None):
    
    index_list=np.arange(len(img_ids))

    for ishu in range(100):
        np.random.shuffle(index_list)
   
    img_roots=[x.split('/')[-1] for x in img_ids]

    img_roots=[x[0:len(x)-len('.tif')] for x in img_roots]

    img_roots=[x[len('img_'):len(x)] for x in img_roots]

    img_roots=np.array(img_roots)
    img_roots=img_roots[index_list]

    partition = {}

    partition['nsamples']=len(img_roots)

    if train_fraction==100:
        partition['train'] = img_roots
        partition['val']=['']
    else:
        if test_fraction is None:
            nlen=len(img_ids)
            n_training=int(nlen*train_fraction*0.01)
            
            training_index=random.sample(list(range(nlen)),n_training)

            val_index=set(list((range(nlen))))-set(training_index)
            val_index=list(val_index)

            print(type(training_index), len(training_index))
            print(type(val_index), len(val_index))

            partition['train']=img_roots[training_index]
            partition['val']=img_roots[val_index]
        else:
            nlen=len(img_ids)
            print('test_fraction=', test_fraction, 'train_fraction=', train_fraction)

            val_fraction=100-test_fraction-train_fraction
            print('val_fraction=', val_fraction)

            training_index=int(nlen*train_fraction*0.01)
            
            val_index=int( nlen* val_fraction*0.01)

            partition['train']=img_roots[0:training_index]
            partition['val']=img_roots[training_index:training_index+val_index]
            partition['test']=img_roots[training_index+val_index:nlen]

    return partition


if __name__=="__main__":

    sub_set_flag=0;;    # train the model using part of the training samples 
   
    datadir='dataset path'

    dir_output='model_path'

    training_data_dir=datadir+'hls_train/'
    val_data_dir=datadir+'hls_val/'

    path_train_img=training_data_dir+'img/'
    path_train_lab=training_data_dir+'label/'

    imgs=sorted(glob.glob(path_train_img+'*.tif'))
    labs=sorted(glob.glob(path_train_lab+'*.tif'))

    val_imgs=sorted(glob.glob(val_data_dir+'img/*tif'))
    val_labs=sorted(glob.glob(val_data_dir+'label/*tif'))

    val_imgs=np.array(val_imgs)
    val_labs=np.array(val_labs)

    imgs=np.array(imgs)
    labs=np.array(labs)

    if sub_set_flag ==1:
        nsamples=40000
        indexes=list(range(len(imgs)))
        random.seed(10)
        sub_indexes = random.sample(indexes, nsamples)
        imgs=imgs[sub_indexes]
        labs=labs[sub_indexes]
    else:
        nsamples=len(imgs)

    print('nsamples=', nsamples)

    train_fraction=100

    partition=train_vs_val(train_fraction, imgs, labs)
    val_partiion=train_vs_val(100, val_imgs, val_labs)

    partition['val']=val_partiion['train']

    partition['nsamples']= partition['nsamples'] +val_partiion['nsamples']
    
    print(partition)

    available_gpus = len(tf.config.list_physical_devices('GPU'))

    nb_gpus=4
    
    if nb_gpus> available_gpus:
        nb_gpus = available_gpus
    

    unit_batch_size=16

    unit_batch_size=4

    unit_batch_size=32

    if available_gpus!=0:
        batch_size = int(unit_batch_size * nb_gpus) # batch
    else:
        batch_size = unit_batch_size

    n_epochs = 350
    
    window = TMP_IMG_SIZE  # Get the patch size in pixels
    scale_factor = 10000.0  # Scale factor applied on satellite data (convert integer to float)
    patch_fmt_file='.tif'
    
    nb_channels=13
    #nb_channels=7
    nb_classes=4
    
    params = {'dim': (window, window),
              'batch_size': batch_size,
              'n_classes': nb_classes,
              'n_channels': nb_channels,
              'shuffle': True,
              'patch_fmt_file': patch_fmt_file,
              'scale_factor': scale_factor}
    
    print(partition.keys())
    print(partition['nsamples'])
    print(len(partition['train']))
    print(len(partition['val']))
    #print(len(partition['test']))

    
    tmp_dir_output=dir_output+'test_swin_nsample_'+str(nsamples)+'/'

    path_model=tmp_dir_output+'swin'

    if not os.path.exists(tmp_dir_output): os.makedirs(tmp_dir_output)

    print(tmp_dir_output)

    nth_epoch=0

    continue_training=False

    if continue_training == False:
        with open(tmp_dir_output+'partition_'+str(nsamples)+'.p','wb') as f:
                 pickle.dump(partition,f)
    else:
        with open(tmp_dir_output+'partition_'+str(nsamples)+'.p','rb') as f:
                 partition=pickle.load(f)

    training_generator = DataGenerator(partition['train'], \
            path_train_img, path_train_lab, **params)
    
    validation_generator = DataGenerator(partition['val'], val_data_dir+'img/',val_data_dir+'label/', **params)


    gpulist=[f'/gpu:{int(i)}' for i in range(0, nb_gpus)]
    
    print(gpulist)

    if len(gpulist)>1:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        
        strategy = tf.distribute.MirroredStrategy(gpulist)  # (['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3'])
        
        with strategy.scope():
            build_model,nth_epoch,nth_round = get_compiled_model(path_model, continue_training, nb_classes, window, nb_channels)
    else:  # CPU
        build_model, nth_epoch, nth_round = get_compiled_model(path_model, continue_training, nb_classes, window, nb_channels)

    def decay_schedule(epoch, lr):
        if (epoch % 5 == 0) and (epoch != 0):
            lr = lr * 0.99
            print('lr=', lr)
        return lr

    print('nth_epoch=', nth_epoch, 'nth_round=', nth_round)

    per_epoch=int(len(partition['train'])/batch_size)

    split_epoch=10

    hold_epoch=0

    momentum=0.9

    start_rate=0.0025

    start_rate=0.001

    total_steps = per_epoch*n_epochs

    warmup_steps = per_epoch*split_epoch

    hold_steps = per_epoch*hold_epoch

    print(total_steps, warmup_steps)
    
    learning_rate_base = start_rate

    global_steps=per_epoch*(nth_epoch+1)

    print('global_step=', global_steps)

    adam=Adam( beta_1=momentum, beta_2=0.999, epsilon=1e-07)

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
            total_steps=total_steps,
            global_step_init=global_steps,
            warmup_learning_rate=0,
            warmup_steps=warmup_steps,
            hold_base_rate_steps=0)


    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    build_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[ "accuracy",f1_m])

    #epoch=epoch+nth_epoch 
    nth_round=nth_round+1

    if continue_training:
        check_name=path_model+'_round_'+str(nth_round).zfill(3)+'_epoch_{epoch:03d}_checkpoint'
    else:
        check_name=path_model+'_round_'+str(0).zfill(3)+'_epoch_{epoch:03d}_checkpoint'

    print('check_name=', check_name)


    #checkpoint = ModelCheckpoint(path_model+'_epoch_{epoch:03d}_checkpoint', monitor='acc', verbose=1, mode='auto', save_freq='epoch')
    checkpoint = ModelCheckpoint(check_name, monitor='acc', verbose=1, mode='auto', save_freq='epoch',save_weights_only=False,)

    print('nth_epoch=', nth_epoch)

    if train_fraction!=100:
        history_progress = build_model.fit_generator(generator=training_generator,
                                                     validation_data=validation_generator,
                                                     epochs=n_epochs,
                                                     max_queue_size=32,
                                                     workers=32,
                                                     callbacks=[warm_up_lr, checkpoint,csv_logger],
                                                     initial_epoch =nth_epoch
                                                     )
    else:
        history_progress = build_model.fit_generator(generator=training_generator,
                                                    validation_data=validation_generator,
                                                     epochs=n_epochs,
                                                     max_queue_size=32,
                                                     workers=32,
                                                     callbacks=[warm_up_lr, checkpoint,csv_logger],
                                                     initial_epoch=nth_epoch
                                                     )
    #
    ## Save model
    build_model.save(path_model, overwrite=True)
    #
    ## Save history
    #save_history(file_history, history_progress, continue_training, val_valid, path_model)
    #
    ## RUN TESTING DATASET
    #if test_valid:
    #    score = build_model.evaluate_generator(generator=test_generator)
    #    print("Test score:", score[0])
    #    print('Test accuracy:', score[1])
    #

    with open('train_swin_aws_lr_cloudsen12_hls.p','wb') as f:
        pickle.dump([warm_up_lr.learning_rates,total_steps,n_epochs],f)

    plt.plot(warm_up_lr.learning_rates)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('lr', fontsize=20)
    plt.axis([0, total_steps, 0, learning_rate_base*1.1])
    plt.xticks(np.arange(0, total_steps, 50))
    plt.grid()
    plt.title('Cosine decay with warmup', fontsize=20)
    plt.show()




