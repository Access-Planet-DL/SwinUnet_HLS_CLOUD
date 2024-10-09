"""
Created on Mon May  9 09:27:06 2022
@author: Dong.Luo
This code is the application of SwinUnet
Note:
    (1) original Swin transformer: depth is a list, but SwinUnet depth is an integer
    (2) original Swin transformer: window size is an integer, but SwinUnet is a list
    (3) using layer normalization not batch normalization
    (4) dense layer (similar with nn.linear in pytorch)
"""
import numpy as np
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate

import sys
# sys.path.append('../')
#import swin_layers;import transformer_layers

import swin_layers
import transformer_layers


# stack_num = 2; shift_window=True; window_size = 8; num_heads=4; num_patch=(num_patch_x, num_patch_y)
def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, name='', 
        mlp_drop_rate=0, attn_drop_rate=0, proj_drop_rate=0, drop_path_rate=0):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    Args:
        stack_num: number of blocks
    '''
    # Turn-off dropouts
    # mlp_drop_rate  = 0 # Droupout after each MLP layer
    # attn_drop_rate = 0 # Dropout after Swin-Attention
    # proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    # drop_path_rate = 0 # Drop-path within skip-connections
    # drop_path_rate = 0.1 # Drop-path within skip-connections
    # mlp_drop_rate = proj_drop_rate = 0.1
    print ("drop_path_rate={:4.2f}, mlp_drop_rate={:4.2f}, attn_drop_rate={:4.2f}, proj_drop_rate={:4.2f},".format(drop_path_rate,mlp_drop_rate,attn_drop_rate,proj_drop_rate) )
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0

    for i in range(stack_num):

        #print('i=', i)
    
        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size
        
        X = swin_layers.SwinTransformerBlock(dim=embed_dim, 
                                             num_patch=num_patch, 
                                             num_heads=num_heads, 
                                             window_size=window_size, 
                                             shift_size=shift_size_temp, 
                                             num_mlp=num_mlp, 
                                             qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate, 
                                             attn_drop=attn_drop_rate, 
                                             proj_drop=proj_drop_rate, 
                                             drop_path_prob=drop_path_rate, 
                                             name='name{}'.format(i))(X)
    return X


# IN = train_x[:2,:,:,:]; name='swin_unet'
def swin_unet_2d_model(IMG_HEIGHT, IMG_WIDTH, IMG_BANDS_N, 
                       filter_num_begin, depth, stack_num_down, stack_num_up, 
                       patch_size, num_heads, window_size, num_mlp, n_class, filter_nums=[64,128,256,512],
                       shift_window=True, name='swin_unet'):
    '''
    The base of Swin-Unet. The general structure:    
    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head    
    '''
    # Compute number be patches to be embeded
    # input_size = input_tensor.shape.as_list()[1:]
    input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_BANDS_N)

    #print('SwinUnet: input_size =', input_size)
    #print(input_size)
    #print('n_class=', n_class)

    IN = Input(input_size)
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]

    print('num_patch_x = ', num_patch_x, 'num_patch_y=', num_patch_y, 'patch_size=', patch_size)
    print('SwinUnet: window_size =', window_size)

    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    embed_dim = filter_nums[0]

    #print('embed_dim=', embed_dim)
    
    depth_ = depth
    
    X_skip = []
    mlp_ratio = 2
    mlp_ratio = 4
    
    # Patch extraction
    X = transformer_layers.patch_extract(patch_size)(IN)

    # Embed patches to token
    X = transformer_layers.patch_embedding(num_patch_x*num_patch_y, embed_dim, is_position=False)(X)

    

    DROP_RATE = 0.1 
    DROP_RATE = 0.3
    #print ('embed_dim = {:d} and depth_decode i = {:d}'.format(embed_dim, 0) )
    # The first Swin Transformer stack
    #
    # stack_num_down=2
    X = swin_transformer_stack(X, stack_num=stack_num_down, embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], window_size=window_size[0], num_mlp=embed_dim*mlp_ratio, 
                               shift_window=shift_window,  name='{}_swin_down0'.format(name)
                               , mlp_drop_rate=DROP_RATE, proj_drop_rate=DROP_RATE, attn_drop_rate=DROP_RATE,drop_path_rate=0)

    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_-1):

        # update token shape info by downsample the patch_size and double dimension
        embed_dim = embed_dim*2
        embed_dim = filter_nums[i+1]
        
        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)

        
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        print ('embed_dim = {:d} and depth_encode i = {:d}'.format(embed_dim, i) )         
        # Swin Transformer stacks
        X = swin_transformer_stack(X, stack_num=stack_num_down, embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], window_size=window_size[i+1], num_mlp=embed_dim*mlp_ratio, 
                                   shift_window=shift_window, name='{}_swin_down{}'.format(name, i+1)
                                   , mlp_drop_rate=DROP_RATE, proj_drop_rate=DROP_RATE, attn_drop_rate=DROP_RATE,drop_path_rate=0
                                   )

        
        
        # Store tensors for concat
        X_skip.append(X)

    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)

        
    for i in range(depth_decode):

        # Patch expanding
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                               embed_dim=embed_dim, 
                                               upsample_rate=2, 
                                               return_vector=True)(X)        
        
        # update token shape info
        embed_dim = embed_dim//2
        embed_dim = filter_nums[depth_decode-i-1]
        
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        print ('embed_dim = {:d} and depth_decode i = {:d}'.format(embed_dim, i) ) 
        # Concatenation and linear projection
        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, stack_num=stack_num_up, embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y),  num_heads=num_heads[i], 
                                   window_size=window_size[i], num_mlp=embed_dim*mlp_ratio, 
                                   shift_window=shift_window,  name='{}_swin_up{}'.format(name, i)
                                   , mlp_drop_rate=DROP_RATE, proj_drop_rate=DROP_RATE, attn_drop_rate=DROP_RATE,drop_path_rate=0
                                   )

        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    #
    #
    
    X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                           embed_dim=embed_dim*mlp_ratio, 
                                           upsample_rate=patch_size[0], 
                                           return_vector=False)(X)
    n_labels = n_class
    
    
    OUT = Conv2D(n_labels, kernel_size=1, use_bias=False, activation='sigmoid')(X)
    #OUT = Conv2D(n_labels, kernel_size=1, use_bias=False)(X)

    # Model configuration
    model = Model(inputs=[IN,], outputs=[OUT,])      
    return model



# IN = train_x[:2,:,:,:]; name='swin_unet'
def swin_unet_2d_model_up(IMG_HEIGHT, IMG_WIDTH, IMG_BANDS_N, 
                       filter_num_begin, depth, stack_num_down, stack_num_up, 
                       patch_size, num_heads, window_size, num_mlp, n_class, filter_nums=[64,128,256,512],
                       shift_window=True, name='swin_unet'):
    '''
    The base of Swin-Unet. The general structure:    
    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head    
    '''
    # Compute number be patches to be embeded
    # input_size = input_tensor.shape.as_list()[1:]
    input_size = (IMG_HEIGHT, IMG_WIDTH, IMG_BANDS_N)
    IN = Input(input_size)
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    
    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    embed_dim = filter_nums[0]
    
    depth_ = depth
    
    X_skip = []
    mlp_ratio = 2
    mlp_ratio = 4
    
    # Patch extraction
    X = transformer_layers.patch_extract(patch_size)(IN)

    # Embed patches to tokens
    X = transformer_layers.patch_embedding(num_patch_x*num_patch_y, embed_dim, is_position=False)(X)
    
    DROP_RATE = 0.1 
    print ('embed_dim = {:d} and depth_decode i = {:d}'.format(embed_dim, 0) ) 
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, stack_num=stack_num_down, embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], window_size=window_size[0], num_mlp=embed_dim*mlp_ratio, 
                               shift_window=shift_window,  name='{}_swin_down0'.format(name)
                               , mlp_drop_rate=DROP_RATE, proj_drop_rate=DROP_RATE, attn_drop_rate=DROP_RATE,drop_path_rate=0
                               )
    
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        # update token shape info by downsample the patch_size and double dimension
        embed_dim = embed_dim*2
        embed_dim = filter_nums[i+1]
        
        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
        
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        print ('embed_dim = {:d} and depth_encode i = {:d}'.format(embed_dim, i) )         
        # Swin Transformer stacks
        X = swin_transformer_stack(X, stack_num=stack_num_down, embed_dim=embed_dim, num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], window_size=window_size[i+1], num_mlp=embed_dim*mlp_ratio, 
                                   shift_window=shift_window, name='{}_swin_down{}'.format(name, i+1)
                                   , mlp_drop_rate=DROP_RATE, proj_drop_rate=DROP_RATE, attn_drop_rate=DROP_RATE,drop_path_rate=0
                                   )
        
        # Store tensors for concat
        X_skip.append(X)
    
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        # Patch expanding
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                               embed_dim=embed_dim, 
                                               upsample_rate=2, upsample_method=1, 
                                               return_vector=True)(X)        
        
        # update token shape info
        embed_dim = embed_dim//2
        embed_dim = filter_nums[depth_decode-i-1]
        print (embed_dim)
        print (X_decode[i].shape )
        
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        print ('embed_dim = {:d} and depth_decode i = {:d}'.format(embed_dim, i) ) 
        # Concatenation and linear projection
        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(name, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(name, i))(X)
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, stack_num=stack_num_up, embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y),  num_heads=num_heads[i], 
                                   window_size=window_size[i], num_mlp=embed_dim*mlp_ratio, 
                                   shift_window=shift_window,  name='{}_swin_up{}'.format(name, i)
                                   , mlp_drop_rate=DROP_RATE, proj_drop_rate=DROP_RATE, attn_drop_rate=DROP_RATE,drop_path_rate=0
                                   )
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    
    X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                           embed_dim=embed_dim*mlp_ratio, 
                                           upsample_rate=patch_size[0], upsample_method=1, 
                                           return_vector=False)(X)
        
    n_labels = n_class
    # OUT = Conv2D(n_labels, kernel_size=1, use_bias=False, activation='softmax')(X)
    OUT = Conv2D(n_labels, kernel_size=1, use_bias=False)(X)

    # Model configuration
    model = Model(inputs=[IN,], outputs=[OUT,])      
    return model

    
# ## hyperparameters to help understand the model
# filter_num_begin = 128     # number of channels in the first downsampling , X.shape)block; it is also the number of embedded dimensions
# depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
# stack_num_down = 2         # number of Swin Transformers per downsampling level
# stack_num_up = 2           # number of Swin Transformers per upsampling level
# patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
# num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
# window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
# num_mlp = 512              # number of MLP nodes within the Transformer
# shift_window=True          # Apply window shifting, i.e., Swin-MSA

##******************************************************************************************************
## model configuration (example)
# Input section
# input_size = (256, 256, 7)   # shape (256, 256, bands)
# IN = Input(input_size)
# n_class = 10
# # Base architecture
# model = swin_unet_2d_model(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
#                            patch_size, num_heads, window_size, num_mlp, 
#                            n_class,
#                            shift_window=shift_window, name='swin_unet')

# # Optimization
# # <---- !!! gradient clipping is important
# opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt)


