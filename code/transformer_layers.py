
# from __future__ import absolute_import

import tensorflow as tf
from tensorflow.image import extract_patches
from tensorflow.keras.layers import Conv2D, Layer, Dense
from tensorflow.keras.layers import Conv2DTranspose as Conv2DTranspose_old
from tensorflow.keras.layers import Embedding as Embedding_old

class patch_extract(Layer):
    '''
    Extract patches from the input feature map.    
    patches = patch_extract(patch_size)(feature_map)    
    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.
    
    Input
    ----------
        feature_map: a four-dimensional tensor of (num_sample, width, height, channel)
        patch_size: size of split patches (width=height)
        
    Output
    ----------
        patches: a two-dimensional tensor of (num_sample*num_patch, patch_size*patch_size)
                 where `num_patch = (width // patch_size) * (height // patch_size)`
                 
    For further information see: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches        
    '''    
    def __init__(self, patch_size):
        super(patch_extract, self).__init__()
        super().__init__()
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]
        
    def call(self, images):
        
        batch_size = tf.shape(images)[0]

        #print('batch_size=', tf.shape(images))
        
        patches = extract_patches(images=images,
                                  sizes=(1, self.patch_size_x, self.patch_size_y, 1),
                                  strides=(1, self.patch_size_x, self.patch_size_y, 1),
                                  rates=(1, 1, 1, 1), padding='VALID',)
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)     
        #
        #print('transformer_layers: patches.shape=',patches.shape)

        
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(patches, (batch_size, patch_num*patch_num, patch_dim))

        #print('transformer_layers: patches.shape=',patches.shape)

        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)
        
        return patches
    def get_config(self):
        config=super().get_config().copy()
        config.update({
                      'patch_size_x':self.patch_size_x,
                      'patch_size_y':self.patch_size_y,})
        return config

class patch_embedding(Layer):
    '''    
    Embed patches to tokens.    
    patches_embed = patch_embedding(num_patch, embed_dim)(pathes)    
    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.
    
    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions. 
        
    Output
    ----------
        embed: Embedded patches.
    
    For further information see: https://keras.io/api/layers/core_layers/embedding/    
    '''    
    def __init__(self, num_patch, embed_dim, is_position=True):
        super(patch_embedding, self).__init__()
        super().__init__()
        self.num_patch = num_patch
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding_old(input_dim=num_patch, output_dim=embed_dim)
        self.is_position = is_position

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        if self.is_position: 
            embed = self.proj(patch) + self.pos_embed(pos)
        else:
            embed = self.proj(patch)
        
        return embed
    
    def get_config(self):
        config=super().get_config().copy()
        config.update({
                      'num_patch':self.num_patch,
                      'proj':self.proj,
                      'is_position':self.is_position,
                      'pos_embed':self.pos_embed})
        return config
        
class patch_merging(tf.keras.layers.Layer):
    '''
    Downsample embedded patches; it halfs the number of patches
    and double the embedded dimensions (c.f. pooling layers).
    
    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions. 
        
    Output
    ----------
        x: downsampled patches.    
    '''
    def __init__(self, num_patch, embed_dim, name=''):
        super().__init__()
        
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        
        # A linear transform that doubles the channels 
        # self.linear_trans = Dense(2*embed_dim, use_bias=False, name='{}linear_trans'.format(name))
        self.linear_trans = Dense(embed_dim, use_bias=False, name='{}linear_trans'.format(name)) # Hank changed on Apr 25, 2023 

    def call(self, x):
        
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        
        assert (L == H * W), 'input feature has wrong size'
        assert (H % 2 == 0 and W % 2 == 0), '{}-by-{} patches received, they are not even.'.format(H, W)
        
        # Convert the patch sequence to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))
        
        # Downsample
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        
        # Convert to the patch squence
        x = tf.reshape(x, shape=(-1, (H//2)*(W//2), 4*C))
       
        # Linear transform
        x = self.linear_trans(x)
        return x
    def get_config(self):
        config=super().get_config().copy()
        config.update({
                      'num_patch':self.num_patch,
                      'embed_dim':self.embed_dim,
                      'linear_trans':self.linear_trans})
        return config
        
class patch_expanding(tf.keras.layers.Layer):

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, upsample_method=0, name=''):
        super().__init__()
        
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        self.upsample_method = upsample_method
        
        # Linear transformations that doubles the channels 
        self.linear_trans1 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}linear_trans1'.format(name))
        # 
        # self.linear_trans2 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}linear_trans2'.format(name))
        self.linear_trans2 = Conv2DTranspose_old(embed_dim//2, kernel_size=[upsample_rate, upsample_rate], strides=[upsample_rate, upsample_rate], padding='same', use_bias=False, name='{}linear_trans2'.format(name))
        self.prefix = name
        
    def call(self, x):
        
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        print('H=', H, 'W=', W)
        print('B=', B, 'L=', L, 'C=', C)
        
        assert (L == H * W), 'input feature has wrong size'

        
        x = tf.reshape(x, (-1, H, W, C))

        print('x1=', x)


        if self.upsample_method==0:
            x = self.linear_trans1(x) # TensorShape([2, 32, 32, 1024])

            print('x2=', x)
            
            # rearange depth to number of patches
            x = tf.nn.depth_to_space(x, self.upsample_rate, data_format='NHWC', name='{}d_to_space'.format(self.prefix)) # TensorShape([2, 64, 64, 256])

            print('x3=', x)
            
        else:
            print("use Conv2DTranspose_old")
            x = self.linear_trans2(x)

            print('x=', x)
        
        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1, L*self.upsample_rate*self.upsample_rate, C//2))
        
        return x
    
    def get_config(self):
        config=super().get_config().copy()
        config.update({
                      'num_patch':self.num_patch,
                      'embed_dim':self.embed_dim,                       
                      'upsample_rate':self.upsample_rate,
                      'upsample_method':self.upsample_method,
                      'return_vector':self.return_vector})
        return config
           
