
# from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization
from tensorflow.keras.activations import softmax

from util_layers import drop_path

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    # Get the static shape of the input tensor
    # (Sample, Height, Width, Channel)
    _, H, W, C = x.get_shape().as_list()
    
    # Subset tensors to patches
    patch_num_H = H//window_size
    patch_num_W = W//window_size

    #print('window_partition: x.shape=',x.shape)
    #print('window_partition: window_size=', window_size)
    #print('window_partition: x.shape=',x.get_shape().as_list())
    #print('H=', H, 'W=',W, 'window_size=', window_size, 'patch_num_H=', patch_num_H, 'patch_num_W=', patch_num_W,'C=', C)
    
    x = tf.reshape(x, shape=(-1, patch_num_H, window_size, patch_num_W, window_size, C))

    #print('swin_layers: after reshape: x.shape=',x.shape)

    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    
    # Reshape patches to a patch sequence
    windows = tf.reshape(x, shape=(-1, window_size, window_size, C))

    #print('swin_layers: after reshape: x.shape=',x.shape)

    #print('swin_layers: windows.shape=', windows.shape)

    return windows

def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    # Reshape a patch sequence to aligned patched 
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(windows, shape=(-1, patch_num_H, patch_num_W, window_size, window_size, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    
    # Merge patches to spatial frames
    x = tf.reshape(x, shape=(-1, H, W, C))
    
    return x


class Mlp(tf.keras.layers.Layer):
    def __init__(self, filter_num, drop=0., name=''):
        
        super().__init__()
        
        # MLP layers
        self.fc1 = Dense(filter_num[0], name='{}_mlp_0'.format(name))
        self.fc2 = Dense(filter_num[1], name='{}_mlp_1'.format(name))
        
        # Dropout layer
        self.drop = Dropout(drop)
        
        # GELU activation
        self.activation = tf.keras.activations.gelu
        #self.activation = tf.nn.gelu

        
    def call(self, x):
        
        # MLP --> GELU --> Drop --> MLP --> Drop
        x = self.fc1(x)
        self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class WindowAttention(tf.keras.layers.Layer):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0., name=''):
        super().__init__()
        
        self.dim = dim # number of input dimensions
        self.window_size = window_size # size of the attention window
        self.num_heads = num_heads # number of self-attention heads
        
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # query scaling factor
        
        self.prefix = name
        
        # Layers
        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name='{}_attn_qkv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)
    
    def build(self, input_shape):
        # zero initialization
        num_window_elements = (2*self.window_size[0] - 1) * (2*self.window_size[1] - 1)
        # 对于一个8(window_size)*8(window_size)的window来说，相对距离有(2*window-1)*(2*window-1)种可能性
        # 这样做每个都有一个相对position
        self.relative_position_bias_table = self.add_weight('{}_attn_pos'.format(self.prefix),
                                                            shape=(num_window_elements, self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)
        
        # Indices of relative positions
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 
        relative_position_index = relative_coords.sum(-1) # 对于每个window来说 相对距离有64(window_size^2)*64(window_size^2)个 
        
        # convert to the tf variable
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False, name='{}_attn_pos_ind'.format(self.prefix))
        
        self.built = True
        
        # added by Hank on May 26 2023 for Swin V2 
        self.tau = self.add_weight('{}_tau'.format(self.prefix), shape=(1,self.num_heads,1,1), initializer=tf.keras.initializers.Constant(0.02), trainable=True)
    
    def call(self, x, mask=None):
        # x shape = ([2048, 64, 32]) with 2048 = batch size*window number, 64=window size^2 (N), 32=dim (C)
        # Get input tensor static shape
        _, N, C = x.get_shape().as_list()
        head_dim = C//self.num_heads
        
        x_qkv = self.qkv(x) 
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4)) #([3, 2048, 4, 64, 8]) 3 = qkv, 2048=batch size*window number, 4 = head, 64=window number^2, 8 = dim(C)//head
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        
        # Query rescaling
        q = q * self.scale #([2048, 4, 64, 8]： 2048=batch size*window number, 4 = head, 64=window number^2, 8 = dim(C)//head
        
        # multi-headed self-attention
        # k = tf.transpose(k, perm=(0, 1, 3, 2)) # TensorShape([2048, 4, 8, 64]) see above 
        # attn = (q @ k) # TensorShape([128, 4, 64, 64]) 128 is batch size*window number, 4 is head, 64 is element number within each window (window size^2), and 64 is element number within each window (window size^2)
        
        # changed by Hank on May 26 2023 for Swin V2 
        # attn = tf.math.cos(q,k)/self.tau # 
        # attn = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)(q[:,:,:,:,tf.newaxis],k[:,:,:,:,tf.newaxis])
        
        # normalize_a = tf.math.l2_normalize(q,-1)        
        # normalize_b = tf.math.l2_normalize(k,-1)
        # similarity = tf.matmul(normalize_a, normalize_b, transpose_b=True)
        # attn = similarity/self.tau # 
        attn = tf.matmul(tf.math.l2_normalize(q,-1), tf.math.l2_normalize(k,-1), transpose_b=True)/self.tau # 
        
        # Shift window
        num_window_elements = self.window_size[0] * self.window_size[1]

        #print('self.window_size[0]=',self.window_size[0],'self.window_size[1]=', self.window_size[1])

        #print('swin_layers: num_window_elements=', num_window_elements)

        #input()


        relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
        relative_position_bias = tf.reshape(relative_position_bias, shape=(num_window_elements, num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)
        
        # Hank revised on Apr 4 2023, do not use mask (wrong !) 
        # attn = softmax(attn, axis=-1) # shape no change 
        
        # original
        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1) # shape no change 
        
        # Dropout after attention
        attn = self.attn_drop(attn)
        
        # Merge qkv vectors
        x_qkv = (attn @ v) # 
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))
        
        # Linear projection
        x_qkv = self.proj(x_qkv)
        
        # Dropout after projection
        x_qkv = self.proj_drop(x_qkv)
        
        return x_qkv

class SwinTransformerBlock(tf.keras.layers.Layer):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_patch (tuple[int]): number of embedded patches; a tuple of  (heigh, width)
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        num_mlp: number of MLP nodes.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        mlp_drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0        
    """
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024,
                 qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0, proj_drop=0, drop_path_prob=0, name=''):
        super().__init__()
        
        self.dim = dim # number of input dimensions
        self.num_patch = num_patch # number of embedded patches; a tuple of  (heigh, width)
        self.num_heads = num_heads # number of attention heads
        self.window_size = window_size # size of window
        self.shift_size = shift_size # size of window shift
        self.num_mlp = num_mlp # number of MLP nodes
        self.prefix = name

        
        # Layers
        self.norm1 = LayerNormalization(epsilon=1e-5, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = LayerNormalization(epsilon=1e-5, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], drop=mlp_drop, name=self.prefix)
        
        # Assertions
        assert 0 <= self.shift_size, 'shift_size >= 0 is required'
        assert self.shift_size < self.window_size, 'shift_size < window_size is required'
        
        # <---!!!
        # Handling too-small patch numbers
        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)
            
    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.num_patch
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            
            # attention mask
            mask_array = np.zeros((1, H, W, 1))

            #print(mask_array.shape)
            
            ## initialization
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            
            
            #print('swin_layers: mask_array.shape=', mask_array.shape)
            #print('swin_layers: self.window_size=',self.window_size)

            #input()

            mask_array = tf.convert_to_tensor(mask_array)
            
            # mask array to windows
            # 这里其实并没有错，这里并不是直观意义上的只对mask_array(64(window number)*64(window size))不等于0的attention 
            # 而是对attn_mask(64(window number)*64(window size)*64(window size))等于0的attention
            # 见MARD1NO commented on Apr 20, 2021 in https://github.com/microsoft/Swin-Transformer/issues/38
            

            mask_windows = window_partition(mask_array, self.window_size)

            ##print('zzzzzz')


            mask_windows = tf.reshape(mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False, name='{}_attn_mask'.format(self.prefix)) 
            # ! noted on April 25, 2023 - this is a super heavy mask；
            # ! (window_num*window_num)*(window_size*window_size)*(window_size*window_size)
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()

        #print('x.shape=', x.get_shape(), 'H=', H, 'W=',W)
        
        # Checking num_path and tensor sizes
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'
        
        # Skip connection I (start)
        x_skip = x
        
        # Layer normalization
        # x = self.norm1(x)
        
        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x


        #print('swin_layers: shifted_x.shape=', x.get_shape())

        # Window partition 
        x_windows = window_partition(shifted_x, self.window_size) # the window number has been assigned to batch 
        x_windows = tf.reshape(x_windows, shape=(-1, self.window_size * self.window_size, C))

        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows, shape=(-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
            
        # Convert back to the patch sequence
        x = tf.reshape(x, shape=(-1, H*W, C))

        # Layer normalization
        # changed by Hank on May 26 2023 for Swin V2 
        x = self.norm1(x)
        
        # Drop-path
        ## if drop_path_prob = 0, it will not drop
        x = self.drop_path(x)
        
        # Skip connection I (end)
        x = x_skip +  x
        
        # Skip connection II (start)
        x_skip = x
        
        # x = self.norm2(x)
        x = self.mlp(x)
        x = self.norm2(x) # changed by Hank on May 26 2023 for Swin V2 
        x = self.drop_path(x)
        
        # Skip connection II (end)
        x = x_skip + x
        return x
    
    def get_config(self):
        config=super().get_config().copy()
        config.update({
                  'dim':self.dim,
                  'num_patch':self.num_patch,
                  'num_heads':self.num_heads,
                  'window_size':self.window_size,
                  'shift_size':self.shift_size,
                  'num_mlp':self.num_mlp})
        return config
           
