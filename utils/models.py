from tensorflow.keras.models import Model,load_model

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Activation, Input, Add, AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape,Dropout, Multiply, Flatten,UpSampling2D
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.metrics import MeanIoU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

import tensorflow.keras.backend as K
import tensorflow as tf
#from tensorflow.keras.utils import control_flow_util


def build_model_multi(base_model, dropout_rate, n_class): 
    
    #1/8 resolution output
    out_1_8= base_model.get_layer('expanded_conv_6/expand/act_1').output
    #1/16 resolution output
    out_1_16= base_model.get_layer('expanded_conv_12/expand/act_1').output
    
    # branch1
    x1 = Conv2D(128, (1, 1))(out_1_16)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    # branch2
    s = x1.shape
    x2 = AveragePooling2D(pool_size=(12, 12), strides=(4, 5),data_format='channels_last')(out_1_16)
    x2 = Conv2D(128, (1, 1))(x2)    
    x2 = Activation('sigmoid')(x2)
    x2 = UpSampling2D(size=(int(s[1]), int(s[2])),data_format='channels_last',interpolation="bilinear")(x2)
    
    # branch3
    x3 = Conv2D(n_class, (1, 1))(out_1_8)
    
    #head
    m1 = Multiply()([x1, x2])
    m1 = UpSampling2D(size=(2, 2),data_format='channels_last',interpolation="bilinear")(m1)
    m1 = Conv2D(n_class, (1, 1))(m1)
    m2 = Add()([m1, x3])
    predictions = UpSampling2D(size=(8, 8),data_format='channels_last',interpolation="bilinear")(m2)
    
    # final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def build_model_binary(base_model, dropout_rate, n_class, sigmoid=True, inst_norm=None, p=None, eps=1e-5, return_feats=True): 


    #1/8 resolution output
    out_1_8 = base_model.get_layer('expanded_conv_6/expand/act_1').output
    #1/16 resolution output
    out_1_16 = base_model.get_layer('expanded_conv_12/expand/act_1').output
    
    ls = ['instance_normalization',
          'instance_normalization_1',
          'instance_normalization_2']
    
    if inst_norm == 'ISW':
        features = []
        for l in ls:
            features.append(base_model.get_layer(l).output)
    else:
        features = [out_1_16]
    
    # branch1
    x1 = Conv2D(128, (1, 1))(out_1_16)
#     if inst_norm == 'IN':
#         x1 = _instance_norm_block(x1, inst_norm, p, eps=eps)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    # branch2
    s = x1.shape
    x2 = AveragePooling2D(pool_size=(12, 12), strides=(4, 5),data_format='channels_last')(out_1_16)
    x2 = Conv2D(128, (1, 1))(x2)
#     if inst_norm == 'IN':
#         x2 = _instance_norm_block(x2, inst_norm, p, eps=eps)
    x2 = Activation('sigmoid')(x2)    
    x2 = UpSampling2D(size=(int(s[1]), int(s[2])),data_format='channels_last',interpolation="bilinear")(x2)
    
    # branch3
    x3 = Conv2D(n_class, (1, 1))(out_1_8)

    # head
    m1 = Multiply()([x1, x2])
    m1 = UpSampling2D(size=(2, 2),data_format='channels_last',interpolation="bilinear")(m1)
    m1 = Conv2D(n_class, (1, 1))(m1)
    m2 = Add()([m1, x3])
    predictions = UpSampling2D(size=(8, 8),data_format='channels_last',interpolation="bilinear")(m2)
    predictions = Activation('sigmoid')(predictions) if sigmoid else predictions
    
    
    
    # final model
    model = Model(inputs=base_model.input, outputs=[predictions, *features] if return_feats else predictions)
    return model


class ModelBinary(tf.keras.Model):
    def __init__(self, base_model, dropout_rate, n_class, sigmoid=True, inst_norm=None, p=None, eps=1e-5):
        super(ModelBinary, self).__init__()
        
        self.inp = base_model.input
        self.backbone = base_model
        self.in_dec_1 = base_model.get_layer('activation_15').output
        self.in_dec_2 = base_model.get_layer('activation_29').output

        self.conv_1 = Conv2D(128, (1, 1))
        self.i_norm_1 = INormBlock(dg, padain_p)
        self.bn_1 = BatchNormalization()
        self.act_1 = Activation('relu')
        
        self.avg_1 = AveragePooling2D(pool_size=(12, 12), strides=(4, 5), data_format='channels_last')
        self.conv_2 = Conv2D(128, (1, 1))
        self.i_norm_2 = INormBlock(dg, padain_p)
        self.act_2 = Activation('sigmoid')
        # self.ups_1 = UpSampling2D(size=(int(s[1]), int(s[2])), data_format='channels_last', interpolation="bilinear")
    
        self.conv_3 = Conv2D(n_class, (1, 1))
        self.mult_1 = Multiply()
        self.ups_2 = UpSampling2D(size=(2, 2),data_format='channels_last',interpolation="bilinear")
        self.conv_4 = Conv2D(n_class, (1, 1))
        self.add_1 = Add()
        self.ups_3 = UpSampling2D(size=(8, 8),data_format='channels_last',interpolation="bilinear")
        self.act_3 = Activation('sigmoid') if sigmoid else tf.keras.layers.Identity()
        
    def call(self, x, norm=False):
        x = backbone(x)
        
        
        
def _instance_norm_block(x, mode=None, p=0.01, eps=1e-5):
    if mode == 'IN':
        return InstanceNormalization()(x)
    elif mode == 'PADAIN':
        return PAdaIN(p=p, eps=eps)(x)
    else:
        return x
    
    
def INormBlock(mode=None, p=0.01, eps=1e-5):
    if mode == 'IN':
        return InstanceNormalization()
    elif mode == 'PADAIN':
        return PAdaIN(p=p, eps=eps)
    else:
        return tf.keras.layers.Identity()
    

class PAdaIN(tf.keras.layers.Layer):
    
    def __init__(self, p=0.01, eps=1e-5):
        super(PAdaIN, self).__init__()
        self.p = p
        self.eps = eps
        
    def call(self, inputs, training=None):
        
        if training is None:
            training = K.learning_phase()
        
        permute = tf.random.uniform([], minval=0, maxval=1) < self.p
        
        if permute:
            perm_indices = tf.random.shuffle(tf.range(0, inputs.shape[0]))
        else:
            return tf.identity(inputs)
        
        shape = inputs.shape
        N, H, W, C = shape

        out = self.ada_in(inputs, tf.gather(inputs, perm_indices))

        #output = control_flow_util.smart_cond(training, out, lambda: tf.identity(inputs))
        
        return out if training else tf.identity(inputs)

    def get_mean_std(self, x):
        epsilon = self.eps
        axes = [1, 2]
        # Compute the mean and standard deviation of a tensor.
        mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
        standard_deviation = tf.sqrt(variance + epsilon)
        return mean, standard_deviation

    def ada_in(self, style, content):
        """
        Computes the AdaIn feature map.
        Args:
            style: The style feature map.
            content: The content feature map.
        Returns:
            The AdaIN feature map.
        """
        content_mean, content_std = self.get_mean_std(content)
        style_mean, style_std = self.get_mean_std(style)
        t = style_std * (content - content_mean) / content_std + style_mean
        return t
    
    def get_config(self):
        cfg = super().get_config()
        return cfg 