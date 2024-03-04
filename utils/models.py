from keras.layers import Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Multiply, UpSampling2D
import tensorflow as tf
from utils.instance_norm import _instance_norm_block


def build_model_binary(base_model, dropout_rate, n_class, sigmoid=False, mode=None, p=None, eps=1e-5, return_feats=True, fwcta=False): 

    #1/8 resolution output
    out_1_8 = base_model.get_layer('expanded_conv_6/expand/act_1').output
    #1/16 resolution output
    out_1_16 = base_model.get_layer('expanded_conv_12/expand/act_1').output
    
    ls = ['instance_normalization',
          'instance_normalization_1',
          'instance_normalization_2']
    
    if mode == 'ISW':
        features = []
        for l in ls:
            features.append(base_model.get_layer(l).output)
    else:
        features = [out_1_16]
    

    if mode == 'KD' and fwcta:
        out_1_8 = _instance_norm_block(out_1_8, mode='KD_WCTA')
        out_1_16 = _instance_norm_block(out_1_16, mode='KD_WCTA')

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

    # head
    m1 = Multiply()([x1, x2])
    m1 = UpSampling2D(size=(2, 2),data_format='channels_last',interpolation="bilinear")(m1)
    m1 = Conv2D(n_class, (1, 1))(m1)
    m2 = Add()([m1, x3])
    predictions = UpSampling2D(size=(8, 8),data_format='channels_last',interpolation="bilinear")(m2)
    predictions = Activation('sigmoid')(predictions) if sigmoid else predictions
    
    # final model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=[predictions, *features] if return_feats else predictions)
    return model



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
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model
