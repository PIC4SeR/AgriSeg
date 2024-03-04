from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils

def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

import os
import warnings

from utils.instance_norm import _instance_norm_block

backend = None
layers = None
models = None
keras_utils = None

BASE_WEIGHT_PATH = ('https://github.com/DrSlink/mobilenet_v3_keras/'
                    'releases/download/v1.0/')
WEIGHTS_HASHES = {
    'large_224_0.75_float': (
        '765b44a33ad4005b3ac83185abf1d0eb',
        'c256439950195a46c97ede7c294261c6'),
    'large_224_1.0_float': (
        '59e551e166be033d707958cf9e29a6a7',
        '12c0a8442d84beebe8552addf0dcb950'),
    'large_minimalistic_224_1.0_float': (
        '675e7b876c45c57e9e63e6d90a36599c',
        'c1cddbcde6e26b60bdce8e6e2c7cae54'),
    'small_224_0.75_float': (
        'cb65d4e5be93758266aa0a7f2c6708b7',
        'c944bb457ad52d1594392200b48b4ddb'),
    'small_224_1.0_float': (
        '8768d4c2e7dee89b9d02b2d03d65d862',
        '5bec671f47565ab30e540c257bba8591'),
    'small_minimalistic_224_1.0_float': (
        '99cd97fb2fcdad2bf028eb838de69e37',
        '1efbf7e822e03f250f45faa3c6bbe156'),
}

def MobileNetV3(stack_fn,
                last_point_ch,
                input_shape=None,
                alpha=1.0,
                model_type='large',
                minimalistic=False,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                classes=1000,
                pooling=None,
                dropout_rate=0.2,
                include_preprocessing=True,
                mode=None, p=None, eps=None, whiten_layers=[], wcta=False,
                **kwargs):
    
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    keras_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is not type input_tensor')
        if is_input_t_tensor:
            if backend.image_data_format == 'channels_first':
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
        else:
            raise ValueError('input_tensor specified: ', input_tensor,
                             'is not a keras tensor')

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor,
                             'is type: ', type(input_tensor),
                             'which is not a valid type')

        if backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
                input_shape = (3, cols, rows)
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]
                input_shape = (cols, rows, 3)
    # If input_shape is None and input_tensor is None using standart shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                         str(input_shape) + '`')
    if weights == 'imagenet':
        if minimalistic is False and alpha not in [0.75, 1.0] \
                or minimalistic is True and alpha != 1.0:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of `0.75`, `1.0` for non minimalistic'
                             ' or `1.0` for minimalistic only.')

        if rows != cols or rows != 224:
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not 224.'
                          ' Weights for input shape (224, 224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    if include_preprocessing:
        aug_input = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(img_input)
    else:
        aug_input = img_input

    if mode == 'KD' and wcta:
        aug_input = _instance_norm_block(aug_input, mode='KD_WCTA')

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = layers.ZeroPadding2D(padding=correct_pad(backend, aug_input, 3),
                             name='Conv_pad')(aug_input)
    x = layers.Conv2D(16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv')(x)
    
    if mode in ['PADAIN']:
        x = _instance_norm_block(x, mode=mode, p=p, eps=eps)
        
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv/BatchNorm')(x)
    x = layers.Activation(activation)(x)

    if -1 in whiten_layers and mode in ['ISW', 'XDED', 'IN', 'KD']:
        x = _instance_norm_block(x, mode=mode, p=p, eps=eps)
#        if mode == 'ISW':
#            features.append(x)
        
    x, feats = stack_fn(x, kernel, activation, se_ratio, mode=mode, whiten_layers=whiten_layers)
    
#    features.append(feats)
    
    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)

    x = layers.Conv2D(last_conv_ch,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name='Conv_1')(x)
    
    if mode in ['PADAIN']:
        x = _instance_norm_block(x, mode=mode, p=p, eps=eps)

    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1/BatchNorm')(x)
    x = layers.Activation(activation)(x)
    
    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        if channel_axis == 1:
            x = layers.Reshape((last_conv_ch, 1, 1))(x)
        else:
            x = layers.Reshape((1, 1, last_conv_ch))(x)
        x = layers.Conv2D(last_point_ch,
                          kernel_size=1,
                          padding='same',
                          name='Conv_2')(x)
        x = layers.Activation(activation)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name='Predictions/Softmax')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='MobilenetV3' + model_type)

    # Load weights.
    if weights == 'imagenet':
        model_name = "{}{}_224_{}_float".format(
            model_type, '_minimalistic' if minimalistic else '', str(alpha))
        if include_top:
            file_name = 'weights_mobilenet_v3_' + model_name + '.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = 'weights_mobilenet_v3_' + model_name + '_no_top.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = keras_utils.get_file(file_name,
                                            BASE_WEIGHT_PATH + file_name,
                                            cache_subdir='models',
                                            file_hash=file_hash)
        
        model.load_weights(weights_path, skip_mismatch=True, by_name=True)
    
    elif weights is not None:
        model.load_weights(weights)

    return model

def MobileNetV3Large(input_shape=None,
                     alpha=1.0,
                     minimalistic=False,
                     include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     classes=1000,
                     pooling=None,
                     dropout_rate=0.2,
                     include_preprocessing=True,
                     mode=None, 
                     p=None, 
                     eps=None, 
                     whiten_layers=[],
                     wcta=False,
                     **kwargs):

    def stack_fn(x, kernel, activation, se_ratio, mode=mode, whiten_layers=[]):
        def depth(d):
            return _depth(d * alpha)
        # x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id, mode, p, eps, whiten_layers
        x, f1 = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0, mode, p, eps, whiten_layers)
        x, f2 = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 10, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 6, depth(112), 3, 1, se_ratio, activation, 11, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 6, depth(160), kernel, 2, se_ratio, activation, 12, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 13, mode, p, eps, whiten_layers)
        x, _ = _inverted_res_block(x, 6, depth(160), kernel, 1, se_ratio, activation, 14, mode, p, eps, whiten_layers)
        return x, f1+f2
    
        # alternating stride 1-2
        # block id 0..14
    
    return MobileNetV3(stack_fn,
                       1280,
                       input_shape,
                       alpha,
                       'large',
                       minimalistic,
                       include_top,
                       weights,
                       input_tensor,
                       classes,
                       pooling,
                       dropout_rate,
                       include_preprocessing=include_preprocessing,
                       mode=mode, 
                       p=p, 
                       eps=eps,
                       whiten_layers=whiten_layers,
                       wcta=wcta,
                       **kwargs)



def relu(x):
    return layers.ReLU()(x)

def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)

def hard_swish(x):
    return layers.Multiply()([layers.Activation(hard_sigmoid)(x), x])

def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
    
def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(_depth(filters * se_ratio),
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      name=prefix + 'squeeze_excite/Conv_1')(x)
    x = layers.Activation(hard_sigmoid)(x)
    
    if backend.backend() == 'theano':
        # For the Theano backend, we have to explicitly make
        # the excitation weights broadcastable.
        x = layers.Lambda(
            lambda br: backend.pattern_broadcast(br, [True, True, True, False]),
            output_shape=lambda input_shape: input_shape,
            name=prefix + 'squeeze_excite/broadcast')(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    
    return x

def _inverted_res_block(x, expansion, filters, kernel_size, stride,
                        se_ratio, activation, block_id, mode=None, p=None, eps=None, whiten_layers=[]):
    
    features = []
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(_depth(infilters * expansion),
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        
        if mode == 'PADAIN':
            x = _instance_norm_block(x, mode=mode, p=p, eps=eps)
        
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand/BatchNorm')(x)
        x = layers.Activation(activation, name=prefix + 'expand/act_1')(x)
      
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, kernel_size),
                                 name=prefix + 'depthwise/pad')(x)
    x = layers.DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same' if stride == 1 else 'valid',
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    
    if mode == 'PADAIN':
        x = _instance_norm_block(x, mode=mode, p=p, eps=eps)
    
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = layers.Activation(activation, name=prefix + 'expand/act_2')(x)

    if block_id in whiten_layers and mode in []:
        x = _instance_norm_block(x, mode=mode, p=p, eps=eps)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)
        
    x = layers.Conv2D(filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)
    
    if mode == 'PADAIN':
        x = _instance_norm_block(x, mode=mode, p=p, eps=eps)
    
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
        
    if block_id in whiten_layers and mode in ['ISW', 'IN', 'XDED', 'KD']:
        x = _instance_norm_block(x, mode=mode, p=p, eps=eps)
        if mode == 'ISW':
            features.append(x) 
            
    return x, features

