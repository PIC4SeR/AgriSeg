import tensorflow as tf

from keras import backend
from keras import models
from keras.applications import imagenet_utils
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

# isort: off
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

# TODO(scottzhu): Change this to the GCS path.
BASE_WEIGHT_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/"
)
WEIGHTS_HASHES = {
    "large_224_0.75_float": (
        "765b44a33ad4005b3ac83185abf1d0eb",
        "40af19a13ebea4e2ee0c676887f69a2e",
    ),
    "large_224_1.0_float": (
        "59e551e166be033d707958cf9e29a6a7",
        "07fb09a5933dd0c8eaafa16978110389",
    ),
    "large_minimalistic_224_1.0_float": (
        "675e7b876c45c57e9e63e6d90a36599c",
        "ec5221f64a2f6d1ef965a614bdae7973",
    ),
    "small_224_0.75_float": (
        "cb65d4e5be93758266aa0a7f2c6708b7",
        "ebdb5cc8e0b497cd13a7c275d475c819",
    ),
    "small_224_1.0_float": (
        "8768d4c2e7dee89b9d02b2d03d65d862",
        "d3e8ec802a04aa4fc771ee12a9a9b836",
    ),
    "small_minimalistic_224_1.0_float": (
        "99cd97fb2fcdad2bf028eb838de69e37",
        "cde8136e733e811080d9fcd8a252f7e4",
    ),
}

layers = VersionAwareLayers()

def MobileNetV3(
    stack_fn,
    last_point_ch,
    input_shape=None,
    alpha=1.0,
    model_type="large",
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  "
            f"Received weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            f"Received classes={classes}"
        )

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    layer_utils.get_source_inputs(input_tensor)
                )
            except ValueError:
                raise ValueError(
                    "input_tensor: ",
                    input_tensor,
                    "is not type input_tensor.  "
                    f"Received type(input_tensor)={type(input_tensor)}",
                )
        if is_input_t_tensor:
            if backend.image_data_format() == "channels_first":
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError(
                        "When backend.image_data_format()=channels_first, "
                        "input_shape[1] must equal "
                        "backend.int_shape(input_tensor)[1].  Received "
                        f"input_shape={input_shape}, "
                        "backend.int_shape(input_tensor)="
                        f"{backend.int_shape(input_tensor)}"
                    )
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError(
                        "input_shape[1] must equal "
                        "backend.int_shape(input_tensor)[2].  Received "
                        f"input_shape={input_shape}, "
                        "backend.int_shape(input_tensor)="
                        f"{backend.int_shape(input_tensor)}"
                    )
        else:
            raise ValueError(
                "input_tensor specified: ",
                input_tensor,
                "is not a keras tensor",
            )

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(
                "input_tensor: ",
                input_tensor,
                "is type: ",
                type(input_tensor),
                "which is not a valid type",
            )

        if backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == "channels_first":
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
                input_shape = (3, cols, rows)
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]
                input_shape = (cols, rows, 3)
    # If input_shape is None and input_tensor is None using standard shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError(
            "Input size must be at least 32x32; Received `input_shape="
            f"{input_shape}`"
        )
    if weights == "imagenet":
        if (
            not minimalistic
            and alpha not in [0.75, 1.0]
            or minimalistic
            and alpha != 1.0
        ):
            raise ValueError(
                "If imagenet weights are being loaded, "
                "alpha can be one of `0.75`, `1.0` for non minimalistic "
                "or `1.0` for minimalistic only."
            )

        if rows != cols or rows != 224:
            logging.warning(
                "`input_shape` is undefined or non-square, "
                "or `rows` is not 224. "
                "Weights for input shape (224, 224) will be "
                "loaded as the default."
            )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = img_input
    if include_preprocessing:
        x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv/BatchNorm"
    )(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="Conv_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm"
    )(x)
    x = activation(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Conv2D(
            last_point_ch,
            kernel_size=1,
            padding="same",
            use_bias=True,
            name="Conv_2",
        )(x)
        x = activation(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(
            classes, kernel_size=1, padding="same", name="Logits"
        )(x)
        x = layers.Flatten()(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(
            activation=classifier_activation, name="Predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name="MobilenetV3" + model_type)

    # Load weights.
    if weights == "imagenet":
        model_name = "{}{}_224_{}_float".format(
            model_type, "_minimalistic" if minimalistic else "", str(alpha)
        )
        if include_top:
            file_name = "weights_mobilenet_v3_" + model_name + ".h5"
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = "weights_mobilenet_v3_" + model_name + "_no_top_v2.h5"
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHT_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export("keras.applications.MobileNetV3Small")
def MobileNetV3Small(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
        x = _inverted_res_block(x, 72.0 / 16, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88.0 / 24, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(
            x, 4, depth(40), kernel, 2, se_ratio, activation, 3
        )
        x = _inverted_res_block(
            x, 6, depth(40), kernel, 1, se_ratio, activation, 4
        )
        x = _inverted_res_block(
            x, 6, depth(40), kernel, 1, se_ratio, activation, 5
        )
        x = _inverted_res_block(
            x, 3, depth(48), kernel, 1, se_ratio, activation, 6
        )
        x = _inverted_res_block(
            x, 3, depth(48), kernel, 1, se_ratio, activation, 7
        )
        x = _inverted_res_block(
            x, 6, depth(96), kernel, 2, se_ratio, activation, 8
        )
        x = _inverted_res_block(
            x, 6, depth(96), kernel, 1, se_ratio, activation, 9
        )
        x = _inverted_res_block(
            x, 6, depth(96), kernel, 1, se_ratio, activation, 10
        )
        return x

    return MobileNetV3(
        stack_fn,
        1024,
        input_shape,
        alpha,
        "small",
        minimalistic,
        include_top,
        weights,
        input_tensor,
        classes,
        pooling,
        dropout_rate,
        classifier_activation,
        include_preprocessing,
    )


def MobileNetV3Large(
    input_shape=None,
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = _inverted_res_block(x, 1, depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(
            x, 6, depth(112), 3, 1, se_ratio, activation, 10
        )
        x = _inverted_res_block(
            x, 6, depth(112), 3, 1, se_ratio, activation, 11
        )
        x = _inverted_res_block(
            x, 6, depth(160), kernel, 2, se_ratio, activation, 12
        )
        x = _inverted_res_block(
            x, 6, depth(160), kernel, 1, se_ratio, activation, 13
        )
        x = _inverted_res_block(
            x, 6, depth(160), kernel, 1, se_ratio, activation, 14
        )
        return x

    return MobileNetV3(
        stack_fn,
        1280,
        input_shape,
        alpha,
        "large",
        minimalistic,
        include_top,
        weights,
        input_tensor,
        classes,
        pooling,
        dropout_rate,
        classifier_activation,
        include_preprocessing,
    )



def relu(x):
    return layers.ReLU()(x)

def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)

def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])

def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(
        keepdims=True, name=prefix + "squeeze_excite/AvgPool"
    )(inputs)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv",
    )(x)
    x = layers.ReLU(name=prefix + "squeeze_excite/Relu")(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv_1",
    )(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + "squeeze_excite/Mul")([inputs, x])
    return x

def _inverted_res_block(
    x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id
):
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    shortcut = x
    prefix = "expanded_conv/"
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = f"expanded_conv_{block_id}/"
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand/BatchNorm",
        )(x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=prefix + "depthwise/pad",
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise/BatchNorm",
    )(x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project/BatchNorm",
    )(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + "Add")([shortcut, x])
    return x