import tensorflow as tf
import kmeans1d
from keras.layers import GroupNormalization
import keras.layers as nn
import math 

def unistyle(x, whiten_cov=False):
    x_mu = tf.math.reduce_mean(x, axis=[2,3], keepdims=True)
    x_var = tf.math.reduce_variance(x, axis=[2,3], keepdims=True)
    x_sig = tf.math.sqrt(x_var+1e-6)
    if whiten_cov:
        return x-x_mu
    else:
        return (x-x_mu)/x_sig
    


def _instance_norm_block(x, mode=None, p=0.01, eps=1e-5, training=None, i=0):
    print(mode)
    if mode in ['ISW', 'IBN']:
        return IBN(channels=16 if i<1 else 24)(x)
#        return GroupNormalization(groups=-1)(x)
    elif mode == 'PADAIN':
        return PAdaIN(p=p, eps=eps)(x)
    elif mode in ['KD', 'XDED']:
        return UniStyle()(x)
    elif mode == 'KD_WCTA':
        return WCTA()(x)
    else:
        return tf.identity(x)
    
    
    
def INormBlock(mode=None, p=0.01, eps=1e-5):
    if mode == 'IBN':
        return GroupNormalization(groups=-1)
    elif mode == 'PADAIN':
        return PAdaIN(p=p, eps=eps)
    elif mode == 'WCTA':
        return WCTA()
    else:
        return tf.keras.layers.Identity()
    
class UniStyle(tf.keras.layers.Layer):
    
    def __init__(self, whiten_cov=False):
        super(UniStyle, self).__init__()
        self.whiten_cov = whiten_cov
        
    def call(self, x, training=None):
        
        # if training is None:
        #     training = tf.keras.backend.learning_phase()
        # if training:
        #     return tf.identity(x)
        
        x_mu = tf.math.reduce_mean(x, axis=[2,3], keepdims=True)
        x_var = tf.math.reduce_variance(x, axis=[2,3], keepdims=True)
        x_sig = tf.math.sqrt(x_var+1e-6)
        
        if self.whiten_cov:
            return x-x_mu
        else:
            return (x-x_mu)/x_sig


class PAdaIN(tf.keras.layers.Layer):
    
    def __init__(self, p=0.01, eps=1e-5):
        super(PAdaIN, self).__init__()
        self.p = p
        self.eps = eps
        
    def call(self, inputs, training=None):
        
        if training is None:
            training = tf.keras.backend.learning_phase()
        if training:
            return tf.identity(inputs)
        
        permute = tf.random.uniform([], minval=0, maxval=1) < self.p
        
        if permute:
            perm_indices = tf.random.shuffle(tf.range(0, inputs.shape[0]))
        else:
            return tf.identity(inputs)

        out = self.ada_in(inputs, tf.gather(inputs, perm_indices))
        return out

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
    

class WCTA(tf.keras.layers.Layer):
    def __init__(self, p=0.1):
        super(WCTA, self).__init__()
        self.p = p
    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if training:
            return tf.identity(inputs)

        if tf.random.uniform([], minval=0, maxval=1) < self.p:
            perm_indices = tf.random.shuffle(tf.range(0, inputs.shape[0]))
        else:
            return tf.identity(inputs)
        
        out = self.transfer_global_statistics(inputs, tf.gather(inputs, perm_indices))        
        return out

    def transfer_global_statistics(self, trg, src):
        trg = tf.transpose(trg, [0, 3, 1, 2])
        src = tf.transpose(src, [0, 3, 1, 2])

        src_shape = tf.shape(src)
        batch_size = src_shape[0]
        chans = src_shape[1]
        src_height = src_shape[2]
        src_width = src_shape[3]
        trg_shape = tf.shape(trg)
        trg_height = trg_shape[2]
        trg_width = trg_shape[3]

        src_flattened = tf.reshape(src, (batch_size, chans, -1)) # [B x C x H1*W1]
        trg_flattened = tf.reshape(trg, (batch_size, chans, -1)) # [B x C x H2*W2]
        src_mean = tf.reduce_mean(src_flattened, axis=-1, keepdims=True) # [B, C, 1]
        trg_mean = tf.reduce_mean(trg_flattened, axis=-1, keepdims=True) # [B, C, 1]
        src_reduced = src_flattened - src_mean # [B x C x H1*W1]
        trg_reduced = trg_flattened - trg_mean # [B x C x H2*W2]
        src_cov_mat = tf.linalg.matmul(src_reduced, src_reduced, transpose_b=True) / float(src_height*src_width - 1)# [B x C x C]
        trg_cov_mat = tf.linalg.matmul(trg_reduced, trg_reduced, transpose_b=True) / float(trg_height*trg_width - 1) # [B x C x C]
        src_eigvals, src_eigvecs = tf.linalg.eigh(src_cov_mat) # eigval -> [B, C], eigvecs -> [B, C, C]
        src_eigvals = tf.clip_by_value(src_eigvals, clip_value_min=1e-8, clip_value_max=float(tf.reduce_max(src_eigvals))) # valid op since covmat is positive (semi-)definit
        src_eigvals_sqrt = tf.sqrt(src_eigvals)[..., None] # [B, C, 1]
        trg_eigvals, trg_eigvecs = tf.linalg.eigh(trg_cov_mat) # eigval -> [B, C], eigvecs -> [B, C, C]
        trg_eigvals = tf.clip_by_value(trg_eigvals, clip_value_min=1e-8, clip_value_max=float(tf.reduce_max(trg_eigvals))) # valid op since covmat is positive (semi-)definit
        trg_eigvals_sqrt = tf.sqrt(trg_eigvals)[..., None] # [B, C, 1]

        # transfer color statistics form source to target
        W_trg = tf.linalg.matmul(trg_eigvecs, (1 / trg_eigvals_sqrt) * trg_eigvecs, transpose_b=True)
        trg_white = tf.linalg.matmul(W_trg, trg_reduced)
        W_src_inv = tf.linalg.matmul(src_eigvecs, src_eigvals_sqrt * src_eigvecs, transpose_b=True)
        trg_transformed = tf.linalg.matmul(W_src_inv, trg_white) + src_mean
        trg_transformed = tf.reshape(trg_transformed, (batch_size, chans, trg_height, trg_width))

        alpha = tf.random.uniform((batch_size, 1, 1, 1))
        alpha = tf.clip_by_value(alpha, clip_value_min=0.0, clip_value_max=0.95)
        trg_transformed = (alpha * trg) + ((1 - alpha) * trg_transformed)

        trg_transformed = tf.transpose(trg_transformed, [0, 2, 3, 1])
        return trg_transformed

    def getcfg(self):
        cfg = super().get_cfg()
        return cfg 
    

class CovMatrix_ISW(tf.keras.layers.Layer):
    def __init__(self, dim, relax_denom=0, clusters=50):
        super(CovMatrix_ISW, self).__init__()

        self.dim = dim
        self.i = tf.eye(dim, dim)

        # print(torch.ones(16, 16).triu(diagonal=1))
        self.reversal_i = tf.ones((dim, dim))
        self.reversal_i = tf.linalg.band_part(self.reversal_i,0,-1) - tf.linalg.band_part(self.reversal_i,0,0)
        # num_off_diagonal = ((dim * dim - dim) // 2)  # number of off-diagonal
        self.num_off_diagonal = tf.math.reduce_sum(self.reversal_i)
        self.num_sensitive = 0
        self.var_matrix = None
        self.count_var_cov = 0
        self.mask_matrix = None
        self.clusters = clusters
        #print("num_off_diagonal", self.num_off_diagonal)
        if relax_denom == 0:    # kmeans1d clustering setting for ISW
            #print("relax_denom == 0!!!!!")
            #print("cluster == ", self.clusters)
            self.margin = 0
        else:                   # do not use
            self.margin = self.num_off_diagonal // relax_denom

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def get_mask_matrix(self, mask=True):
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, 0, self.num_sensitive

    def reset_mask_matrix(self):
        self.mask_matrix = None

    def set_mask_matrix(self):
        # torch.set_printoptions(threshold=500000)
        self.var_matrix = self.var_matrix / self.count_var_cov
        var_flatten = tf.reshape(self.var_matrix, [-1])

        if self.margin == 0:    # kmeans1d clustering setting for ISW
            clusters, centroids = kmeans1d.cluster(var_flatten, self.clusters) # 50 clusters
            num_sensitive = var_flatten.shape[0] - clusters.count(0)  # 1: Insensitive Cov, 2~50: Sensitive Cov
            #print("num_sensitive, centroids =", num_sensitive, centroids)
            _, indices = tf.math.top_k(input=var_flatten, k=int(num_sensitive))
        else:                   # do not use
            num_sensitive = self.num_off_diagonal - self.margin
            #print("num_sensitive = ", num_sensitive)
            _, indices = tf.math.top_k(input=var_flatten, k=int(num_sensitive))
        
        updates = tf.ones_like(indices[:,None])[:,0]
        shape = tf.constant([self.dim**2])
        mask_matrix = tf.scatter_nd(indices[:,None], updates, shape)            

        if self.mask_matrix is not None:
            a = tf.cast(self.mask_matrix,tf.int32)
            b = tf.cast(tf.reshape(mask_matrix,[self.dim, self.dim]),tf.int32)
            print(a.shape, b.shape)
            self.mask_matrix = tf.cast(a & b, tf.float32())
            (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = tf.reshape(mask_matrix,[self.dim, self.dim])
        self.num_sensitive = tf.math.reduce_sum(self.mask_matrix)
        #print("Check whether two ints are same", num_sensitive, self.num_sensitive)

        self.var_matrix = None
        self.count_var_cov = 0

        if not tf.config.experimental.list_physical_devices('GPU'):
            print("Covariance Info: (CXC Shape, Num_Off_Diagonal)", self.mask_matrix.shape, self.num_off_diagonal)
            print("Selective (Sensitive Covariance)", self.num_sensitive)


    def set_variance_of_covariance(self, var_cov):
        if self.var_matrix is None:
            self.var_matrix = var_cov
        else:
            self.var_matrix = self.var_matrix + var_cov
        self.count_var_cov += 1
        
        
        
@tf.function
def instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov):
    f_cor, B = get_covariance_matrix(f_map, eye=eye)
    f_cor_masked = f_cor * tf.cast(mask_matrix, tf.float32)

    off_diag_sum = tf.math.reduce_sum(tf.math.abs(f_cor_masked), axis=(1,2), keepdims=True) - margin # B X 1 X 1
    loss = tf.clip_by_value(off_diag_sum / tf.cast(num_remove_cov, tf.float32), clip_value_min=0.0, 
                            clip_value_max=tf.float32.max) # B X 1 X 1
    loss = tf.math.reduce_sum(loss) / B

    return loss


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, H, W, C = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = tf.eye(C)
    f_map = tf.reshape(f_map,[B, -1, C])  # B X C X H X W > B X C X (H X W)
    f_cor = tf.linalg.matmul(tf.transpose(f_map,[0,2,1]), f_map) 
    f_cor = f_cor / (HW-1) + (eps * eye)  # C X C / HW
    return f_cor, B

def is_channels_first(data_format):
    """
    Is tested data format channels first.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns:
    -------
    bool
        A flag.
    """
    return data_format == "channels_first"

def get_channel_axis(data_format):
    """
    Get channel axis.

    Parameters:
    ----------
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.

    Returns:
    -------
    int
        Channel axis.
    """
    return 1 if is_channels_first(data_format) else -1

class BatchNorm(nn.BatchNormalization):
    """
    MXNet/Gluon-like batch normalization.

    Parameters:
    ----------
    momentum : float, default 0.9
        Momentum for the moving average.
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 momentum=0.9,
                 epsilon=1e-5,
                 data_format="channels_last",
                 **kwargs):
        super(BatchNorm, self).__init__(
            axis=get_channel_axis(data_format),
            momentum=momentum,
            epsilon=epsilon,
            **kwargs)


class InstanceNorm(nn.Layer):
    """
    MXNet/Gluon-like instance normalization layer as in 'Instance Normalization: The Missing Ingredient for Fast
    Stylization' (https://arxiv.org/abs/1607.08022). On the base of `tensorflow_addons` implementation.

    Parameters:
    ----------
    epsilon : float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center : bool, default True
        If True, add offset of `beta` to normalized tensor. If False, `beta` is ignored.
    scale : bool, default False
        If True, multiply by `gamma`. If False, `gamma` is not used.
    beta_initializer : str, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer : str, default 'ones'
        Initializer for the gamma weight.
    beta_regularizer : object or None, default None
        Optional regularizer for the beta weight.
    gamma_regularizer : object or None, default None
        Optional regularizer for the gamma weight.
    beta_constraint : object or None, default None
        Optional constraint for the beta weight.
    gamma_constraint : object or None, default None
        Optional constraint for the gamma weight.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 epsilon=1e-5,
                 center=True,
                 scale=False,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 data_format="channels_last",
                 **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = -1
        self.axis = get_channel_axis(data_format)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):
        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super(InstanceNorm, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)
        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)
        outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        return outputs

    def get_cfg(self):
        cfg = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_cfg = super(InstanceNorm, self).get_cfg()
        return dict(list(base_cfg.items()) + list(cfg.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)
        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon)
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError("Axis {} of input tensor should have a defined dimension but the layer received an input "
                             "with shape {}".format(self.axis, input_shape))

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]
        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):
        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError("Number of groups ({}) cannot be more than the number of channels ({})".format(
                self.groups, dim))
        if (dim % self.groups) != 0:
            raise ValueError('Number of groups ({}) must be a multiple of the number of channels ({})'.format(
                self.groups, dim))

    def _check_axis(self):
        if self.axis == 0:
            raise ValueError("You are trying to normalize your batch axis. Do you want to use "
                             "tf.layer.batch_normalization instead")

    def _create_input_spec(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape),
            axes={self.axis: dim})

    def _add_gamma_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint)
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint)
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape


class IBN(nn.Layer):
    """
    Instance-Batch Normalization block from 'Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net,'
    https://arxiv.org/abs/1807.09441.

    Parameters:
    ----------
    channels : int
        Number of channels.
    inst_fraction : float, default 0.5
        The first fraction of channels for normalization.
    inst_first : bool, default True
        Whether instance normalization be on the first part of channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 first_fraction=0.5,
                 inst_first=True,
                 data_format="channels_last",
                 **kwargs):
        super(IBN, self).__init__(**kwargs)
        self.inst_first = inst_first
        self.data_format = data_format
        h1_channels = int(math.floor(channels * first_fraction))
        h2_channels = channels - h1_channels
        self.split_sections = [h1_channels, h2_channels]

        if self.inst_first:
            self.inst_norm = InstanceNorm(
                scale=True,
                data_format=data_format,
                name="inst_norm")
            self.batch_norm = BatchNorm(
                data_format=data_format,
                name="batch_norm")
        else:
            self.batch_norm = BatchNorm(
                data_format=data_format,
                name="batch_norm")
            self.inst_norm = InstanceNorm(
                scale=True,
                data_format=data_format,
                name="inst_norm")

    def call(self, x, training=None):
        axis = get_channel_axis(self.data_format)
        x1, x2 = tf.split(x, num_or_size_splits=self.split_sections, axis=axis)
        if self.inst_first:
            x1 = self.inst_norm(x1, training=training)
            x2 = self.batch_norm(x2, training=training)
        else:
            x1 = self.batch_norm(x1, training=training)
            x2 = self.inst_norm(x2, training=training)
        x = tf.concat([x1, x2], axis=axis)
        return x

