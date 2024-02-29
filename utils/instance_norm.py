import tensorflow as tf
import kmeans1d
from keras.layers import GroupNormalization as InstanceNormalization

def unistyle(x, whiten_cov=False):
    x_mu = tf.math.reduce_mean(x, axis=[2,3], keepdims=True)
    x_var = tf.math.reduce_variance(x, axis=[2,3], keepdims=True)
    x_sig = tf.math.sqrt(x_var+1e-6)
    
    if whiten_cov:
        return x-x_mu
    else:
        return (x-x_mu)/x_sig
    
    

def _instance_norm_block(x, mode=None, p=0.01, eps=1e-5, training=None):
    print(mode)
    if mode in ['ISW', 'IN']:
        return InstanceNormalization(groups=-1)(x)
    elif mode == 'PADAIN':
        return PAdaIN(p=p, eps=eps)(x)
    elif mode in ['KD', 'XDED']:
        return unistyle(x)
    elif mode == 'KD_WCTA':
        return WCTA()(x, training=training)
    
    else:
        return x
    
    
    
def INormBlock(mode=None, p=0.01, eps=1e-5):
    if mode == 'IN':
        return InstanceNormalization(groups=-1)
    elif mode == 'PADAIN':
        return PAdaIN(p=p, eps=eps)
    elif mode == 'WCTA':
        return WCTA()
    else:
        return tf.keras.layers.Identity()
    

    
class PAdaIN(tf.keras.layers.Layer):
    
    def __init__(self, p=0.01, eps=1e-5):
        super(PAdaIN, self).__init__()
        self.p = p
        self.eps = eps
        
    def call(self, inputs, training=None):
        
        if training is None:
            training = tf.keras.backend.learning_phase()
        
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
    

class WCTA(tf.keras.layers.Layer):
    def __init__(self):
        super(WCTA, self).__init__()
        
    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        perm_indices = tf.random.shuffle(tf.range(0, inputs.shape[0]))

        out = self.transfer_global_statistics(inputs, tf.gather(inputs, perm_indices))        
        return out if training else tf.identity(inputs)

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

    def get_config(self):
        cfg = super().get_config()
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
