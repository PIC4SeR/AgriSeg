import tensorflow as tf
import kmeans1d



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