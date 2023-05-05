import tensorflow as tf



class pixelwise_XDEDLoss(tf.keras.losses.Loss):
    def __init__(self, temp_factor=2.0):
        super(pixelwise_XDEDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        self.CLASS_NUM = 1

    def xded_loss(self, input, target):
        
        loss = self.kl_div(tf.nn.softmax(input/self.temp_factor, axis=-1),
                           tf.nn.softmax(target/self.temp_factor, axis=-1)) * (self.temp_factor**2)/input.shape[0]
        return loss

    def call(self, main_out, gts):
        # main_out.shape : [batch, 1, 768, 768]
        # gts.shape : [batch, 768, 768]

        batch_size = main_out.shape[0]
        #print(batch_size)
        flat_gts = tf.reshape(gts,[-1]) # [batch*768*768]
        flat_out = tf.reshape(main_out,(-1, self.CLASS_NUM))

        flat_targets = tf.reshape(main_out,(-1, self.CLASS_NUM))
        # [batch*768*768, 1]

        cur_gt_idx = flat_gts == 1 # [False, True, ...]
        #print(cur_gt_idx.shape)
        x = tf.boolean_mask(flat_out,cur_gt_idx)
        flat_targets = tf.reduce_mean(x) * tf.cast(cur_gt_idx,tf.float32)
        #print(flat_out.shape)
        #print(flat_out.shape)
        
        return self.xded_loss(flat_out, flat_targets)