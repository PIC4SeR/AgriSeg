import tensorflow as tf



class pixelwise_XDEDLoss(tf.keras.losses.Loss):
    def __init__(self, temp_factor=2.0):
        super(pixelwise_XDEDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        self.CLASS_NUM = 1

    def xded_loss(self, input, target):
        pred_t = tf.math.sigmoid(target / self.temp_factor)
        pred_t_h = tf.concat([tf.ones_like(pred_t) - pred_t, pred_t], axis=-1)
        pred = tf.math.sigmoid(input / self.temp_factor)
        pred_h = tf.concat([tf.ones_like(pred) - pred, pred], axis=-1)
        aux_loss = self.kl_div(pred_t_h, pred_h) * self.temp_factor ** 2
        return tf.reduce_mean(aux_loss)

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
        flat_targets = tf.reduce_mean(x) * tf.cast(tf.reshape(cur_gt_idx,(-1, self.CLASS_NUM)),tf.float32)
        #print(flat_out.shape)
        #print(flat_out.shape)
        
        return self.xded_loss(flat_out, flat_targets)