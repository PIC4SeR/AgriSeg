import tensorflow as tf
import keras.backend as K
import numpy as np
from typing import Callable

def loss_IoU(y_true, y_pred): 
    if len(y_pred.shape) != len(y_true.shape):
        y_true = y_true[...,None]
    intersection_tensor=tf.math.multiply(tf.cast(y_true, dtype=np.float32),tf.cast(y_pred, dtype=np.float32))
    inter=tf.reduce_sum(intersection_tensor, keepdims=True)
    
    #union= a+b-intersection
    union=tf.reduce_sum(tf.math.subtract(tf.math.add(tf.cast(y_true, dtype=np.float32),
                                                     tf.cast(y_pred, dtype=np.float32)),intersection_tensor), keepdims=True)
    
    iou= tf.math.divide(inter,union)
    return 1-iou

#metric
def mIoU_old(y_true, y_pred):
    if len(y_pred.shape) != len(y_true.shape):
        #y_true = y_true[...,None]
        y_pred = y_pred[...,0]
    
    threshold = tf.constant([0.9])
    y_pred_threshold=tf.cast(tf.math.greater(y_pred, threshold),tf.int32)
    y_true=tf.cast(y_true,tf.int32)
    

    intersection_tensor=tf.math.multiply(y_true,y_pred_threshold)
    inter=tf.reduce_sum(intersection_tensor)
    
    #union= a+b-intersection
    union=tf.reduce_sum(tf.math.subtract(tf.math.add(y_true,y_pred_threshold),intersection_tensor))
    
    
    return tf.math.divide(inter,union)

#metric
def mIoU(y_true, y_pred):
    if len(y_pred.shape) != len(y_true.shape):
        #y_true = y_true[...,None]
        y_pred = y_pred[...,0]

    y_pred = tf.math.sigmoid(y_pred)
    threshold = tf.constant([0.5])
    y_pred_threshold=tf.cast(tf.math.greater(y_pred, threshold),tf.int32)

    y_true=tf.cast(y_true,tf.int32)
    

    intersection_tensor=tf.math.multiply(y_true,y_pred_threshold)
    inter=tf.reduce_sum(intersection_tensor)
    
    #union= a+b-intersection
    union=tf.reduce_sum(tf.math.subtract(tf.math.add(y_true,y_pred_threshold),intersection_tensor))
    
    
    return tf.math.divide(inter,union)

def weighted_cross_entropy_loss(y_true_labels, y_pred_logits):
    
    class_weights = [2.602, 6.707, 3.522, 9.877, 9.685, 9.398, 10.288, 9.969, 
                                4.336, 9.454, 7.617, 9.405, 10.359, 6.373, 10.231, 10.262, 
                                10.264, 10.394, 10.094, 0.0]

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_labels, logits=y_pred_logits)  # (batch_size, img_h, img_w)

    weights = tf.gather(class_weights, y_true_labels)  # (batch_size, img_h, img_w)
    losses = tf.multiply(losses, weights)

    return tf.reduce_mean(losses)


#Keras
def DiceBCELoss(targets, inputs, smooth=1e-6):    
       
    #flatten label and prediction tensors
    #inputs = K.flatten(inputs)
    #targets = K.flatten(targets)
    
    BCE = K.binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE

def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


class ContrastiveLoss(tf.keras.losses.Loss):
    """
    Constrastive loss for features-matching clustering ispired by ArXiv:2002.05709
    """
    def __init__(self, batch_size, weight=0.1, temperature=0.1, name='cluster_loss'):
        super().__init__(name=name)
        self.contrastive_labels = tf.range(batch_size)
        self.weight = weight
        self.temperature = temperature
        self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                                 reduction=tf.keras.losses.Reduction.NONE)
        self.reduction = tf.keras.losses.Reduction.SUM
        
    def call(self, y_pred, y_pred_):
        B, H, W, C = y_pred.shape
        y_pred = tf.reshape(y_pred, (B, H*W*C))
        y_pred_ = tf.reshape(y_pred_, (B, H*W*C))
        projections_1 = tf.math.l2_normalize(y_pred, axis=1)
        projections_2 = tf.math.l2_normalize(y_pred_, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )
        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = self.cce(self.contrastive_labels, similarities)
        loss_2_1 = self.cce(self.contrastive_labels, tf.transpose(similarities))
        
        return (tf.reduce_sum(loss_1_2) + tf.reduce_sum(loss_2_1)) / 2 * self.weight

def loss_filter(p, n=0.999):
    if p <= n:
        return 1
    elif p > (1+n)/2:
        return 0
    else:
        return ((n+1-2*p) / (1-n)) ** 2