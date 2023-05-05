import tensorflow as tf
import tensorflow.keras.backend as K
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
def mIoU(y_true, y_pred):
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

def weighted_cross_entropy_loss(y_true_labels, y_pred_logits):
    
    class_weights = [2.602, 6.707, 3.522, 9.877, 9.685, 9.398, 10.288, 9.969, 
                                4.336, 9.454, 7.617, 9.405, 10.359, 6.373, 10.231, 10.262, 
                                10.264, 10.394, 10.094, 0.0]

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_labels, logits=y_pred_logits)  # (batch_size, img_h, img_w)

    weights = tf.gather(class_weights, y_true_labels)  # (batch_size, img_h, img_w)
    losses = tf.multiply(losses, weights)

    return tf.reduce_mean(losses)

def binary_weighted_cross_entropy(beta: float = 0.5, from_logits: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Weighted cross entropy. All positive examples get weighted by the coefficient beta:
        WCE(p, p̂) = −[β*p*log(p̂) + (1−p)*log(1−p̂)]
    To decrease the number of false negatives, set β>1. To decrease the number of false positives, set β<1.
    If last layer of network is a sigmoid function, y_pred needs to be reversed into logits before computing the
    weighted cross entropy. To do this, we're using the same method as implemented in Keras binary_crossentropy:
    https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
    Used as loss function for binary image segmentation with one-hot encoded masks.
    :param beta: Weight coefficient (float)
    :param is_logits: If y_pred are logits (bool, default=False)
    :return: Weighted cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the weighted cross entropy.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, 1))
        :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        if not from_logits:
            y_pred = convert_to_logits(y_pred)

        wce_loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred[...,0,0], pos_weight=beta)

        # Average over each data point/image in batch
        axis_to_reduce = range(1, K.ndim(wce_loss))
        wce_loss = K.mean(wce_loss, axis=axis_to_reduce)

        return wce_loss

    return loss

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