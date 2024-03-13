import inspect
import os
import sys
import time
import tensorflow as tf
import keras.backend as K
import numpy as np


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
    a = tf.cast(tf.where(p <= n, 1, 0), bool)
    b = tf.cast(tf.where(p > (1+n)/2, 1, 0), bool)
    c = tf.logical_not(tf.logical_or(a, b))
    # print(a, b, c)
    o = tf.cast(tf.where(a, 1, 0), tf.float32) + tf.cast(tf.where(c, ((n+1-2*p) / (1-n)) ** 2, 0), tf.float32)
    # print(o)
    return o
    

def uniform_soup(model, path, by_name = False):
    
    if not isinstance(path, list):
        path = [path]
    soups = []
    for i, model_path in enumerate(path):
        model.load_weights(model_path, by_name = by_name)
        soup = [np.array(w) for w in model.weights]
        soups.append(soup)
    if 0 < len(soups):
        for w1, w2 in zip(model.weights, list(zip(*soups))):
            tf.keras.backend.set_value(w1, np.mean(w2, axis = 0))
    return model


def greedy_soup(model, path, data, metric, update_greedy = False, compare = np.greater_equal, by_name = False, digits = 4, verbose = True, y_true = "y_true"):
    # try:
    #     import tensorflow as tf
    # except:
    #     print("If you want to use 'Model Soup for Tensorflow2', please install 'tensorflow2'")
    #     return model
    
    if not isinstance(path, list):
        path = [path]
    score, soup = None, []
    input_key = [inp.name for inp in model.inputs]
    input_cnt = len(input_key)
    for i, model_path in enumerate(path):
        if update_greedy:
            model.load_weights(model_path, by_name = by_name)
            for w1, w2 in zip(model.weights, soup):
                tf.keras.backend.set_value(w1, np.mean([w1, w2], axis = 0))
        else:
            model = uniform_soup(model, soup + [model_path], by_name = by_name)
                
        iterator = iter(data)
        history = []
        step = 0
        start_time = time.time()
        while True:
            try:
                text = ""
                iter_data = next(iterator)
                if not isinstance(iter_data, dict):
                    x = iter_data[:input_cnt]
                    y = list(iter_data[input_cnt:])
                    d_cnt = len(y[0])
                else:
                    x = [iter_data[k] for k in input_key if k in iter_data]
                step += 1
                #del x

                logits, _ = model.predict(x)
                if not isinstance(logits, list):
                    logits = [logits]
                if isinstance(iter_data, dict):
                    metric_key = [key for key in inspect.getfullargspec(metric).args if key != "self"]
                    if len(metric_key) == 0:
                        metric_key = [y_true]
                    y = [iter_data[k] for k in metric_key if k in iter_data]
                    d_cnt = len(y[0])
                metric_val = np.array(metric(*(y + logits)))
                if np.ndim(metric_val) == 0:
                    metric_val = [float(metric_val)] * d_cnt
                history += list(metric_val)
                #del y, logits

                if verbose:
                    sys.stdout.write("\r[{name}] step: {step} - time: {time:.2f}s - {key}: {val:.{digits}f}".format(name = os.path.basename(model_path), step = step, time = (time.time() - start_time), key = metric.__name__ if hasattr(metric, "__name__") else str(metric), val = np.nanmean(history), digits = digits))
                    sys.stdout.flush()
            except (tf.errors.OutOfRangeError, StopIteration):
                print("")
                #gc.collect()
                break
        if 0 < len(history) and (score is None or compare(np.nanmean(history), score)):
            score = np.nanmean(history)
            if update_greedy:
                soup = [np.array(w) for w in model.weights]
            else:
                soup += [model_path]
    if len(soup) != 0:
        if update_greedy:
            for w1, w2 in zip(model.weights, soup):
                tf.keras.backend.set_value(w1, w2)
        else:
            model = uniform_soup(model, soup, by_name = by_name)
        if verbose:
            print("greedy soup best score : {val:.{digits}f}".format(val = score, digits = digits))
    return model


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for param in self.model.variables:
            if param.trainable:
                self.shadow[param.name] = param.value()

    def update(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.shadow
                new_average = (1.0 - self.decay) * param.value() + self.decay * self.shadow[param.name]
                self.shadow[param.name] = new_average

    def apply_shadow(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.shadow
                self.backup[param.name] = param.value()
                param.assign(self.shadow[param.name])
                
    def restore(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.backup
                param.assign(self.backup[param.name])
        self.backup = {}

def normalize(logit, mode=None):
    if mode is None or mode == '':
        return logit
    
    stdv = tf.math.reduce_std(logit, axis=[-2,-3], keepdims=True)
    if 'norm' in mode:
        mean = tf.reduce_mean(logit, axis=[-2,-3], keepdims=True)
        return (logit - mean) / (1e-7 + stdv)
    elif 'std' in mode:
        return (logit) / (1e-7 + stdv)
