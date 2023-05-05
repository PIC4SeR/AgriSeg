import tensorflow as tf
import os
import glob
import random

MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

def resize(img, IMG_SIZE):
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def get_image_paths(dir):
    return sorted(glob.glob(os.path.join(dir, '*/*.png')))

def get_segmentation_paths(dir):
    return sorted(glob.glob(os.path.join(dir, '*/*_labelIds.png')))

def load_and_resize_image(path, IMG_SIZE):
    with tf.device('/cpu:0'):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def load_and_resize_segmentation(path, IMG_SIZE):  # read segmentation (class label for each pixel), and resize it to image_size
    id2label = tf.constant([19, 19, 19, 19, 19, 19, 19, 0, 1, 19, 19, 2, 3, 
                                     4, 19, 19, 19, 5, 19, 6, 7, 8, 9, 10, 11, 12, 
                                     13, 14, 15, 19, 19, 16, 17, 18, 19], tf.int32)
    
    with tf.device('/cpu:0'):
        seg = tf.io.read_file(path)
        seg = tf.image.decode_png(seg, channels=1, dtype=tf.uint8)
        seg = tf.image.resize(seg, [IMG_SIZE, IMG_SIZE], method='nearest')  # resize with 'nearest' method to avoid creating new classes
        seg = tf.squeeze(seg)
        seg = tf.gather(id2label, tf.cast(seg, tf.int32))  # (image_size[0], image_size[1])
    return seg

def normalize(img):
    return img / 255.0

def normalize_imagenet(image):
    """Normalize the image to zero mean and unit variance."""
    offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
    image -= offset

    scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
    image /= scale
    return image

def standardize_image(img):  # map pixel intensities to float32 in [-1, 1]
    return tf.cast(img, tf.float32) / 127.5 - 1.0


def unstandardize_image(img, imagenet=False):  # map pixel intensities to uint8 in [0, 255]
    if imagenet:
        scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
        img *= scale
        offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
        img += offset
    else:
        img = img * 255.0
        img = tf.clip_by_value(img, 0.0, 255.0)
    return tf.cast(img, tf.uint8)

def data_aug(img, seg):
    if tf.random.uniform([], minval=0, maxval=1) > 0.5:  # horizontal flip with probability 0.5
        img = tf.reverse(img, axis=[1])
        seg = tf.reverse(seg, axis=[1])
    if tf.random.uniform([], minval=0, maxval=1) > 1.0:  
        img = tf.image.random_brightness(img, max_delta=0.08)  # random brightness
        img = tf.image.random_contrast(img, lower=0.95, upper=1.05)  # random contrast
    return img, seg

    
def horizontal_flip(img, seg):
    if tf.random.uniform([], minval=0, maxval=1) > 0.5:  # horizontal flip with probability 0.5
        img = tf.reverse(img, axis=[1])
        seg = tf.reverse(seg, axis=[1])
    return img, seg


def grayscale(x, y):
    """
    Image to gray with 10% probability.
    """
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    if choice < 0.3:
        gray_tensor = tf.image.rgb_to_grayscale(x)
        return tf.repeat(gray_tensor, repeats=3, axis=-1), y
    else:
        return x, y


# def random_crop(x, y):
#     """
#     Random crop between 80% and 100%.
#     """
#     INPUT_SIZE = 224
#     perc = tf.math.floor(tf.random.uniform(shape=[], minval=0.8, maxval=1., dtype=tf.float32)*224)
#     image_crops = tf.image.random_crop(x, [perc, perc, 3])
#     return tf.image.resize(image_crops, [INPUT_SIZE,INPUT_SIZE]), y

    
    
    
    