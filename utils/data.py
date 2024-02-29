import os
import glob
import math
import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

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

def load_and_resize_segmentation(path, IMG_SIZE):  # read segmentation (class label for each pixel), and resize it
    id2label = tf.constant([19, 19, 19, 19, 19, 19, 19, 0, 1, 19, 19, 2, 3, 
                            4, 19, 19, 19, 5, 19, 6, 7, 8, 9, 10, 11, 12, 
                            13, 14, 15, 19, 19, 16, 17, 18, 19], tf.int32)
    
    with tf.device('/cpu:0'):
        seg = tf.io.read_file(path)
        seg = tf.image.decode_png(seg, channels=1, dtype=tf.uint8)
        seg = tf.image.resize(seg, [IMG_SIZE, IMG_SIZE], method='nearest')  # 'nearest' to avoid creating new classes
        seg = tf.squeeze(seg)
        seg = tf.gather(id2label, tf.cast(seg, tf.int32))  # (image_size[0], image_size[1])
    return seg

def binarize_mask(mask):
    return tf.clip_by_value(mask, 0, 1)

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


def unstandardize_image(img):  # map pixel intensities to uint8 in [0, 255]
    scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
    img *= scale
    offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
    img += offset
    return img


# DATA AUGMENTATION
    
def data_aug(img, seg, seed=None):
    if tf.random.uniform([], minval=0, maxval=1, seed=seed) > 0.5:  # horizontal flip with probability 0.5
        img = tf.reverse(img, axis=[1])
        seg = tf.reverse(seg, axis=[1])
    if tf.random.uniform([], minval=0, maxval=1, seed=seed) > 1.0:  
        img = tf.image.random_brightness(img, max_delta=0.08, seed=seed)  # random brightness
        img = tf.image.random_contrast(img, lower=0.95, upper=1.05, seed=seed)  # random contrast
    return img, seg

def random_flip(img, seg, p=0.5, seed=None):
    """
    Randomly flip horizontally
    """
    if tf.random.uniform([], minval=0, maxval=1, seed=seed) < p:  # horizontal flip with probability 0.5
        img = tf.reverse(img, axis=[1])
        seg = tf.reverse(seg, axis=[1])
    return img, seg

def random_grayscale(img, seg, p=0.1, seed=None):
    """
    Randomly grayscale image
    """
    if tf.random.uniform([], minval=0, maxval=1, seed=seed) < p:  # grayscale with probability p
        img = tf.image.rgb_to_grayscale(img)
        img = tf.repeat(img, repeats=[3], axis=-1)
    return img, seg

def random_jitter(img, seg, p=1, r=0.05, seed=None):
    """
    Randomly change contrast and brightness
    """
    if tf.random.uniform([], minval=0, maxval=1, seed=seed) < p:  
        img = tf.image.random_brightness(img, max_delta=r, seed=seed)  # random brightness
        img = tf.image.random_contrast(img, lower=1-r, upper=1+r, seed=seed)  # random contrast
    return img, seg

def grayscale(x, y, p=0.3, seed=None):
    """
    Image to gray with 10% probability.
    """
    if tf.random.uniform(shape=[], minval=0., maxval=1., seed=seed) < p:
        gray_tensor = tf.image.rgb_to_grayscale(x)
        return tf.repeat(gray_tensor, repeats=3, axis=-1), y
    else:
        return x, y

def random_resize_crop(x, y, min_p=0.8, input_size=224, seed=None):
    """
    Random crop between min_p% and 100%.
    """
    stacked_image = tf.concat([x, y], axis=-1)
    perc = tf.math.floor(tf.random.uniform(shape=[], minval=min_p, maxval=1., dtype=tf.float32, seed=seed) * input_size)
    image_crops = tf.image.random_crop(stacked_image, [perc,perc,4], seed=seed)
    res_stacked_image = tf.image.resize(image_crops, [input_size,input_size])
    return res_stacked_image[...,:-1], tf.math.round(res_stacked_image[...,-1])

def zca_whitening(x, y, epsilon=1e-5):
    """
    Apply ZCA whitening to the input data.
    """
    x = x - tf.reduce_mean(x, axis=0)
    cov_matrix = tf.matmul(x, x, transpose_a=True) / x.shape[0]
    s, u, v = tf.linalg.svd(cov_matrix)
    s_inv = 1. / tf.sqrt(s + epsilon)
    whitening_matrix = tf.matmul(tf.matmul(u, tf.linalg.diag(s_inv)), u, transpose_b=True)
    return tf.matmul(x, whitening_matrix), y
    
# LOAD DATASET
    
def load_subdataset(root, config):
    #s = np.random.randint(0,255)
    print(root)
    img_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=root.joinpath('images'),
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode="rgb",
        batch_size=1, # cannot set None in TF 2.6!
        image_size=(config['IMG_SIZE'], config['IMG_SIZE']),
        shuffle=False,
        #seed=s,
        interpolation="bilinear",
        follow_links=False)
    
    if config['SUBSAMPLE'] and ('zucchini' in str(root)):
        img_ds = img_ds.take(math.ceil(0.25*len(img_ds)))
    
    if config['NORM'] == 'torch':
        img_ds = img_ds.map(lambda x: tf.keras.applications.imagenet_utils.preprocess_input(x, mode='torch'))

    mask_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=root.joinpath('masks'),
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode="grayscale",
        batch_size=1, # cannot set None in TF 2.6!
        image_size=(config['IMG_SIZE'], config['IMG_SIZE']),
        shuffle=False,
        #seed=s,
        interpolation="bilinear",
        follow_links=False)
        
    if config['SUBSAMPLE'] and ('zucchini' in str(root)):
        mask_ds = mask_ds.take(math.ceil(0.25*len(mask_ds)))
        
    #if 'tree' in str(root):
    mask_ds = mask_ds.map(binarize_mask)
    #else:
    #    mask_ds = mask_ds.map(normalize)

    return tf.data.Dataset.zip((img_ds, mask_ds))


def load_dataset(root, config):
    for f in sorted([root.joinpath(d) for d in os.listdir(root) if not d.startswith('.') and not d.endswith('.yaml')]):
        if 'ds' in locals():
            ds = ds.concatenate(load_subdataset(f, config))
        else:
            ds = load_subdataset(f, config)
    return ds


def load_multi_dataset(source_dataset, target_dataset, config):
    if source_dataset is None:
        return None, load_dataset(target_dataset, config)
    source_ds = []
    for crop in source_dataset:
        source_ds.append(load_dataset(crop, config))
    try:
        return source_ds, load_dataset(target_dataset, config)
    except:
        return source_ds, None
    
    
def split_data(ds_source, ds_target, config):
    if ds_target:
        d_len = len(ds_target)
        ds_test = ds_target.unbatch()
        ds_test = ds_test.apply(tf.data.experimental.assert_cardinality(d_len))
        if ds_source is None:
            return None, None, ds_test
            
        ds_train, ds_val = [], []
        len_tra, len_val = 0, 0
        for d in ds_source:
            d_len = len(d)
            val_len = int(math.floor(d_len * config['SPLIT_SIZE']))
            train_len = d_len - val_len
            len_tra += train_len
            len_val += val_len
            d = d.unbatch()
            d = d.shuffle(d_len, seed=config['SEED'] if config['SEED'] else None)
            
            ds_val.append(d.take(val_len))
            ds_train.append(d.skip(val_len))
    
        ds_train = tf.data.Dataset.from_tensor_slices(ds_train)
        ds_train = ds_train.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.apply(tf.data.experimental.assert_cardinality(len_tra))

        ds_val = tf.data.Dataset.from_tensor_slices(ds_val)
        ds_val = ds_val.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = ds_val.apply(tf.data.experimental.assert_cardinality(len_val))
        
    else:
        ds_train, ds_val, ds_test = [], [], []
        len_tra, len_tes, len_val = 0, 0, 0
        
        for d in ds_source:
            ds_len = len(d)
            test_len = math.floor(ds_len * config['SPLIT_SIZE'])
            val_len = math.floor(ds_len * config['SPLIT_SIZE'] * (1 - config['SPLIT_SIZE']))
            train_len = ds_len - test_len - val_len
            len_tra += train_len
            len_tes += test_len
            len_val += val_len
            
            d = d.unbatch()
            d = d.shuffle(ds_len, seed=config['SEED'] if config['SEED'] else None)
            
            ds_test.append(d.take(test_len))
            d_train = d.skip(test_len)
            ds_val.append(d_train.take(val_len))
            ds_train.append(d_train.skip(val_len))
            
        ds_train = tf.data.Dataset.from_tensor_slices(ds_train)
        ds_train = ds_train.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.apply(tf.data.experimental.assert_cardinality(len_tra))

        ds_val = tf.data.Dataset.from_tensor_slices(ds_val)
        ds_val = ds_val.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
        ds_val = ds_val.apply(tf.data.experimental.assert_cardinality(len_val))
        
        ds_test = tf.data.Dataset.from_tensor_slices(ds_test)
        ds_test = ds_test.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.apply(tf.data.experimental.assert_cardinality(len_tes))

    return ds_train, ds_val, ds_test