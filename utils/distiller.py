import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import glob
import math
import random
import argparse
from pathlib import Path
from contextlib import redirect_stdout
import time, datetime
import gc

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm

import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
import optuna 

from utils.data import load_multi_dataset, split_data
from utils.tools import read_yaml, save_log, get_args, ValCallback, TBCallback
from utils.data import random_resize_crop, random_jitter, random_flip, data_aug, normalize_imagenet, random_grayscale
from utils.training_tools import mIoU, loss_IoU, DiceBCELoss, ContrastiveLoss
from utils.models import build_model_multi, build_model_binary
from utils.cityscapes_utils import CityscapesDataset
from utils.mobilenet_v3 import MobileNetV3Large 
from utils.lovasz_loss import lovasz_hinge

from utils.train import Trainer



class Distiller(Trainer):
    
    def __init__(self, config, logger, strategy=None, trial=None, test=False):

        self.config = config
        self.logger = logger
        self.strategy = strategy
        self.trial = trial

        # define paths and names
        self.model_name = f"{config['NAME']}_{config['TARGET']}_{config['METHOD']}"
        self.model_dir = Path(config['MODEL_PATH'])
        self.log_dir = Path(config['LOG_PATH'])
        self.data_dir = Path(config['DATA_PATH'])
        tb_name = f"{self.config['TARGET']}_{self.config['ID']}_{datetime.datetime.now().strftime('%m_%d_%H_%M')}"
        self.tb_dir = self.log_dir.joinpath("tb").joinpath(tb_name)

        self.seed=self.config['SEED'] if self.config['SEED'] else None

        self.model_file = self.model_dir.joinpath(f"{self.model_name}_{self.config['ID']}.h5")
        self.log_file = self.log_dir.joinpath(f"{self.model_name}.txt")
        save_log(config, self.log_file)

        self.get_data(test_only=test)
        self.get_student()
        if not test:
            self.get_teacher()
        self.get_optimizer()
        self.get_callbacks()        

        self.kd_loss_fn = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        self.cov_matrix_layer = None
    
        print('Distiller Created')
    
    def get_student(self):    
        self.model = self.get_single_model(whiten=True)
        
        
    def get_single_model(self, weights=None, feats=True, whiten=False):
        
        whiten_layers = self.config['WHITEN_LAYERS'] if whiten \
                        and self.config['UNISTYLE'] \
                        and self.config['METHOD'] in ['KD'] else []
            
        if self.strategy:
            # load pretrained model
            with self.strategy.scope():

                backbone = MobileNetV3Large(input_shape=(self.config['IMG_SIZE'], self.config['IMG_SIZE'], 3),
                                            alpha=1.0,
                                            minimalistic=False,
                                            include_top=False,
                                            weights='imagenet',
                                            input_tensor=None,
                                            classes=self.config['N_CLASSES'],
                                            pooling='avg',
                                            dropout_rate=False,
                                            mode=self.config['METHOD'], p=self.config['PADAIN']['P'],
                                            eps=float(self.config['PADAIN']['EPS']),
                                            whiten_layers=whiten_layers,
                                            backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, 
                                            utils=tf.keras.utils)

                if self.config['CITYSCAPES']:
                    pre_trained_model = build_model_multi(backbone, False, 20)
                    pre_trained_model.load_weights(self.model_dir.joinpath('lr_aspp_pretrain_cityscapes.h5'))
                else:
                    pre_trained_model = backbone

                # binary segmentation model
                model = build_model_binary(pre_trained_model, False, self.config['N_CLASSES'], 
                                           sigmoid=self.config['LOSS']=='iou', mode=self.config['METHOD'],
                                           p=self.config['PADAIN']['P'], eps=float(self.config['PADAIN']['EPS']),
                                           return_feats=feats)
                
        else:
            backbone = MobileNetV3Large(input_shape=(self.config['IMG_SIZE'], self.config['IMG_SIZE'], 3),
                                        alpha=1.0,
                                        minimalistic=False,
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        classes=self.config['N_CLASSES'],
                                        pooling='avg',
                                        dropout_rate=False,
                                        mode=self.config['METHOD'], p=self.config['PADAIN']['P'],
                                        eps=float(self.config['PADAIN']['EPS']),
                                        whiten_layers=whiten_layers,
                                        backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, 
                                        utils=tf.keras.utils)

            if self.config['CITYSCAPES']:
                pre_trained_model = build_model_multi(backbone, False, 20)
                pre_trained_model.load_weights(self.model_dir.joinpath('lr_aspp_pretrain_cityscapes.h5'))
            else:
                pre_trained_model = backbone

            # binary segmentation model
            model = build_model_binary(pre_trained_model, False, self.config['N_CLASSES'], 
                                       sigmoid=self.config['LOSS']=='iou', mode=self.config['METHOD'],
                                       p=self.config['PADAIN']['P'], eps=float(self.config['PADAIN']['EPS']),
                                       return_feats=feats)
            
            if weights:
                model.load_weights(self.model_dir.joinpath(weights))
            model.trainable = True
            
            del pre_trained_model
            del backbone
            gc.collect()
            
            return model
    
    
    def get_teacher(self):
        domains = [w for w in self.config['SOURCE'] if w != self.config['TARGET']]
        weights = [f'teachers/teacher_aug_{w}.h5' for w in domains]
        print(f'Loaded Teachers: {domains}')
        
        models = [self.get_single_model(w, feats=False) for w in weights]
        
        model_input = tf.keras.Input(shape=(self.config['IMG_SIZE'], self.config['IMG_SIZE'], 3))
        model_outputs = [model(model_input) for model in models]
        ensemble_output = tf.keras.layers.Average()(model_outputs)
        self.teacher = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
        
        del models
        gc.collect()
   
    
    @tf.function    
    def train_step(self, x, y, cov_matrix_layer=None):
        
        with tf.GradientTape() as tape:
            
            aux_loss, kd_loss = 0.0, 0.0
            
            pred, feat = self.model(x, training=True)
            out_loss = self.compute_loss(y, pred)

            metr = self.compute_metric(y, pred)

            if self.config['AUX_LOSS']:
                _, feat_b = self.model(x, training=False)
                aux_loss = self.aux_loss(feat_b, feat)

            elif self.config['METHOD'] in ['KD']: 
                pred_t = self.teacher(x, training=False)  
                pred_t = tf.reshape(pred_t,(self.config['BATCH_SIZE'], -1))
                pred = tf.reshape(pred,(self.config['BATCH_SIZE'], -1))
                
                aux_loss = self.kd_loss_fn(tf.nn.softmax(pred_t / self.config['KD']['T'], axis=-1),
                                           tf.nn.softmax(pred / self.config['KD']['T'], axis=-1)) * self.config['KD']['T'] ** 2
            loss = out_loss + self.config['KD']['ALPHA'] * tf.reduce_mean(aux_loss)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))

        return out_loss, self.config['KD']['ALPHA'] * aux_loss, metr, None