import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import datetime
import gc
from tqdm.notebook import tqdm as tqdm

import tensorflow as tf

from utils.tools import save_log
from utils.models import build_model_multi, build_model_binary
from utils.mobilenet_v3 import MobileNetV3Large 
from utils.training_tools import loss_filter, uniform_soup, greedy_soup, mIoU, normalize
from utils.train import Trainer



class Distiller(Trainer):
    
    def __init__(self, cfg, logger, strategy=None, trial=None, test=False):

        self.cfg = cfg
        self.logger = logger
        self.strategy = strategy
        self.trial = trial

        # define paths and names
        self.model_name = f"{cfg['NAME']}_{cfg['TARGET']}_{cfg['METHOD']}"
        self.model_dir = Path(cfg['MODEL_PATH'])
        self.log_dir = Path(cfg['LOG_PATH'])
        self.data_dir = Path(cfg['DATA_PATH'])
        tb_name = f"{self.cfg['TARGET']}_{self.cfg['ID']}_{datetime.datetime.now().strftime('%m_%d_%H_%M')}"
        self.tb_dir = self.log_dir.joinpath("tb").joinpath(tb_name)

        self.seed = self.cfg['SEED'] if self.cfg['SEED'] else None

        self.model_file = self.model_dir.joinpath(f"{self.model_name}_{self.cfg['ID']}.h5")
        self.log_file = self.log_dir.joinpath(f"{self.model_name}.txt")
        save_log(cfg, self.log_file)

        self.get_data(test_only=cfg['TEST'] or test)
        if not (cfg['TEST'] and test):
            self.get_teacher()
        else:
            self.model = None
        self.get_student()
        self.get_optimizer()
        self.get_callbacks()

        self.kd_loss_fn = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        self.cov_matrix_layer = None
    
        print('Distiller Created')
    
    def get_student(self):
        if self.model is not None:
            return
        self.model = self.get_single_model(whiten=True)
        
        
    def get_single_model(self, weights=None, feats=True, whiten=False):
        
        whiten_layers = self.cfg['WHITEN_LAYERS'] if whiten \
                        and self.cfg['UNISTYLE'] \
                        and self.cfg['METHOD'] in ['KD'] else []
            
        if self.strategy:
            # load pretrained model
            with self.strategy.scope():

                backbone = MobileNetV3Large(input_shape=(self.cfg['IMG_SIZE'], self.cfg['IMG_SIZE'], 3),
                                            alpha=1.0,
                                            minimalistic=False,
                                            include_top=False,
                                            weights='imagenet',
                                            input_tensor=None,
                                            classes=self.cfg['N_CLASSES'],
                                            pooling='avg',
                                            dropout_rate=False,
                                            include_preprocessing=self.cfg['NORM']=='tf',
                                            mode=self.cfg['METHOD'], p=self.cfg['PADAIN']['P'],
                                            eps=float(self.cfg['PADAIN']['EPS']),
                                            whiten_layers=whiten_layers,
                                            wcta=self.cfg['WCTA'] if feats or 'wcta' in self.cfg['TEACHERS'] else False,  
                                            backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, 
                                            utils=tf.keras.utils)

                if self.cfg['CITYSCAPES']:
                    pre_trained_model = build_model_multi(backbone, False, 20)
                    pre_trained_model.load_weights(self.model_dir.joinpath('lr_aspp_pretrain_cityscapes.h5'))
                else:
                    pre_trained_model = backbone

                # binary segmentation model
                model = build_model_binary(pre_trained_model, False, self.cfg['N_CLASSES'], 
                                           sigmoid=self.cfg['LOSS']=='iou', mode=self.cfg['METHOD'],
                                           p=self.cfg['PADAIN']['P'], eps=float(self.cfg['PADAIN']['EPS']),
                                           fwcta=self.cfg['FWCTA'] if feats or 'fwcta' in self.cfg['TEACHERS'] else False,
                                           return_feats=feats)
                
        else:
            backbone = MobileNetV3Large(input_shape=(self.cfg['IMG_SIZE'], self.cfg['IMG_SIZE'], 3),
                                        alpha=1.0,
                                        minimalistic=False,
                                        include_top=False,
                                        weights='imagenet',
                                        input_tensor=None,
                                        classes=self.cfg['N_CLASSES'],
                                        pooling='avg',
                                        dropout_rate=False,
                                        include_preprocessing=self.cfg['NORM']=='tf',
                                        mode=self.cfg['METHOD'], p=self.cfg['PADAIN']['P'],
                                        eps=float(self.cfg['PADAIN']['EPS']),
                                        whiten_layers=whiten_layers,
                                        wcta=self.cfg['WCTA'] if feats or 'wcta' in self.cfg['TEACHERS'] else False, 
                                        backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, 
                                        utils=tf.keras.utils
                                        )

            if self.cfg['CITYSCAPES']:
                pre_trained_model = build_model_multi(backbone, False, 20)
                pre_trained_model.load_weights(self.model_dir.joinpath('lr_aspp_pretrain_cityscapes.h5'))
            else:
                pre_trained_model = backbone
                
            if self.cfg['FREEZE_BACKBONE']:
                pre_trained_model.trainable = False

            # binary segmentation model
            model = build_model_binary(pre_trained_model, False, self.cfg['N_CLASSES'], 
                                       sigmoid=self.cfg['LOSS']=='iou', mode=self.cfg['METHOD'],
                                       p=self.cfg['PADAIN']['P'], eps=float(self.cfg['PADAIN']['EPS']),
                                       fwcta=self.cfg['FWCTA'] if feats or 'fwcta' in self.cfg['TEACHERS'] else False,
                                       return_feats=feats)
            
            if weights:
                model.load_weights(self.model_dir.joinpath(weights))
            
            del pre_trained_model
            del backbone
            gc.collect()
            
            return model
    
    
    def get_teacher(self):
        domains = [w for w in self.cfg['SOURCE'] if w != self.cfg['TARGET']]
        if self.cfg['ERM_TEACHER']:
            weights = [f'teachers/erm/teacher_{self.cfg["TARGET"]}.h5']
        else:
            weights = [f'teachers/{self.cfg["TEACHERS"]}/teacher_{w}.h5' for w in domains]
        print(f'Loaded Teachers: {domains}')
        
        models = [self.get_single_model(w, feats=False) for w in weights]
        
        if self.cfg['SOUP'] == 'uniform':
            # average teacher weights
            self.model = uniform_soup(self.get_single_model(feats=True), [self.model_dir.joinpath(w) for w in weights])
        elif self.cfg['SOUP'] == 'greedy':
            self.model = greedy_soup(self.get_single_model(feats=True), [self.model_dir.joinpath(w) for w in weights],
                                     self.ds_val, mIoU, update_greedy=True, verbose=False)
        else:
            self.model = None
            
        model_input = tf.keras.Input(shape=(self.cfg['IMG_SIZE'], self.cfg['IMG_SIZE'], 3))
        model_outputs = [model(model_input) for model in models]
        # ensemble_output = tf.keras.layers.Average()(model_outputs)
        # self.teacher = tf.keras.Model(inputs=model_input, outputs=ensemble_output)
        self.teacher = tf.keras.Model(inputs=model_input, outputs=model_outputs)
        
        del models
        gc.collect()
   
    
    @tf.function    
    def train_step(self, x, y, cov_matrix_layer=None):
        
        with tf.GradientTape() as tape:
            
            aux_loss, kd_loss = 0.0, 0.0
            
            pred, feat = self.model(x, training=True)
            out_loss = self.compute_loss(y, pred)

            metr = self.compute_metric(y, pred)

            if self.cfg['AUX_LOSS']:
                _, feat_b = self.model(x, training=False)
                aux_loss = self.aux_loss(feat_b, feat)

            elif self.cfg['METHOD'] in ['KD']: 
                pred_t = self.teacher(x, training=False)  
                if self.cfg['KD']['ENSEMBLE'] == 'mean':
                    if self.cfg['KD']['NORM'] and 'pre' in self.cfg['KD']['NORM']:
                        pred = normalize(pred, self.cfg['KD']['NORM'])
                        pred_t = normalize(pred_t, self.cfg['KD']['NORM'])
                    pred_t = tf.reduce_mean(pred_t, axis=0)
                elif self.cfg['KD']['ENSEMBLE'] == 'w_mean':
                    print(tf.reduce_min(pred_t), tf.reduce_max(pred_t), tf.reduce_mean(pred_t))
                    alpha = tf.exp(pred_t) / tf.reduce_sum(tf.exp(pred_t), axis=0)
                    pred_t = tf.reduce_sum(pred_t * alpha, axis=0)
                    print(tf.reduce_min(alpha), tf.reduce_max(alpha), tf.reduce_mean(alpha))
                    print(tf.reduce_min(pred_t), tf.reduce_max(pred_t), tf.reduce_mean(pred_t))
                
                if self.cfg['KD']['NORM'] and 'post' in self.cfg['KD']['NORM']:
                    pred = normalize(pred, self.cfg['KD']['NORM'])
                    pred_t = normalize(pred_t, self.cfg['KD']['NORM'])

                if self.cfg['KD']['LOSS'] == 'old': # old kld version (CWD)
                    pred_t = tf.reshape(pred_t,(self.cfg['BATCH_SIZE'], -1))
                    pred = tf.reshape(pred,(self.cfg['BATCH_SIZE'], -1))
                    aux_loss = self.kd_loss_fn(
                        tf.nn.softmax(pred_t / self.cfg['KD']['T'], axis=-1),
                        tf.nn.softmax(pred / self.cfg['KD']['T'], axis=-1)
                        ) * self.cfg['KD']['T'] ** 2
                elif self.cfg['KD']['LOSS'] == 'kld': # kld loss
                    # create additional class channel by difference
                    pred_t = tf.math.sigmoid(pred_t / self.cfg['KD']['T'])
                    pred_t_h = tf.concat([tf.ones_like(pred_t) - pred_t, pred_t], axis=-1)
                    pred = tf.math.sigmoid(pred / self.cfg['KD']['T'])
                    pred_h = tf.concat([tf.ones_like(pred) - pred, pred], axis=-1)
                    aux_loss = self.kd_loss_fn(pred_t_h, pred_h) * self.cfg['KD']['T'] ** 2
                    if self.cfg['KD']['FILTER'] == "error":
                        pred_t_bin = tf.math.greater(pred_t, tf.constant([0.5]))[...,0]
                        mask = tf.equal(pred_t_bin, tf.cast(y, tf.bool))
                        aux_loss = aux_loss * tf.cast(mask, tf.float32)
                        n = tf.math.count_nonzero(aux_loss)
                        aux_loss = tf.reduce_sum(aux_loss) / tf.cast(n, tf.float32) if n > 0 else 0.0
                    elif self.cfg['KD']['FILTER'] == "confidence":
                        w = loss_filter(pred_t)[...,0]
                        aux_loss = aux_loss * w
                    elif self.cfg['KD']['WEIGHT'] == "iou":
                        iou_t = mIoU(y, pred_t, reduce=False)
                        iou_s = mIoU(y, pred, reduce=False)
                        w = tf.sigmoid(iou_t / (iou_s + 1e-5) - 1) * 2
                        aux_loss = aux_loss * tf.cast(w, tf.float32)[:,None,None]
                        
                # elif self.cfg['KD']['LOSS'] == 'logsum':
                #     feature-based distillation
                # elif self.cfg['KD']['LOSS'] == 'mse':
                #     feature-based distillation
                # elif self.cfg['KD']['LOSS'] == 'mae':
                #     feature-based distillation
                
            loss = out_loss + self.cfg['KD']['ALPHA'] * tf.reduce_mean(aux_loss)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))

        return out_loss, self.cfg['KD']['ALPHA'] * tf.reduce_mean(aux_loss), metr, None