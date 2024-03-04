import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from contextlib import redirect_stdout
import time, datetime
import gc
from tqdm.notebook import tqdm as tqdm

import tensorflow as tf
import tensorflow_addons as tfa
#import tensorflow_probability as tfp
import optuna 

from utils.data import load_multi_dataset, split_data
from utils.tools import save_log
from utils.data import random_resize_crop, random_jitter, random_flip, random_grayscale, zca_whitening
from utils.training_tools import mIoU, loss_IoU, ContrastiveLoss, mIoU_old
from utils.models import build_model_multi, build_model_binary
from utils.mobilenet_v3 import MobileNetV3Large 
from utils.instance_norm import CovMatrix_ISW, instance_whitening_loss
from utils.xded import pixelwise_XDEDLoss

class Trainer:  
    
    def __init__(self, config, logger, strategy=None, trial=None, test=False):

        self.config = config
        self.logger = logger
        self.strategy = strategy
        self.trial = trial
        self.seed=self.config['SEED'] if self.config['SEED'] else None
        
        # define paths and names
        self.model_name = f"{config['NAME']}_{config['TARGET']}_{config['METHOD']}"
        self.model_dir = Path(config['MODEL_PATH'])
        self.log_dir = Path(config['LOG_PATH'])
        self.data_dir = Path(config['DATA_PATH'])
        tb_name = f"{self.model_name}_{self.config['ID']}_{datetime.datetime.now().strftime('%m_%d_%H_%M')}"
        self.tb_dir = self.log_dir.joinpath("tb").joinpath(tb_name)

        self.model_file = self.model_dir.joinpath(f"{tb_name}.h5")
        self.log_file = self.log_dir.joinpath(f"{self.model_name}.txt")
        #save_log(config, self.log_file)

        self.get_data(test_only=test)
        
        if self.strategy:
            with self.strategy.scope():
                self.get_model() 
        else:
            self.get_model() 
            
        self.get_optimizer()
        self.get_callbacks()
    
        self.cov_matrix_layer = None
        if config['METHOD'] == 'ISW':
            self.whitening = True
            in_channel_list = [16, 16, 24] #[16, 24, 24]
            self.cov_matrix_layer = []
            for i, c in enumerate(in_channel_list):
                self.cov_matrix_layer.append(CovMatrix_ISW(dim=c, relax_denom=0, clusters=3))

    def set_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].set_mask_matrix()


    def reset_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].reset_mask_matrix()
    
    def log_results(self, epoch, logs, now):
        
        log = f"{epoch+1:03d} - "
        log += f"Train: {logs['loss']:.4f}, {logs['loss_aux']:.4f}, {logs['mIoU']:.4f} "
        log += f"Val: {logs['val_mIoU']:.4f} "
        log += f"Test: {logs['test_mIoU']:.4f} "
        log += f" ({int(time.perf_counter() - now)}s)"
        self.logger.save_log(log)
    
    
    
    def get_data(self, test_only=False):
        
        # load dataset
        target_dataset = self.data_dir.joinpath(self.config['TARGET'])
        
        if test_only:
            source_dataset = None
            
        elif not self.config['DG']:
            source_dataset = [target_dataset]
            target_dataset = None
            
        else:        
            source_dataset = sorted([self.data_dir.joinpath(d) 
                                     for d in self.config['SOURCE'] if d != self.config['TARGET']])
        
        with redirect_stdout(None):
            ds_source, ds_target = load_multi_dataset(source_dataset, target_dataset, self.config)

        self.ds_train, self.ds_val, self.ds_test = split_data(ds_source, ds_target, self.config)
        
        # build dataset
        if self.ds_train is not None:
            self.train_len = len(self.ds_train)
            self.ds_train = self.ds_train.cache()
            self.ds_train = self.ds_train.shuffle(self.train_len, seed=self.seed)
            self.ds_train = self.ds_train.map(lambda x, y: random_flip(x, y, p=self.config['RND_FLIP'], seed=self.seed),
                                              tf.data.experimental.AUTOTUNE)
            self.ds_train = self.ds_train.map(lambda x, y: random_resize_crop(x, y, min_p=self.config['RND_CROP'],
                                                                              seed=self.seed),
                                              tf.data.experimental.AUTOTUNE)
            if self.config['STYLE_AUG']:
                self.ds_train = self.ds_train.map(lambda x, y: random_jitter(x, y, p=self.config['RND_JITTER'], 
                                                                             r=self.config['RND_JITTER_RNG'], 
                                                                             seed=self.seed),
                                                  tf.data.experimental.AUTOTUNE)
                self.ds_train = self.ds_train.map(lambda x, y: random_grayscale(x, y, p=self.config['RND_GREY'],
                                                                                seed=self.seed),
                                                  tf.data.experimental.AUTOTUNE)
            if self.config['ZCA']:
                self.ds_train = self.ds_train.map(lambda x, y: zca_whitening(x, y), tf.data.experimental.AUTOTUNE)
            
            self.ds_train = self.ds_train.batch(self.config['BATCH_SIZE'], drop_remainder=True)
            self.ds_train = self.ds_train.prefetch(tf.data.experimental.AUTOTUNE)
            self.ds_train = self.strategy.experimental_distribute_dataset(self.ds_train) if self.strategy else self.ds_train
        else: 
            self.train_len = 0
            
        if self.ds_val is not None:
            self.val_len = len(self.ds_val)
            self.ds_val = self.ds_val.cache()
            self.ds_val = self.ds_val.batch(self.config['BATCH_SIZE'], drop_remainder=False)
            self.ds_val = self.ds_val.prefetch(tf.data.experimental.AUTOTUNE)
            self.ds_val = self.strategy.experimental_distribute_dataset(self.ds_val) if self.strategy else self.ds_val
        else: 
            self.val_len = 0

        if self.ds_test is not None:
            self.test_len = len(self.ds_test)
            self.ds_test = self.ds_test.cache()
            # self.ds_test = self.ds_test.shuffle(self.test_len, seed=self.seed)
            self.ds_test = self.ds_test.batch(self.config['BATCH_SIZE'], drop_remainder=False)
            self.ds_test = self.ds_test.prefetch(tf.data.experimental.AUTOTUNE)
            self.ds_test = self.strategy.experimental_distribute_dataset(self.ds_test) if self.strategy else self.ds_test
        else: 
            self.test_len = 0
            
        print(f'Loaded data: Train {self.train_len}, Val {self.val_len}, Test {self.test_len}')

        del ds_source
        del ds_target
        gc.collect()
       
    
    
    def get_model(self):    
        if self.config['UNISTYLE'] and self.config['METHOD'] in ['ISW','XDED','IN','KD']:
            whiten_layers = self.config['WHITEN_LAYERS'] 
        else:
            whiten_layers = [],
        # load pretrained model
        backbone = MobileNetV3Large(input_shape=(self.config['IMG_SIZE'], self.config['IMG_SIZE'], 3),
                                    alpha=1.0,
                                    minimalistic=False,
                                    include_top=False,
                                    weights='imagenet',
                                    input_tensor=None,
                                    classes=self.config['N_CLASSES'],
                                    pooling='avg',
                                    dropout_rate=False,
                                    include_preprocessing=self.config['NORM']=='tf',
                                    mode=self.config['METHOD'], 
                                    p=self.config['PADAIN']['P'],
                                    eps=float(self.config['PADAIN']['EPS']),
                                    whiten_layers=whiten_layers,
                                    wcta=self.config['WCTA'],
                                    backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, 
                                    utils=tf.keras.utils)

        if self.config['CITYSCAPES']:
            pre_trained_model = build_model_multi(backbone, False, 20)
            pre_trained_model.load_weights(self.model_dir.joinpath('lr_aspp_pretrain_cityscapes.h5'))
        else:
            pre_trained_model = backbone
            
        if self.config['FREEZE_BACKBONE']:
            pre_trained_model.trainable = False
        
        # binary segmentation model
        self.model = build_model_binary(base_model=pre_trained_model, 
                                        dropout_rate=False, 
                                        n_class=self.config['N_CLASSES'], 
                                        sigmoid=self.config['LOSS']=='iou', 
                                        mode=self.config['METHOD'],
                                        p=self.config['PADAIN']['P'], 
                                        fwcta=self.config['FWCTA'],
                                        eps=float(self.config['PADAIN']['EPS']))
        
        del pre_trained_model
        del backbone
        gc.collect()
            
    def get_optimizer(self):

        self.n_steps = self.config['N_EPOCHS'] * (self.train_len) // self.config['BATCH_SIZE']
        
        if self.strategy:
            with self.strategy.scope():
                self.get_opt_loss()
        else:
            self.get_opt_loss()
            
        self.metric = mIoU if self.config['METRIC'] == 'iou' else mIoU_old

        self.train_loss_mean, self.train_metr_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        self.train_aux_mean = tf.keras.metrics.Mean()
        self.val_loss_mean, self.val_metr_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        self.test_loss_mean, self.test_metr_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
        
        self.model.compile(self.optim, self.loss, self.metric)
        
        
    def get_opt_loss(self):
        
        if self.config['OPTIMIZER'] == 'adamw':
            lr_sched = tf.keras.optimizers.schedules.PolynomialDecay(
                           initial_learning_rate=float(self.config['ADAMW']['LR']), 
                           decay_steps=self.n_steps, 
                           end_learning_rate=float(self.config['ADAMW']['LR_END']), 
                           power=self.config['ADAMW']['DECAY'])
            
            self.optim = tfa.optimizers.AdamW(learning_rate=lr_sched, weight_decay=float(self.config['ADAMW']['WD']))
            
        elif self.config['OPTIMIZER'] == 'sgd':
            lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
                           float(self.config['SGD']['LR']),
                           decay_steps=self.n_steps,
                           decay_rate=self.config['SGD']['DECAY'],
                           staircase=False)
            
            self.optim = tf.keras.optimizers.SGD(learning_rate=lr_sched, momentum=self.config['SGD']['MOMENTUM'],
                                                 nesterov=self.config['SGD']['NESTEROV'])
            
        elif self.config['OPTIMIZER'] == 'adam':
            self.optim = tf.keras.optimizers.Adam(learning_rate=float(self.config['ADAM']['LR']))

            
        if self.config['LOSS'] == 'bce':
            self.loss = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            #self.loss = binary_weighted_cross_entropy(from_logits=True)
            
        elif self.config['LOSS'] == 'iou':
            self.loss = loss_IoU

            
        if self.config['AUX_LOSS']:
            self.aux_loss = ContrastiveLoss(self.config['BATCH_SIZE']//2, weight=self.config['CL']['WEIGHT'], 
                                            temperature=self.config['CL']['TEMP'])
            
        if self.config['METHOD'] == 'XDED':
            self.xded = pixelwise_XDEDLoss(temp_factor=self.config['KD']['T'])

        
    def get_callbacks(self):
        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tb_dir, histogram_freq=1)
        self.tb_callback.set_model(self.model)
        self.test_writer = tf.summary.create_file_writer(str(self.tb_dir.joinpath('test')))

        
        
    def train(self):
        if self.ds_train is None:
            print('Cannot train without training dataset!')
            return None
        
        best_metr , best_test_metr, btm = 0, 0, 0
        now = time.perf_counter()
        
        for epoch in range(self.config['N_EPOCHS']):
            for step, (x, y) in enumerate(self.ds_train, 1):
                self.step = step
                if self.strategy is None:
                    train_loss, train_aux, train_metr, self.cov_matrix_layer = self.train_step(x, y, self.cov_matrix_layer)
                else:
                    train_loss, train_aux, train_metr, self.cov_matrix_layer = self.distributed_train_step(x, y, self.cov_matrix_layer)
                
                self.train_loss_mean(tf.reduce_sum(train_loss))
                self.train_aux_mean(tf.reduce_sum(train_aux))
                self.train_metr_mean(tf.reduce_mean(train_metr))
            
            train_loss = self.train_loss_mean.result()
            train_aux = self.train_aux_mean.result()
            train_metr = self.train_metr_mean.result()
            self.train_loss_mean.reset_states()
            self.train_aux_mean.reset_states()
            self.train_metr_mean.reset_states()

            val_loss, val_metr = self.evaluate(self.ds_val, split='val')
#             val_loss = self.val_loss_mean.result()
#             val_metr = self.val_metr_mean.result()
#             self.val_loss_mean.reset_states()
#             self.val_metr_mean.reset_states()
                
            test_loss, test_metr = self.evaluate(self.ds_test, split='test')
#             test_loss = self.test_loss_mean.result()
#             test_metr = self.test_metr_mean.result()
#             self.test_loss_mean.reset_states()
#             self.test_metr_mean.reset_states()
        
            if val_metr > best_metr:
                best_metr = val_metr
                best_test_metr = test_metr
                best_metr_epoch = epoch
                self.model.save_weights(self.model_file)
                
            if test_metr > btm:
                btm = test_metr
                btv = val_metr
                bte = epoch

            logs = {"learning_rate": self.optim._decayed_lr('float32'),
                    "loss": train_loss + train_aux,
                    "mIoU": train_metr,
                    "val_loss": val_loss,
                    "val_mIoU": val_metr}

            self.tb_callback.on_epoch_end(epoch, logs)
            with self.test_writer.as_default():
                tf.summary.scalar("epoch_loss", test_loss, step=epoch) 
                tf.summary.scalar("epoch_mIoU", test_metr, step=epoch)           
                
            logs["loss_aux"] = train_aux
            logs["loss"] = train_loss
            logs["test_loss"] = test_loss
            logs["test_mIoU"] = test_metr
            self.log_results(epoch, logs, now)

            
                    
            if self.trial is not None:
                self.trial.report(test_metr, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        
            now = time.perf_counter()
        
        out = f'Best Val {best_metr:.4f} Test {best_test_metr:.4f} ({best_metr_epoch}) | '
        out += f'Val {btv:.4f} Best Test {btm:.4f} ({bte})'
        self.logger.save_log(out)
        
        return best_test_metr if self.config['SAVE_BEST'] else test_metr
        
        
        
    @tf.function    
    def train_step(self, x, y, cov_matrix_layer=None):
        
        with tf.GradientTape() as tape:

            pred, *feat = self.model(x, training=True)
            #print(y, pred)
            
            aux_loss = tf.zeros((self.config['BATCH_SIZE'],))
            
            if self.config['AUX_LOSS']:
                _, feat_b = self.model(x, training=False)
                aux_loss = self.aux_loss(feat_b, feat)
                
            if self.config['METHOD'] == 'XDED':
                aux_loss = self.xded(pred, y) * self.config['KD']['ALPHA']
               
            elif self.config['METHOD'] == 'ISW':
                
                feat_array = tf.TensorArray(tf.float32, len(feat), infer_shape=False, clear_after_read=False)
                for i, f in enumerate(feat):
                    feat_array = feat_array.write(i,f)
                for index in range(len(self.cov_matrix_layer)):
                    if self.step < 2:
                        # Instance Whitening
                        sh = tf.shape(feat_array.read(index))  # i-th feature size (B X C X H X W)
                        B, H, W, C = sh[0], sh[1], sh[2], sh[3]
                        HW = H * W
                        f_map = tf.reshape(tf.experimental.numpy.ascontiguousarray(feat_array.read(index)),[B, -1, C]) 
                        eye, reverse_eye = self.cov_matrix_layer[int(index)].get_eye_matrix()
                        f_cor = tf.linalg.matmul(tf.transpose(f_map,[0,2,1]),f_map) 
                        #print(f_cor.shape, eye.shape)
                        f_cor = f_cor / tf.cast((HW-1), tf.float32) + (1e-5 * tf.cast(eye, tf.float32))  
                        
                        off_diag_elements = f_cor * reverse_eye
                        v = tf.math.reduce_variance(off_diag_elements, axis=0)
                        cov_matrix_layer[index].set_variance_of_covariance(v)
                        
                    else:
                        eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[index].get_mask_matrix()
                        loss = instance_whitening_loss(feat_array.read(index), eye, mask_matrix, margin, num_remove_cov)
                        aux_loss = aux_loss + loss * self.config['KD']['ALPHA']
                        
                aux_loss = aux_loss / tf.cast(len(cov_matrix_layer), tf.float32)

            out_loss = self.compute_loss(y, pred[...,0])
            loss = out_loss + aux_loss
        
            metr = self.compute_metric(y, pred)
        
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))

        return out_loss, aux_loss, metr, cov_matrix_layer
        
        
    @tf.function
    def distributed_train_step(self, x, y, cov_matrix_layer=None):
        per_replica_losses, per_replica_aux, per_replica_metr, cov_matrix_layer = self.strategy.run(self.train_step, args=(x, y, cov_matrix_layer))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        aux = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_aux, axis=None)
        metr = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_metr, axis=None)
        return tf.reduce_mean(loss), tf.reduce_mean(aux), tf.reduce_mean(metr), cov_matrix_layer
        
        
    @tf.function        
    def compute_loss(self, labels, predictions, model_losses=None):
        loss = self.loss(labels, predictions)
        loss = tf.reduce_mean(loss)
        #loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.config['BATCH_SIZE'])
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss
    
    
    @tf.function        
    def compute_metric(self, labels, predictions):
        per_example_metr = self.metric(labels, predictions)
        #metr = tf.nn.compute_average_loss(per_example_metr, global_batch_size=self.config['BATCH_SIZE'])
        return per_example_metr
    
        
    def test(self):
        
        self.model.trainable = False

        # test the model
#         with self.strategy.scope():
#             self.model.load_weights(self.model_file)
        
#         val_loss, val_miou = self.model.evaluate(self.ds_val, steps=self.val_len//self.config['BATCH_SIZE'],
#                                                  workers=8, use_multiprocessing=True, verbose=2)
        test_loss, test_miou = self.model.evaluate(self.ds_test, steps=self.test_len//self.config['BATCH_SIZE'],
                                                   workers=24, use_multiprocessing=True, verbose=2)
#         save_log(f"Val Loss and mIoU: {val_loss} {val_miou}", self.log_file)
        save_log(f"Test Loss and mIoU: {test_loss} {test_miou}", self.log_file)


    def evaluate(self, dataset, split):
        i, losses, metrics = 0, 0, 0
        for x, y in dataset:
            if self.strategy is None:
                loss, metric = self.evaluate_step(x, y)
            else:
                per_replica_losses, per_replica_metr = self.strategy.run(self.evaluate_step, args=(x, y))
                loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                metric = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_metr, axis=None)
            losses += loss
            metrics += metric
            if split == 'val':
                self.val_loss_mean(loss)
                self.val_metr_mean(metric)
            elif split == 'test':
                self.test_loss_mean(loss)
                self.test_metr_mean(metric)
            i += 1
        return losses/i, metrics/i
    
    
    @tf.function
    def evaluate_step(self, x, y):
        pred, *_ = self.model(x, training=False)
        #pred = tf.math.sigmoid(pred)
        out_loss = self.compute_loss(y, pred)
        # aux_loss = self.aux_loss(y, pred)
        loss = out_loss # + self.config[self.config['MODE']]['ALPHA'] * aux_loss
        
        metr = self.compute_metric(y, pred)
        
        return tf.reduce_mean(loss), tf.reduce_mean(metr)
    
    
    @tf.function
    def resolve(self, x):
        y = self.model(x, training=False)
        return y