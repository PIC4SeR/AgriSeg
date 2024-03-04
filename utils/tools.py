import argparse
import yaml
import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm.notebook import tqdm
import glob
from utils.preprocess import load_and_resize_image, load_and_resize_segmentation
import pprint


class Logger(object):
    
    def __init__(self, file):
        
        self.file = file
        self.pp = pprint.PrettyPrinter(depth=2)
        
    def save_log(self, text):
        
        if type(text) is dict:
            text = self.pp.pformat(text)
            
        print(text)
        with open(self.file, 'a') as f:
            f.write(text + '\n')
            

def read_yaml(path):
    stream = open(path, 'r')
    dictionary = yaml.safe_load(stream)
    return dictionary


def custom_shuffle(img_array, mask_array):
    assert len(img_array) == len(mask_array)
    p = np.random.permutation(len(img_array))
    return img_array[p], mask_array[p]


def loadData2(images, maps):
    X = []
    y = []
    for i, m in zip(images, maps):         
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(m, 0)
        X.append(img)
        y.append(mask)
    #return (np.array(X), np.array(y))
    return X, y


def loadData(imgList, maskList, rgb_path, mask_path):
    X = []
    y = []
    for i in range(0,len(imgList)):
        img_name = os.path.join(rgb_path,imgList[i])
        mask_name = os.path.join(mask_path,maskList[i])           
        img = cv2.imread(img_name,cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name,0)
        X.append(img)
        y.append(mask)
    return (np.array(X), np.array(y))


def loadCityS(path_img, path_gt, img_size):
    X = []
    y = []
    for i in tqdm(range(len(path_img))):
        X.append(load_and_resize_image(path_img[i], img_size))
        y.append(load_and_resize_segmentation(path_gt[i], img_size))         
    return np.array(X), np.array(y)
    

def loadTestData():
    X = []
    y = []
    img_name = sorted(glob.glob(r'dataset_vineyards/test/vineyard/img_*[!_refined_mask].jpg'))
    mask_name = []
    for i in img_name:
        i = i.split('.jpg')
        mask_name.append(i[0] + '_refined_mask.jpg')
    for i in range(0, len(img_name)):           
        img = cv2.imread(img_name[i],cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name[i],0)
        X.append(img)
        y.append(mask)
    return (np.array(X), np.array(y))


def predict_model(model, img):
    if len(img.shape) == 3:
        img = img[None]
    pred = model.predict(img)
    return np.argmax(pred, axis=-1)


def predict_model_binary(model, img, thresh):
    if len(img.shape) == 3:
        img = img[None]
    pred = model.predict(img)
    return pred > thresh


def pred_to_color(img):
    class_colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), 
                    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), 
                    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), 
                    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), 
                    (0, 0, 230), (119, 11, 32), (0, 0, 0)]
    img = tf.gather(class_colors, tf.cast(img, tf.int32))  
    return img


def compute_intersection_and_union_in_batch(y_true_labels, y_pred_labels, num_classes):
    # y_true_labels: (val_batch_size, img_h, img_w)
    # y_pred_labels: (val_batch_size, img_h, img_w)
    batch_intersection, batch_union = [], []  # for each class, store the sum of intersections and unions in the batch
    for class_label in range(num_classes - 1):  # ignore class 'other'
        true_equal_class = tf.cast(tf.equal(y_true_labels, class_label), tf.int32)
        pred_equal_class = tf.cast(tf.equal(y_pred_labels, class_label), tf.int32)
        intersection = tf.reduce_sum(tf.multiply(true_equal_class, pred_equal_class))  # TP (true positives)
        union = tf.reduce_sum(true_equal_class) + tf.reduce_sum(pred_equal_class) - intersection  
        # TP + FP + FN = (TP + FP) + (TP + FN) - TP
        batch_intersection.append(intersection)
        batch_union.append(union)
    return tf.cast(tf.stack(batch_intersection, axis=0), tf.int64), tf.cast(tf.stack(batch_union, axis=0), tf.int64)  # (19,)


def evaluate(ds_x, ds_y, model):
    # Compute IoU on validation set (IoU = Intersection / Union)
    total_intersection = tf.zeros((19), tf.int64)
    total_union = tf.zeros((19), tf.int64)
    print(f'Evaluating on validation set: {len(ds_x)}')
    for x, y_true_labels in tqdm(zip(ds_x, ds_y)):
        y_pred_labels = predict_model(model, x)
        batch_intersection, batch_union = compute_intersection_and_union_in_batch(y_true_labels, y_pred_labels, num_classes=20)
        total_intersection += batch_intersection
        total_union += batch_union
    iou_per_class = tf.divide(total_intersection, total_union)  # IoU for each of the 19 classes
    iou_mean = tf.reduce_mean(iou_per_class)  # Mean IoU over the 19 classes
    return iou_per_class, iou_mean


def save_log(text, file):
    if isinstance(text, dict):
        pprint.pprint(text)
        with open(file, 'a') as f:
            pprint.pprint(text, f)       
    else:
        print(text)
        with open(file, 'a') as f:
            f.write(str(text)+'\n')

            
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Utils
    parser.add_argument('--config', default='utils/config.yaml', help="Configuration file")
    parser.add_argument('--name', default=None, help="Model name")
    parser.add_argument('--id', default=None, help="Model id")
    parser.add_argument("--cuda", default=0, type=int, help="Select cuda device")
    # Dataset
    parser.add_argument("--target", default=None, help="Target domain")
    # DG Methodology
    parser.add_argument("--method", default=None, help="DG methodology")
    parser.add_argument("--alpha", type=float, default=None, help="Auxiliary loss weight")
    parser.add_argument("--temperature", type=float, default=None, help="Auxiliary loss temperature")
    parser.add_argument("--whiten_layers", type=list, default=None, help="Layers to whiten")
    parser.add_argument("--erm_teacher", default=False, help="ERM teachers", action="store_true")
    parser.add_argument("--weights", default=None, help="Weights to be tested")
    parser.add_argument("--test", default=False, help="Test only", action="store_true")
    return parser.parse_args()


def get_cfg(args):
    config = read_yaml(args.config)
    config['TARGET'] = args.target if args.target is not None else config['TARGET']
    config['NAME'] = args.name if args.name is not None else config['NAME']
    config['METHOD'] = args.method if args.method is not None else config['METHOD']
    config['KD']['ALPHA'] = args.alpha if args.alpha is not None else config['KD']['ALPHA']
    config['KD']['T'] = args.temperature if args.temperature is not None else config['KD']['T']
    config['WHITEN_LAYERS'] = args.whiten_layers if args.whiten_layers is not None else config['WHITEN_LAYERS']
    config['ID'] = args.id if args.id is not None else 0
    config['ERM_TEACHERS'] = True if args.erm_teacher else False
    config['WEIGHTS'] = args.weights if args.weights is not None else None
    config['TEST'] = True if args.test else False
    config['TEACHERS'] = f"{config['NORM']}_{'style' if config['STYLE_AUG'] else 'geom'}" + \
                         f"{'_wcta' if config['WCTA'] else ''}" if config['TEACHERS'] is None else config['TEACHERS']
    return config

def get_args_and_cfg():
    args = get_args()
    config = get_cfg(args)
    return args, config
    


class ValCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds_val, log_dir):
        self.ds_val = ds_val
        self.test_summary_writer = tf.summary.create_file_writer(str(log_dir))
        self.best_val = 0.0
        self.best_epoch = 0

    def on_train_end(self, logs=None):
        print(f"Best test {self.best_val}, Epoch {self.best_epoch}")
        
    def on_epoch_end(self, epoch, logs={}):
        loss, metric = self.model.evaluate(self.ds_val, verbose=2, workers=8, use_multiprocessing=True)
        if metric > self.best_val:
            self.best_val = metric
            self.best_epoch = epoch
        with self.test_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', loss, step=epoch)
            tf.summary.scalar('epoch_mIoU', metric, step=epoch)
            
            
class TBCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, tb_callback):
        self.writer = tb_callback.writer

    def on_epoch_end(self, epoch, logs=None):

        for name, value in logs.items():
            summary = tf.summary.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
            self.writer.flush()
            
def seed_everything(seed: int):
    tf.random.set_seed(seed)  # set random seed for keras, numpy, tensorflow, and the 'random' module
    os.environ['PYTHONHASHSEED'] = str(seed)