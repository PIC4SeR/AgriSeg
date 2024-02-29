import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import time

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

from utils.tools import read_yaml, get_args, Logger
from utils.train import Trainer
from utils.distiller import Distiller
from utils.hp_search import HPSearcher



def main():
    start_time = time.time()

    # define some variables and read config
    args = get_args()
    config = read_yaml(args.config)
    config['TARGET'] = args.target if args.target is not None else config['TARGET']
    config['NAME'] = args.name if args.name is not None else config['NAME']
    config['METHOD'] = args.method if args.method is not None else config['METHOD']
    config['KD']['ALPHA'] = args.alpha if args.alpha is not None else config['KD']['ALPHA']
    config['KD']['T'] = args.temperature if args.temperature is not None else config['KD']['T']
    config['WHITEN_LAYERS'] = args.whiten_layers if args.whiten_layers is not None else config['WHITEN_LAYERS']
    config['ID'] = args.id if args.id is not None else 0
    config['ERM_TEACHERS'] = True if args.erm_teacher else False
    
    #select the working GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.cuda], 'GPU')
    devices = []
    for g in config['GPU']:
        tf.config.experimental.set_memory_growth(gpus[g], True)
        devices.append(f'GPU:{g}')
    
    if len(config['GPU']) > 1:
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    else:
        strategy = None
    
    logger = Logger(f"{config['LOG_PATH']}_{config['NAME']}_{start_time}.txt")

    if config['SEED']:
        #seed_everything(config['SEED'])
        tf.keras.utils.set_random_seed(config['SEED'])  # sets seeds for base-python, numpy and tf
        tf.config.experimental.enable_op_determinism()
    if config['HP_SEARCH']:
        searcher = HPSearcher(config=config, logger=logger, strategy=strategy, trial=None)
        searcher.hp_search()

    elif config['METHOD'] in ['KD']:
        distiller = Distiller(config, logger, strategy)
        distiller.train()
        #distiller.test()        
        
    else:
        trainer = Trainer(config, logger, strategy)
        trainer.train()
        #trainer.test()
    
    print(f"--- {(time.time() - start_time):.1f} seconds ---")
    return

  
if __name__ == "__main__":
    main()