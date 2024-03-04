import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import datetime

import tensorflow as tf

from utils.tools import get_args_and_cfg, Logger
from utils.train import Trainer
from utils.distiller import Distiller
from utils.hp_search import HPSearcher



def main():
    start_time = datetime.datetime.now().strftime('%m_%d_%H_%M')

    # define some variables and read config
    args, config = get_args_and_cfg()

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
        searcher = HPSearcher(args=args, config=config, logger=logger, strategy=strategy, trial=None)
        searcher.hp_search()

    elif config['METHOD'] in ['KD']:
        distiller = Distiller(config, logger, strategy)
        distiller.train()
        #distiller.test()        
        
    else:
        trainer = Trainer(config, logger, strategy)
        trainer.train()
        #trainer.test()
    
    print(f"--- {(datetime.time() - start_time):.1f} seconds ---")
    return

  
if __name__ == "__main__":
    main()