import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import datetime, time

import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

from utils.tools import Logger, get_args_and_cfg
from utils.train import Trainer
from utils.distiller import Distiller
from utils.hp_search import HPSearcher



def main():
    start_time = time.time()

    # define some variables and read config
    args, config = get_args_and_cfg()
    #select the working GPU
    if config['NAME'] == 'test' or config['METHOD'] == 'ISW':
        tf.config.run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.cuda], 'GPU')
    devices = []
    for g in [args.cuda]:
        tf.config.experimental.set_memory_growth(gpus[g], True)
        devices.append(f'GPU:{g}')
    
    if len(config['GPU']) > 1:
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    else:
        strategy = None
    
    logger = Logger(f"{config['LOG_PATH']}_{config['NAME']}_{start_time}.txt")
    start_time = time.time()
    
    if config['SEED']:
        #seed_everything(config['SEED'])
        tf.keras.utils.set_random_seed(config['SEED'])  # sets seeds for base-python, numpy and tf
        tf.config.experimental.enable_op_determinism()
    if config['HP_SEARCH']:
        searcher = HPSearcher(args=args, config=config, logger=logger, strategy=strategy, trial=None)
        searcher.hp_search()

    elif config['METHOD'] in ['KD']:
        distiller = Distiller(config, logger, strategy)
        if config['TEST']:
            test_loss, test_metr = distiller.evaluate(trainer.ds_test, "test")
            print(f"Test loss: {test_loss}, Test mIoU: {test_metr}")
        else:
            distiller.train()
        
    else:
        trainer = Trainer(config, logger, strategy)
        if config['TEST']:
            test_loss, test_metr = trainer.evaluate(trainer.ds_test, "test")
            print(f"Test loss: {test_loss}, Test mIoU: {test_metr}")
        else:
            trainer.train()
    
    print(f"--- {(time.time() - start_time):.1f} seconds ---")
    return

  
if __name__ == "__main__":
    main()