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

    # define some variables and read cfg
    args, cfg = get_args_and_cfg()
    #select the working GPU
    if cfg['NAME'] == 'test' or cfg['METHOD'] == 'ISW':
        tf.cfg.run_functions_eagerly(True)
    gpus = tf.cfg.experimental.list_physical_devices('GPU')
    tf.cfg.experimental.set_visible_devices(gpus[args.cuda], 'GPU')
    devices = []
    for g in [args.cuda]:
        tf.cfg.experimental.set_memory_growth(gpus[g], True)
        devices.append(f'GPU:{g}')
    
    if len(cfg['GPU']) > 1:
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    else:
        strategy = None
    
    logger = Logger(f"{cfg['LOG_PATH']}_{cfg['NAME']}_{start_time}.txt")
    start_time = time.time()
    
    if cfg['SEED']:
        #seed_everything(cfg['SEED'])
        tf.keras.utils.set_random_seed(cfg['SEED'])  # sets seeds for base-python, numpy and tf
        tf.cfg.experimental.enable_op_determinism()
    if cfg['HP_SEARCH']:
        searcher = HPSearcher(args=args, cfg=cfg, logger=logger, strategy=strategy, trial=None)
        searcher.hp_search()

    elif cfg['METHOD'] in ['KD']:
        distiller = Distiller(cfg, logger, strategy)
        if cfg['TEST']:
            test_loss, test_metr = distiller.evaluate(trainer.ds_test, "test")
            print(f"Test loss: {test_loss}, Test mIoU: {test_metr}")
        else:
            distiller.train()
        
    else:
        trainer = Trainer(cfg, logger, strategy)
        if cfg['TEST']:
            test_loss, test_metr = trainer.evaluate(trainer.ds_test, "test")
            print(f"Test loss: {test_loss}, Test mIoU: {test_metr}")
        else:
            trainer.train()
    
    print(f"--- {(time.time() - start_time):.1f} seconds ---")
    return

  
if __name__ == "__main__":
    main()