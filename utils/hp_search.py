import os
import joblib

import optuna

from utils.train import Trainer
from utils.distiller import Distiller
from utils.tools import get_cfg
from keras import backend as K
import gc


class HPSearcher:
      
    def __init__(self, args, cfg, logger=None, strategy=None, trial=None):
        self.args = args
        self.cfg = cfg
        self.logger = logger
        self.strategy = strategy
        self.trial = trial
        
        self.search_space = {
            # "LR": [5e-5, 5e-4],
            # "TARGET": ["tree_2", "chard", "lettuce", "vineyard"],
            # "FREEZE_BACKBONE": [True, False],
            "T": [2, 3],
            "ALPHA": [0.1],
            "WCTA": [0, 0.001],
            # "SOUP": ['uniform', False],
            # "STYLE_AUG": [True, False],
            # "WHITEN_LAYERS": [(),(0,0),(0,1),(0,1,2)],
            # "FILTER": ['error', False]
            "NORM": [None, "post_norm"],
            # "TEACHERS": ["tf_geom_wcta"],
            # "WEIGHT": ['iou'],
            # "CL": [0.1, 0.5, 1.0, 2.0, 5.0],
            }
    
    def get_random_hps(self):

        self.cfg['KD']['T'] = self.trial.suggest_categorical("T", [2, 3])
        self.cfg['KD']['ALPHA'] = self.trial.suggest_categorical("ALPHA", [0.1])
        # self.cfg['KD']['FILTER'] = self.trial.suggest_categorical("FILTER", ['error', False])
        # self.cfg['KD']['SOUP'] = self.trial.suggest_categorical("SOUP", ['uniform', False])
        self.cfg['WCTA'] = self.trial.suggest_categorical("WCTA", [0, 0.001])
        # self.cfg['STYLE_AUG'] = self.trial.suggest_categorical("STYLE_AUG", [True, False])
        self.cfg['KD']['NORM'] = self.trial.suggest_categorical("NORM", [None, "post_norm"])
        # self.cfg['TEACHERS'] = self.trial.suggest_categorical("TEACHERS", ["tf_geom", "tf_geom_wcta"])
        self.cfg['TEACHERS'] = f"{self.cfg['NORM']}_" + \
                               f"{'style' if self.cfg['STYLE_AUG'] else 'geom'}" + \
                               f"{'_wcta' if self.cfg['WCTA'] else ''}"
        # self.cfg['FWCTA'] = self.trial.suggest_categorical("FWCTA", [True, False])
        # self.cfg['WHITEN_LAYERS'] = self.trial.suggest_categorical("WHITEN_LAYERS", [(),(0,0),(0,1),(0,1,2)])
        # self.cfg['ADAMW']['LR'] = self.trial.suggest_categorical("LR", [5e-5, 5e-4])
        # self.cfg['FREEZE_BACKBONE'] = self.trial.suggest_categorical("FREEZE_BACKBONE", [True, False])        
        # self.cfg['KD']['WEIGHT'] = self.trial.suggest_categorical("WEIGHT", ['iou'])
        # self.cfg['CL']['TEMP'] = self.trial.suggest_categorical("CL", [0.1, 0.5, 1.0, 2.0, 5.0])

        if self.cfg['VERBOSE']:
            self.logger.save_log(self.cfg[self.cfg['MODE']])

        # print(f"KD={self.cfg['KD']['T']}, ALPHA={self.cfg['KD']['ALPHA']}, WCTA={self.cfg['WCTA']}")
        # print(f"WCTA={self.cfg['WCTA']}, SOUP={self.cfg['KD']['SOUP']}")
        # print(f"FILTER={self.cfg['KD']['FILTER']}")
        # print(f"FWCTA={self.cfg['FWCTA']}, WHITEN_LAYERS={self.cfg['WHITEN_LAYERS']}")
        # print(f"FILTER={self.cfg['KD']['FILTER']}")
        print(f"NORM={self.cfg['KD']['NORM']}, WCTA={self.cfg['WCTA']}")
        print(f"T={self.cfg['KD']['T']}, ALPHA={self.cfg['KD']['ALPHA']}")
    
    def objective(self, trial):
        name = 'gridsearch_' + str(trial.datetime_start) + str(trial.number)
        self.trial = trial 
        self.cfg = get_cfg(self.args)
        self.get_random_hps()
        target_domains = self.cfg['TARGET'] if isinstance(self.cfg['TARGET'], list) else [self.cfg['TARGET']]
        metrics = []
        for domain in target_domains:
            self.cfg['TARGET'] = domain
            if self.cfg['KD']:
                distiller = Distiller(cfg=self.cfg, logger=self.logger, strategy=self.strategy, trial=self.trial)
                metr = distiller.train()
                #distiller.test()        
                del distiller.model
                del distiller.teacher
                del distiller
            else:
                trainer = Trainer(cfg=self.cfg, logger=self.logger, strategy=self.strategy, trial=self.trial)
                metr = trainer.train()
                #trainer.test()
                del trainer.model
                del trainer
            
            metrics.append(metr.numpy())
            K.clear_session()
            gc.collect()
        
        print(f"Metrics: {metrics}, Mean: {sum(metrics) / len(metrics)}")
        return sum(metrics) / len(metrics)
    
    
    def hp_search(self):
        name = f'hp_search_{self.cfg["HP_SEARCH_NAME"]}_{self.cfg["TARGET"]}'
        self.study = optuna.create_study(study_name=name, 
                                         direction='maximize', 
                                         sampler=optuna.samplers.GridSampler(self.search_space),
                                         pruner=optuna.pruners.NopPruner())
        
        if os.path.exists(f'{self.cfg["HP_SEARCH_DIR"]}/{name}.pkl'): 
            study_old = joblib.load(f'{self.cfg["HP_SEARCH_DIR"]}/{name}.pkl')
            self.study.add_trials(study_old.get_trials())
            print('Study resumed!')
        
        save_callback = SaveCallback(self.cfg['HP_SEARCH_DIR'])
        self.study.optimize(lambda trial: self.objective(trial), n_trials=self.cfg['N_TRIALS'],
                            callbacks=[save_callback])

        pruned_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])

        self.logger.save_log("Study statistics: ")
        self.logger.save_log(f"  Number of finished trials: {len(self.study.trials)}")
        self.logger.save_log(f"  Number of pruned trials: {len(pruned_trials)}")
        self.logger.save_log(f"  Number of complete trials: {len(complete_trials)}")
        self.logger.save_log("Best trial:")
        self.logger.save_log(f"  Value: {self.study.best_trial.value}")
        self.logger.save_log("  Params: ")
        for key, value in self.study.best_trial.params.items():
            self.logger.save_log(f"    {key}: {value}")

        return self.study
    
    
    
class SaveCallback:
    
    def __init__(self, directory):
        self.directory = directory

    def __call__(self, study, trial):
        joblib.dump(study, os.path.join(self.directory, f'{study.study_name}.pkl'))