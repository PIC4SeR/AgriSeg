import os
import argparse
import joblib

import optuna

from utils.train import Trainer
from utils.distiller import Distiller
from utils.tools import get_cfg
from keras import backend as K
import gc


class HPSearcher:
      
    def __init__(self, args, config, logger=None, strategy=None, trial=None):
        self.args = args
        self.config = config
        self.logger = logger
        self.strategy = strategy
        self.trial = trial
        
        self.search_space = {
                            "LR": [5e-5, 5e-4],
                            # "TARGET": ["tree_2", "chard", "lettuce", "vineyard"],
                            # "FREEZE_BACKBONE": [True, False],
                            "T": [1, 2, 3],
                            "ALPHA": [0.001, 0.01, 0.1],
                            # "WCTA": [True, False],
                            # "STYLE_AUG": [True, False],
                            # "FWCTA": [True, False],
                            }
    
    def get_random_hps(self):
        
        # self.config['CL']['WEIGHT'] = self.trial.suggest_categorical("AUX_WEIGHT", [0.01, 0.1])
        # self.config['PADAIN']['P'] = self.trial.suggest_categorical("P", [0.001, 0.01])
        # self.config['CL']['TEMP'] = self.trial.suggest_categorical("TEMP", [0.1, 0.5])

        self.config['KD']['T'] = self.trial.suggest_categorical("T", [1, 2, 3])
        self.config['KD']['ALPHA'] = self.trial.suggest_categorical("ALPHA", [0.001, 0.01, 0.1])
        # self.config['WCTA'] = self.trial.suggest_categorical("WCTA", [True, False])
        # self.config['STYLE_AUG'] = self.trial.suggest_categorical("STYLE_AUG", [True, False])
        # self.config['FWCTA'] = self.trial.suggest_categorical("FWCTA", [True, False])
        # self.config['WHITEN_LAYERS'] = self.trial.suggest_categorical("WHITEN_LAYERS", [(),(0,0),(0,1),(0,1,2)])

        self.config['ADAMW']['LR'] = self.trial.suggest_categorical("LR", [5e-5, 5e-4])
        # self.config['TARGET'] = self.trial.suggest_categorical("TARGET", ["tree_2", "chard", "lettuce", "vineyard"])
        # self.config['FREEZE_BACKBONE'] = self.trial.suggest_categorical("FREEZE_BACKBONE", [True, False])        
        
        if self.config['VERBOSE']:
            self.logger.save_log(self.config[self.config['MODE']])

        print(f"LR={self.config['ADAMW']['LR']}, KD={self.config['KD']['T']}, ALPHA={self.config['KD']['ALPHA']}")
    
    
    def objective(self, trial):
        name = 'gridsearch_' + str(trial.datetime_start) + str(trial.number)
        self.trial = trial 
        self.config = get_cfg(self.args)
        self.get_random_hps()
        target_domains = self.config['TARGET'] if isinstance(self.config['TARGET'], list) else [self.config['TARGET']]
        metrics = []
        for domain in target_domains:
            self.config['TARGET'] = domain
            if self.config['KD']:
                distiller = Distiller(config=self.config, logger=self.logger, strategy=self.strategy, trial=self.trial)
                metr = distiller.train()
                #distiller.test()        
                del distiller.model
                del distiller.teacher
                del distiller
            else:
                trainer = Trainer(config=self.config, logger=self.logger, strategy=self.strategy, trial=self.trial)
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
        name = f'hp_search_{self.config["HP_SEARCH_NAME"]}_{self.config["TARGET"]}'
        self.study = optuna.create_study(study_name=name, 
                                         direction='maximize', 
                                         sampler=optuna.samplers.GridSampler(self.search_space),
                                         pruner=optuna.pruners.NopPruner())
        
        if os.path.exists(f'{self.config["HP_SEARCH_DIR"]}/{name}.pkl'): 
            study_old = joblib.load(f'{self.config["HP_SEARCH_DIR"]}/{name}.pkl')
            self.study.add_trials(study_old.get_trials())
            print('Study resumed!')
        
        save_callback = SaveCallback(self.config['HP_SEARCH_DIR'])
        self.study.optimize(lambda trial: self.objective(trial), n_trials=self.config['N_TRIALS'],
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