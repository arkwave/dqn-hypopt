import numpy as np 
from hyperopt import hp, fmin, tpe, Trials
from dqn_main import run


def bayesian_opt(run, space, max_evals=50):
    trials = Trials()
    best_setting = fmin(run, space, algo=tpe.suggest, max_evals=50, trials=trials)
    return best_setting, trials  


if __name__ == "__main__":
    default_params = {'mode': 'train',
                      'render': False,
                      'log_interval': 1000,
                      'env_name': 'MountainCar-v0',
                      'num_iterations': 1000, 
                      'num_episodes': 100,
                      'exploration_noise': 0.1, 
                      'model_name': 'trial_model'} 

    hypopt_params = {
                    'capacity': hp.randint('buffer size', 1, 10000),
                    'update_count': hp.randint('update_count', 1, 100),
                    'gamma': hp.uniform('gamma', 0, 1),
                    'batch_size': hp.randint('batch_size', 2, 128),
                    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd', 'adagrad']),
                    'learning_rate': hp.uniform('learning_rate', 0, 1)
                    }

    space = {}
    space.update(default_params)
    space.update(hypopt_params)
    best_setting, trials = bayesian_opt(run, space, max_evals=50)
