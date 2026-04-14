import sys
import os
from stable_baselines3 import A2C,DDPG,DQN,SAC,TD3,PPO
from sb3_contrib import ARS,CrossQ,QRDQN,TQC,TRPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import torch
import numpy as np

class SB3Registry:
    # We map the model classes to their supported observation spaces
    _MODELS = {
        "box": [ARS, A2C, CrossQ, DDPG, PPO, SAC, TD3, TQC, TRPO],
        "discrete": [A2C, DQN, PPO, QRDQN, TRPO],
        "multi_discrete": [A2C, PPO, TRPO],
        'multi_binary':[A2C,PPO,TRPO]
    }

    @classmethod
    def get_models(cls, space_type: str):
        space_type = space_type.lower()
        if space_type not in cls._MODELS:
            raise ValueError(f"Unsupported space type: {space_type}. Use {list(cls._MODELS.keys())}")
        
        return cls._MODELS[space_type]
    
class bench_mark():

    """
            A utility class that benchmarks and returns the best models for the task, it is intended for the user to 
            compare their own arhitecture with a out of the box solution,
        Args:
            model_type:str (ppo,classification,regression), if the model is a 
            model:dict model must be a dict that either has the key ['NN'] or the keys ['actor'] and ['critic']
            dataset_path:str expecting a pickled df for ppo or pickled torch dataset for regression and classification
            preimplemented_arhitecture:bool uses a pre-implemented arhitecture for comparison with the user given model
    """
        

    def classification(self,dataset:dict):
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025, random_state=42),
            SVC(gamma=2, C=1, random_state=42),
            GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            AdaBoostClassifier(random_state=42),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]
        results={}

        for cls in classifiers:
            cls.fit(dataset['X_train'],dataset['y_train'])
            score = cls.score(dataset['X_test'],dataset['y_test'])
            results[type(cls).__name__]=score
            results[f'{type(cls).__name__}_model'] = cls

        return results

    def regression(self, dataset: dict):
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.linear_model import BayesianRidge

        regressors = [
            KNeighborsRegressor(3),
            SVR(kernel="linear", C=0.025),
            SVR(kernel="rbf", gamma=2, C=1),
            GaussianProcessRegressor(1.0 * RBF(1.0), random_state=42),
            DecisionTreeRegressor(max_depth=5, random_state=42),
            RandomForestRegressor(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            MLPRegressor(alpha=1, max_iter=1000, random_state=42),
            AdaBoostRegressor(random_state=42),
            BayesianRidge(),
            SVR(kernel="poly", degree=3, C=1),
        ]
        
        results = {}
        

        for reg in regressors:
            reg.fit(dataset['X_train'], dataset['y_train'])
            
            score = reg.score(dataset['X_test'], dataset['y_test'])
            results[type(reg).__name__] = score
            results[f'{type(reg).__name__}_model'] = reg

        return results,        
    
    def rl(self,env,episodes):
        from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
        from stable_baselines3.common.evaluation import evaluate_policy
        action_space = env.action_space
        results={}
        match action_space:
            case Box():
                available_models=SB3Registry.get_models('box')
            case Discrete():
                available_models=SB3Registry.get_models('discrete')
            case MultiBinary():
                available_models=SB3Registry.get_models('multi_binary')
            case MultiDiscrete():
                available_models=SB3Registry.get_models('multi_discrete')
            case _:
                raise TypeError(f'action space is not supported: {action_space}')
            
        policy = self._policy_select(env)
            
        for model_name in available_models:
            env.reset()
            model = model_name(policy,env,verbose=0)
            model.learn(episodes)   
            mean_reward,std_reward = evaluate_policy(model,env,n_eval_episodes=10)
            results[f'{type(model).__name__}_mean_reward']=mean_reward
            results[f'{type(model).__name__}_std_reward']=std_reward
            results[f'{type(model).__name__}_obj']=model

        return results

    def _policy_select(self,env):
        obs_space = env.observation_space
        from gymnasium.spaces import Dict,Box
        
        if isinstance(obs_space, Dict):
            return "MultiInputPolicy"
            
        elif isinstance(obs_space, Box) and len(obs_space.shape) == 3:
            return "CnnPolicy"
        else:
            return "MlpPolicy"     
        

        
