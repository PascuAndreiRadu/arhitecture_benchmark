"""
benchmark.py
============
Provides a single ``bench_mark`` class that runs a suite of out-of-the-box
scikit-learn models (classification / regression) or Stable-Baselines3 agents
(reinforcement learning) against a user-supplied dataset or environment.
 
The intent is to give a quick, no-fuss baseline so that custom architectures
can be compared against well-known algorithms with sensible hyper-parameters.
 
Typical usage
-------------
>>> from benchmark import bench_mark
>>>
>>> # --- supervised learning ---
>>> bm = bench_mark()
>>> cls_results = bm.classification({"X_train": X_tr, "y_train": y_tr,
...                                   "X_test": X_te, "y_test": y_te})
>>> reg_results = bm.regression({"X_train": X_tr, "y_train": y_tr,
...                               "X_test": X_te, "y_test": y_te})
>>>
>>> # --- reinforcement learning ---
>>> rl_results = bm.rl(env, time_steps=100_000)
"""
 
import sys
import os
from dataclasses import dataclass
 
import numpy as np
 
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import A2C, DDPG, DQN, SAC, TD3, PPO
from sb3_contrib import ARS, CrossQ, QRDQN, TQC, TRPO
 
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
 
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
 
import gymnasium as gym
import shap
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
 
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch  # noqa: F401  (kept for downstream use / GPU detection)
 
 
class SB3Registry:
    """
    Maps Gymnasium action-space types to the SB3 / sb3_contrib algorithms that
    support them.
 
    Supported space types
    ---------------------
    ``"box"``
        Continuous action spaces (``gymnasium.spaces.Box``).
    ``"discrete"``
        Single-integer action spaces (``gymnasium.spaces.Discrete``).
    ``"multi_discrete"``
        Vector of discrete actions (``gymnasium.spaces.MultiDiscrete``).
    ``"multi_binary"``
        Binary action vector (``gymnasium.spaces.MultiBinary``).
    """
 
    _MODELS: dict[str, list] = {
        "box":            [ARS, A2C, CrossQ, DDPG, PPO, SAC, TD3, TQC, TRPO],
        "discrete":       [A2C, DQN, PPO, QRDQN, TRPO],
        "multi_discrete": [A2C, PPO, TRPO],
        "multi_binary":   [A2C, PPO, TRPO],
    }
 
    @classmethod
    def get_models(cls, space_type: str) -> list:
        """
        Return the list of algorithm *classes* compatible with *space_type*.
 
        Parameters
        ----------
        space_type : str
            One of ``"box"``, ``"discrete"``, ``"multi_discrete"``,
            ``"multi_binary"`` (case-insensitive).
 
        Returns
        -------
        list
            Uninstantiated SB3 algorithm classes.
 
        Raises
        ------
        ValueError
            If *space_type* is not among the supported keys.
        """
        space_type = space_type.lower()
        if space_type not in cls._MODELS:
            raise ValueError(
                f"Unsupported space type: '{space_type}'. "
                f"Choose from {list(cls._MODELS.keys())}"
            )
        return cls._MODELS[space_type]
 
 
@dataclass
class RlResults:
    """Stores the outcome of a single RL algorithm benchmark run.
 
    Attributes
    ----------
    model_name : str
        Class name of the algorithm (e.g. ``"PPO"``).
    model : BaseAlgorithm
        The trained SB3 model instance.
    std_reward : float
        Standard deviation of episode rewards over the evaluation episodes.
    mean_reward : float
        Mean episode reward over the evaluation episodes.
    shap_values : np.ndarray
        SHAP values computed for a sample of observations collected during a
        rollout.  Shape depends on the observation and action space dimensions.
    """
 
    model_name: str
    model: BaseAlgorithm
    std_reward: float
    mean_reward: float
    shap_values: np.ndarray
 
 
@dataclass
class Results:
    """Base dataclass for supervised-learning benchmark results.
 
    Attributes
    ----------
    model_name : str
        Class name of the estimator (e.g. ``"RandomForestClassifier"``).
    model : BaseEstimator
        The fitted scikit-learn estimator.
    results : np.ndarray
        Raw predictions produced by the estimator on the test set.
    shap_values : np.ndarray
        SHAP values computed for the first 100 training samples using a
        ``KernelExplainer`` backed by a k-means summary of the training data.
    """
 
    model_name: str
    model: BaseEstimator
    results: np.ndarray
    shap_values: np.ndarray
 
 
@dataclass
class ClsResults(Results):
    """Classification-specific benchmark results.
 
    Inherits all fields from :class:`Results` and adds classification metrics.
 
    Attributes
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix of shape ``(n_classes, n_classes)``.
    precision : float
        Weighted-average precision score across classes.
    recall : float
        Weighted-average recall score across classes.
    fscore : float
        Weighted-average F1 score across classes.
    """
 
    confusion_matrix: np.ndarray
    precision: float
    recall: float
    fscore: float
 
 
@dataclass
class RegResults(Results):
    """Regression-specific benchmark results.
 
    Inherits all fields from :class:`Results` and adds regression metrics.
 
    Attributes
    ----------
    mae : float
        Mean Absolute Error.
    mse : float
        Mean Squared Error.
    mape : float
        Mean Absolute Percentage Error.
    r2_score : float
        Coefficient of determination R².
    """
 
    mae: float
    mse: float
    mape: float
    r2_score: float
 
 
class bench_mark:
    """
    Benchmarks a collection of out-of-the-box models for a given task.
 
    The class is designed as a quick sanity-check / baseline tool.  Train all
    common scikit-learn estimators (or SB3 RL agents) on the supplied data
    and return structured result objects so that the caller can compare
    metrics side-by-side against a custom architecture.
 
    Attributes
    ----------
    results : list[RlResults]
        Populated after calling :meth:`rl`; holds the results of the most
        recent RL benchmark run.  Not set by :meth:`classification` or
        :meth:`regression`.
 
    Methods
    -------
    classification(dataset)
        Run classification benchmarks.
    regression(dataset)
        Run regression benchmarks.
    rl(env, time_steps, eval_freq)
        Run reinforcement-learning benchmarks.
    """
 
    def classification(self, dataset: dict) -> list[ClsResults]:
        """
        Fit and evaluate a suite of classifiers on the supplied dataset.
 
        The following algorithms are benchmarked with sensible default
        hyper-parameters:
 
        * K-Nearest Neighbours (k=3)
        * Linear SVM
        * RBF SVM
        * Gaussian Process Classifier
        * Decision Tree (max depth 5)
        * Random Forest (max depth 5, 10 estimators)
        * MLP Classifier
        * AdaBoost
        * Gaussian Naïve Bayes
        * Quadratic Discriminant Analysis
 
        Parameters
        ----------
        dataset : dict
            Must contain four keys:
 
            ``"X_train"`` : array-like of shape (n_samples, n_features)
                Training feature matrix.
            ``"y_train"`` : array-like of shape (n_samples,)
                Training labels.
            ``"X_test"`` : array-like of shape (n_samples, n_features)
                Test feature matrix.
            ``"y_test"`` : array-like of shape (n_samples,)
                Test labels.
 
        Returns
        -------
        list[ClsResults]
            One :class:`ClsResults` entry per algorithm, sorted in the order
            the classifiers were evaluated.
        """
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
 
        results: list[ClsResults] = []
 
        for cls in classifiers:
            cls.fit(dataset["X_train"], dataset["y_train"])
            y_pred = cls.predict(dataset["X_test"])
            precision, recall, fscore, _ = precision_recall_fscore_support(
                dataset["y_test"], y_pred, average="weighted", zero_division=0
            )
            shap_values = self._shap_feature_importance(cls, dataset["X_train"])
            results.append(
                ClsResults(
                    model_name=type(cls).__name__,
                    model=cls,
                    results=y_pred,           # BUG FIX: was missing, causing TypeError
                    confusion_matrix=confusion_matrix(dataset["y_test"], y_pred),
                    precision=precision,
                    recall=recall,
                    fscore=fscore,
                    shap_values=shap_values,
                )
            )
 
        return results
 
    def regression(self, dataset: dict) -> list[RegResults]:
        """
        Fit and evaluate a suite of regressors on the supplied dataset.
 
        The following algorithms are benchmarked with sensible default
        hyper-parameters:
 
        * K-Nearest Neighbours (k=3)
        * Linear SVR
        * RBF SVR
        * Gaussian Process Regressor
        * Decision Tree (max depth 5)
        * Random Forest (max depth 5, 10 estimators)
        * MLP Regressor
        * AdaBoost Regressor
        * Bayesian Ridge
        * Polynomial SVR (degree 3)
 
        Parameters
        ----------
        dataset : dict
            Must contain four keys:
 
            ``"X_train"`` : array-like of shape (n_samples, n_features)
                Training feature matrix.
            ``"y_train"`` : array-like of shape (n_samples,)
                Training targets.
            ``"X_test"`` : array-like of shape (n_samples, n_features)
                Test feature matrix.
            ``"y_test"`` : array-like of shape (n_samples,)
                Test targets (ground truth).
 
        Returns
        -------
        list[RegResults]
            One :class:`RegResults` entry per algorithm, sorted in the order
            the regressors were evaluated.
        """
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
 
        results: list[RegResults] = []
 
        for reg in regressors:
            reg.fit(dataset["X_train"], dataset["y_train"])
            y_pred = reg.predict(dataset["X_test"])
            mae  = mean_absolute_error(dataset["y_test"], y_pred)
            mse  = mean_squared_error(dataset["y_test"], y_pred)
            mape = mean_absolute_percentage_error(dataset["y_test"], y_pred)
            r2   = r2_score(dataset["y_test"], y_pred)
            shap_values = self._shap_feature_importance(reg, dataset["X_train"])
 
            results.append(
                RegResults(
                    model_name=type(reg).__name__,
                    model=reg,
                    results=y_pred,           # BUG FIX: was missing, causing TypeError
                    mae=mae,
                    mse=mse,
                    mape=mape,
                    r2_score=r2,
                    shap_values=shap_values,
                )
            )
 
        return results
 
    def rl(self, env, time_steps: int, eval_freq: int = -1) -> list[RlResults]:
        """
        Train and evaluate all SB3 algorithms compatible with *env*'s action
        space.
 
        The appropriate policy network (``"MlpPolicy"``, ``"CnnPolicy"``, or
        ``"MultiInputPolicy"``) is selected automatically based on the
        observation space.
 
        Optionally, an :class:`~stable_baselines3.common.callbacks.EvalCallback`
        is registered during training to save the best checkpoint and log
        evaluation metrics at a regular cadence.
 
        Parameters
        ----------
        env : gymnasium.Env
            A Gymnasium-compatible environment.  Must expose ``action_space``
            and ``observation_space``.
        time_steps : int
            Total number of environment steps each algorithm is trained for.
        eval_freq : int, optional
            Frequency (in environment steps) at which the trained policy is
            evaluated during training.  Pass ``-1`` (default) to auto-set it
            to ``time_steps // 20`` (i.e. 20 evaluation checkpoints).  Pass
            ``0`` to disable mid-training evaluation entirely.
 
        Returns
        -------
        list[RlResults]
            One :class:`RlResults` entry per algorithm, in the order they were
            trained.  Also stored in ``self.results``.
 
        Raises
        ------
        TypeError
            If ``env.action_space`` is not one of the four supported Gymnasium
            space types.
        """
        action_space = env.action_space
        results: list[RlResults] = []
        eval_callback = None
 
        match action_space:
            case Box():
                available_models = SB3Registry.get_models("box")
            case Discrete():
                available_models = SB3Registry.get_models("discrete")
            case MultiBinary():
                available_models = SB3Registry.get_models("multi_binary")
            case MultiDiscrete():
                available_models = SB3Registry.get_models("multi_discrete")
            case _:
                raise TypeError(
                    f"Action space '{type(action_space).__name__}' is not supported."
                )
 
        policy = self._policy_select(env)
 
        if eval_freq == -1:
            eval_freq = time_steps // 20
 
        for model_cls in available_models:
            model_name = model_cls.__name__
 
            if eval_freq > 0:
                eval_callback = EvalCallback(
                    env,
                    best_model_save_path=f"./logs/{model_name}",
                    log_path=f"./logs/{model_name}_results",
                    eval_freq=eval_freq,
                    deterministic=True,
                    render=False,
                )
 
            env.reset()
            model = model_cls(policy, env, verbose=0)
            model.learn(time_steps, callback=eval_callback)
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
            shap_values = self._rl_shap_feature_importance(env, model)
 
            results.append(
                RlResults(
                    model_name=model_name,
                    shap_values=shap_values,
                    model=model,
                    std_reward=std_reward,
                    mean_reward=mean_reward,
                )
            )
 
        self.results = results
        return results
 
    def _shap_feature_importance(self, model: BaseEstimator, data: np.ndarray) -> np.ndarray:
        """
        Compute SHAP feature importances for a fitted scikit-learn estimator.
 
        Uses :class:`shap.KernelExplainer` with a k-means summary of *data* as
        the background dataset.  The explainer is applied to the first 100
        rows of *data* (or fewer if the dataset is smaller).
 
        Parameters
        ----------
        model : BaseEstimator
            A fitted scikit-learn estimator that exposes a ``predict`` method.
        data : np.ndarray of shape (n_samples, n_features)
            The training feature matrix used both to build the background
            summary and as the set of samples to explain.
 
        Returns
        -------
        np.ndarray
            SHAP values array.  For single-output models the shape is
            ``(n_explain_samples, n_features)``; for multi-output models an
            additional leading dimension is added by SHAP.
 
        Notes
        -----
        The number of k-means clusters is capped at ``min(50, len(data))`` to
        avoid errors on small datasets.
        """
        n_clusters = min(50, len(data))          # BUG FIX: crash when len(data) < 50
        background = shap.kmeans(data, n_clusters)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(data[:100])
        return shap_values
 
    def _rl_shap_feature_importance(self, env, model: BaseAlgorithm) -> np.ndarray:
        """
        Compute SHAP feature importances for a trained SB3 RL agent.
 
        Collects up to 500 observations by rolling out the *model*'s
        deterministic policy in *env*, then explains the first 100 of those
        observations using :class:`shap.KernelExplainer`.  The SHAP target
        function maps observations to the (scalar or vector) actions predicted
        by the agent via :meth:`_predict_fn`.
 
        Parameters
        ----------
        env : gymnasium.Env
            The environment used for the rollout.  Its state is reset at the
            start and whenever an episode terminates.
        model : BaseAlgorithm
            A trained SB3 model with a ``predict`` method.
 
        Returns
        -------
        np.ndarray
            SHAP values array of shape ``(n_explain_samples, n_obs_features)``,
            where ``n_explain_samples = min(100, collected_steps)``.
 
        Notes
        -----
        The number of k-means clusters is capped at ``min(50, len(X))`` to
        avoid errors when fewer than 500 non-terminal steps were collected.
        """
        obs, _ = env.reset()
        obs_list = []
 
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            obs_list.append(obs)
            obs = next_obs if not (done or truncated) else env.reset()[0]
 
        X = np.array(obs_list)
 
        n_clusters = min(50, len(X))             # BUG FIX: crash when len(X) < 50
        background = shap.kmeans(X, n_clusters)
        explainer = shap.KernelExplainer(
            lambda x: self._predict_fn(model, x),
            background,
        )
        shap_values = explainer.shap_values(X[:100])
        return shap_values
 
    def _predict_fn(self, model: BaseAlgorithm, obs_batch: np.ndarray) -> np.ndarray:
        """
        Wrap an SB3 model's ``predict`` method so that it accepts a batch of
        observations as a plain NumPy array.
 
        :class:`shap.KernelExplainer` calls its prediction function with a
        2-D array of perturbed samples.  SB3's ``model.predict`` handles one
        observation at a time, so this helper iterates over the batch and
        stacks the results.
 
        Parameters
        ----------
        model : BaseAlgorithm
            A trained SB3 model.
        obs_batch : np.ndarray of shape (n_samples, n_obs_features)
            Batch of observations to predict actions for.
 
        Returns
        -------
        np.ndarray of shape (n_samples, *action_shape)
            Predicted actions for each observation in *obs_batch*.
        """
        actions = []
        for obs in obs_batch:
            action, _ = model.predict(obs, deterministic=True)
            actions.append(action)
        return np.array(actions)
 
    def _policy_select(self, env) -> str:
        """
        Infer the appropriate SB3 policy string from the environment's
        observation space.
 
        Rules
        -----
        * ``Dict`` observation space  →  ``"MultiInputPolicy"``
        * 3-D ``Box`` observation space (image)  →  ``"CnnPolicy"``
        * Anything else  →  ``"MlpPolicy"``
 
        Parameters
        ----------
        env : gymnasium.Env
            The target environment.
 
        Returns
        -------
        str
            One of ``"MultiInputPolicy"``, ``"CnnPolicy"``, ``"MlpPolicy"``.
        """
        from gymnasium.spaces import Dict, Box
 
        obs_space = env.observation_space
 
        if isinstance(obs_space, Dict):
            return "MultiInputPolicy"
        elif isinstance(obs_space, Box) and len(obs_space.shape) == 3:
            # 3-D Box is assumed to be an image (H × W × C)
            return "CnnPolicy"
        else:
            return "MlpPolicy"
 