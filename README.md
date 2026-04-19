# benchmark.py

A zero-configuration baseline tool that trains and evaluates a curated suite of **scikit-learn** models and **Stable-Baselines3** RL agents against your data in a single call. Every result includes SHAP feature-importance values out of the box.

---

## Table of Contents

- [Why use this?](#why-use-this)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [bench\_mark.classification()](#bench_markclassification)
  - [bench\_mark.regression()](#bench_markregression)
  - [bench\_mark.rl()](#bench_markrl)
- [Return Types](#return-types)
  - [ClsResults](#clsresults)
  - [RegResults](#regresults)
  - [RlResults](#rlresults)
- [Supported Models](#supported-models)
- [RL Action Space Compatibility](#rl-action-space-compatibility)
- [SHAP Feature Importance](#shap-feature-importance)
- [Output & Logs](#output--logs)

---

## Why use this?

When building a custom ML model or RL agent, the first question is always: *how much better is it than a standard baseline?*

`bench_mark` answers that in one call — no boilerplate, no tuning, no copy-pasting training loops. It trains every relevant algorithm on your data with sensible hyperparameters, collects all the metrics you care about, and returns structured result objects ready to compare side-by-side.

---

## Installation

```bash
pip install scikit-learn stable-baselines3 sb3-contrib gymnasium shap torch numpy
```

> **Note:** `sb3_contrib` provides ARS, CrossQ, QRDQN, TQC, and TRPO. Make sure your `torch` version is compatible with your CUDA version if you intend to use a GPU.

---

## Quick Start

### Classification

```python
from benchmark import bench_mark
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bm = bench_mark()
results = bm.classification({
    "X_train": X_train,
    "y_train": y_train,
    "X_test":  X_test,
    "y_test":  y_test,
})

for r in results:
    print(f"{r.model_name:35s}  F1={r.fscore:.3f}  P={r.precision:.3f}  R={r.recall:.3f}")
```

### Regression

```python
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = bm.regression({
    "X_train": X_train,
    "y_train": y_train,
    "X_test":  X_test,
    "y_test":  y_test,
})

for r in results:
    print(f"{r.model_name:35s}  R²={r.r2_score:.3f}  MAE={r.mae:.3f}  MSE={r.mse:.3f}")
```

### Reinforcement Learning

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
results = bm.rl(env, time_steps=50_000)

for r in results:
    print(f"{r.model_name:10s}  mean_reward={r.mean_reward:.1f}  std={r.std_reward:.1f}")
```

---

## API Reference

### `bench_mark.classification()`

```python
bench_mark.classification(dataset: dict) -> list[ClsResults]
```

Trains and evaluates 10 classifiers on the supplied dataset. Returns one `ClsResults` object per algorithm.

**Parameters**

| Key | Type | Description |
|---|---|---|
| `X_train` | array-like `(n_samples, n_features)` | Training feature matrix |
| `y_train` | array-like `(n_samples,)` | Training labels |
| `X_test` | array-like `(n_samples, n_features)` | Test feature matrix |
| `y_test` | array-like `(n_samples,)` | Test labels |

---

### `bench_mark.regression()`

```python
bench_mark.regression(dataset: dict) -> list[RegResults]
```

Trains and evaluates 10 regressors on the supplied dataset. Returns one `RegResults` object per algorithm.

**Parameters** — same four-key dict as `classification()`, with `y_train` / `y_test` being continuous targets rather than class labels.

---

### `bench_mark.rl()`

```python
bench_mark.rl(env, time_steps: int, eval_freq: int = -1) -> list[RlResults]
```

Trains every SB3 algorithm compatible with `env`'s action space. The policy type (`MlpPolicy`, `CnnPolicy`, `MultiInputPolicy`) is selected automatically from the observation space. Returns one `RlResults` object per algorithm and also stores them in `self.results`.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `env` | `gymnasium.Env` | — | A Gymnasium-compatible environment |
| `time_steps` | `int` | — | Total training steps per algorithm |
| `eval_freq` | `int` | `-1` | Evaluation frequency during training. `-1` → `time_steps // 20` (20 checkpoints). `0` → disabled |

**Raises** `TypeError` if `env.action_space` is not `Box`, `Discrete`, `MultiDiscrete`, or `MultiBinary`.

---

## Return Types

### `ClsResults`

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Class name of the estimator |
| `model` | `BaseEstimator` | The fitted scikit-learn object |
| `results` | `np.ndarray` | Raw predictions on the test set |
| `confusion_matrix` | `np.ndarray` | Shape `(n_classes, n_classes)` |
| `precision` | `float` | Weighted-average precision |
| `recall` | `float` | Weighted-average recall |
| `fscore` | `float` | Weighted-average F1 score |
| `shap_values` | `np.ndarray` | SHAP values for the first 100 training samples |

### `RegResults`

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Class name of the estimator |
| `model` | `BaseEstimator` | The fitted scikit-learn object |
| `results` | `np.ndarray` | Raw predictions on the test set |
| `mae` | `float` | Mean Absolute Error |
| `mse` | `float` | Mean Squared Error |
| `mape` | `float` | Mean Absolute Percentage Error |
| `r2_score` | `float` | Coefficient of determination R² |
| `shap_values` | `np.ndarray` | SHAP values for the first 100 training samples |

### `RlResults`

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Class name of the SB3 algorithm |
| `model` | `BaseAlgorithm` | The trained SB3 model |
| `mean_reward` | `float` | Mean episode reward over 10 evaluation episodes |
| `std_reward` | `float` | Standard deviation of episode rewards |
| `shap_values` | `np.ndarray` | SHAP values derived from a 500-step rollout |

---

## Supported Models

### Classifiers (10)

| Algorithm | Key Hyperparameters |
|---|---|
| K-Nearest Neighbours | k = 3 |
| SVM (Linear kernel) | C = 0.025 |
| SVM (RBF kernel) | γ = 2, C = 1 |
| Gaussian Process Classifier | RBF kernel, length-scale = 1 |
| Decision Tree | max\_depth = 5 |
| Random Forest | max\_depth = 5, n\_estimators = 10 |
| MLP Classifier | α = 1, max\_iter = 1000 |
| AdaBoost | defaults |
| Gaussian Naïve Bayes | defaults |
| Quadratic Discriminant Analysis | defaults |

### Regressors (10)

| Algorithm | Key Hyperparameters |
|---|---|
| K-Nearest Neighbours | k = 3 |
| SVR (Linear kernel) | C = 0.025 |
| SVR (RBF kernel) | γ = 2, C = 1 |
| Gaussian Process Regressor | RBF kernel, length-scale = 1 |
| Decision Tree | max\_depth = 5 |
| Random Forest | max\_depth = 5, n\_estimators = 10 |
| MLP Regressor | α = 1, max\_iter = 1000 |
| AdaBoost Regressor | defaults |
| Bayesian Ridge | defaults |
| SVR (Polynomial kernel) | degree = 3, C = 1 |

---

## RL Action Space Compatibility

Not every SB3 algorithm supports every action space. `bench_mark` handles this automatically — only compatible algorithms are run.

| Algorithm | Box (continuous) | Discrete | MultiDiscrete | MultiBinary |
|---|:---:|:---:|:---:|:---:|
| A2C | ✅ | ✅ | ✅ | ✅ |
| PPO | ✅ | ✅ | ✅ | ✅ |
| TRPO | ✅ | ✅ | ✅ | ✅ |
| DQN | | ✅ | | |
| QRDQN | | ✅ | | |
| DDPG | ✅ | | | |
| SAC | ✅ | | | |
| TD3 | ✅ | | | |
| ARS | ✅ | | | |
| CrossQ | ✅ | | | |
| TQC | ✅ | | | |

The policy network is also auto-selected:

| Observation space | Policy |
|---|---|
| `Dict` | `MultiInputPolicy` |
| 3-D `Box` (image) | `CnnPolicy` |
| Anything else | `MlpPolicy` |

---

## SHAP Feature Importance

Every result object carries a `shap_values` field computed automatically using `shap.KernelExplainer` with a k-means background summary (up to 50 clusters, capped to dataset size).

```python
import shap
import matplotlib.pyplot as plt

result = results[0]  # e.g. RandomForestClassifier
shap.summary_plot(result.shap_values, X_train[:100])
plt.show()
```

For RL results, SHAP values describe how each observation dimension drives the agent's action choices, computed over a 500-step deterministic rollout.

---

## Output & Logs

When `eval_freq > 0`, the RL benchmark writes training logs and best-model checkpoints under `./logs/`:

```
logs/
├── PPO/                  # best model checkpoint (.zip)
├── PPO_results/          # evaluations.npz — timesteps, mean/std rewards
├── A2C/
├── A2C_results/
└── ...
```

Load a saved checkpoint with:

```python
from stable_baselines3 import PPO

best = PPO.load("./logs/PPO/best_model.zip")
```
