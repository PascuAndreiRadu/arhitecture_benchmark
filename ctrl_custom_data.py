import yfinance as yf
import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import gym_trading_env
from stable_baselines3.common.vec_env import DummyVecEnv
import psycopg
import polars as pl
from app import data_parser
import yaml

with psycopg.connect("dbname=ai_dataset user=postgres password=parola host=localhost port=5432") as conn:
    data=pl.read_database("select distinct on (time) * from ticker_data where ticker='AAPL' order by time asc", conn)
from importlib import resources
with resources.files('app.settings').joinpath('config.yaml').open('r') as file:
    config = yaml.safe_load(file)

parser = data_parser.Parser(config,asset='tbd')
data=parser.add_features([data])[0]
data=parser.clean_df([data])[0]
data=parser.scale([data],True)[0]
df=data.to_pandas()


float64_cols = df.select_dtypes(include=['float64']).columns

df[float64_cols] = df[float64_cols].astype('float32')
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


# 3. SPLIT DATA
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"Training on {len(train_df)} candles, Testing on {len(test_df)} candles.")

# 4. SETUP ENVIRONMENT


def make_env(data_df):
    return gym.make(
        "TradingEnv",
        df=data_df,
        positions=[-1, 0, 1], # Short, Neutral, Long
        trading_fees=0.001,   # 0.1% fees
        borrow_interest_rate=0.0003, # Interest for shorting
        windows=5, 
        # Initial cash
        portfolio_initial_value=1_000_000, 
    )



# 5. TRAIN AGENT
print("Training PPO Agent...")
from app.utils import test_arhitecture
tester = test_arhitecture.bench_mark()
results = tester.rl(make_env(train_df),1000000)

best_key_full = max((k for k in results if '_std_reward' in k), key=lambda k: results[k], default=None)

# Strips the suffix if a match was found
best_model = best_key_full.replace('_std_reward', '') if best_key_full else None
print(best_model)
# 6. BACKTEST 
model = results[f'{best_model}_obj']
env_test = make_env(test_df)

obs, _ = env_test.reset()
done = False
truncated = False

print("Running Backtest...")
while not done and not truncated:
    # Predict action
    action, _states = model.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, done, truncated, info = env_test.step(action)

