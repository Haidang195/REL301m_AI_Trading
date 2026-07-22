"""
Stock NeurIPS2018 Part 2. Train

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introduce how to use FinRL to make data into the gym form environment, and train DRL agents on it.
"""

from __future__ import annotations

import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# %% Part 1. Prepare directories

check_and_make_directories([TRAINED_MODEL_DIR])

# %% Part 2. Build environment

train = pd.read_csv("train_data.csv")
train = train.set_index(train.columns[0])
train.index.names = [""]

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
env_eval, _ = e_train_gym.get_sb_env()  # Evaluation environment

from stable_baselines3.common.vec_env import VecNormalize
env_eval = VecNormalize(env_eval, norm_obs=True, norm_reward=False, clip_obs=10.0)

print(type(env_train))

# %% Part 3. Train DRL Agents

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True
if_using_tqc = True
if_using_rppo = True

TOTAL_TIMESTEPS = 50000 # Update total timesteps to 50,000 for faster training

import torch.nn as nn

def train_agent(agent_name):
    # ARCH1: Upgrade network size to 256x256 (from 64x64) and use GELU activation
    # Defining it inside the function prevents SB3 from mutating a shared global dictionary
    if agent_name == "rppo":
        current_policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]), 
            activation_fn=nn.GELU,
        )
    else:
        current_policy_kwargs = dict(
            net_arch=[256, 256],           
            activation_fn=nn.GELU,
        )
    
    agent = DRLAgent(env=env_train)
    model = agent.get_model(agent_name, policy_kwargs=current_policy_kwargs)
    
    tmp_path = RESULTS_DIR + f"/{agent_name}"
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    eval_callback = EvalCallback(
        env_eval,
        best_model_save_path=f"{TRAINED_MODEL_DIR}/best_{agent_name}/",
        n_eval_episodes=1,
        eval_freq=10_000,
        verbose=0,
    )
    
    print(f"--- Training {agent_name.upper()} ---")
    trained = agent.train_model(
        model=model, 
        tb_log_name=agent_name, 
        total_timesteps=TOTAL_TIMESTEPS,
        callbacks=[eval_callback]
    )
    
    # Save the model and its VecNormalize statistics
    model_path = TRAINED_MODEL_DIR + f"/agent_{agent_name}"
    trained.save(model_path)
    
    vec_env = model.get_vec_normalize_env()
    if vec_env is not None:
        vec_env.save(model_path + "_vecnormalize.pkl")
        
    return trained

if if_using_a2c:
    trained_a2c = train_agent("a2c")
if if_using_ddpg:
    trained_ddpg = train_agent("ddpg")
if if_using_ppo:
    trained_ppo = train_agent("ppo")
if if_using_td3:
    trained_td3 = train_agent("td3")
if if_using_sac:
    trained_sac = train_agent("sac")
if if_using_tqc:
    try:
        trained_tqc = train_agent("tqc")
    except ValueError:
        print("TQC not available (install sb3-contrib)")
if if_using_rppo:
    try:
        trained_rppo = train_agent("rppo")
    except ValueError:
        print("RecurrentPPO not available (install sb3-contrib)")

print("All agents trained and saved to", TRAINED_MODEL_DIR)
