# directory
from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2014-01-06"
TRAIN_END_DATE = "2026-03-31"

TEST_START_DATE = "2026-04-01"
TEST_END_DATE = "2026-06-25"

TRADE_START_DATE = "2026-04-01"
TRADE_END_DATE = "2026-06-25"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


# Model Parameters
# ── A2C: tăng n_steps từ 5→128 (giảm variance), normalize advantage, thêm GAE ─
A2C_PARAMS = {
    "n_steps": 128,              # was 5 — more steps → lower variance estimates
    "ent_coef": 0.005,           # was 0.01 — less entropy → more exploitation
    "learning_rate": 0.0003,     # was 0.0007 — smaller LR for stability
    "gae_lambda": 0.95,          # GAE lambda for advantage smoothing
    "normalize_advantage": True, # normalize advantage to zero-mean unit-var
    "max_grad_norm": 0.5,        # gradient clipping
    "vf_coef": 0.5,              # critic loss weight
}

# ── PPO: thêm gSDE (state-dependent exploration) + n_epochs + tune LR ─────────
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,           # was 0.01
    "learning_rate": 0.0001,     # was 0.00025 — lower for volatile stock data
    "batch_size": 128,           # was 64 — larger batch → more stable gradients
    "n_epochs": 10,              # reuse each rollout buffer 10x
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "use_sde": True,             # gSDE: noise adapts to state (better exploration)
    "sde_sample_freq": 4,        # re-sample noise every 4 steps
}

# ── DDPG: tăng buffer 50K→500K, giảm LR, thêm train_freq ─────────────────────
DDPG_PARAMS = {
    "batch_size": 256,           # was 128
    "buffer_size": 500_000,      # was 50000 — 10x bigger, remembers longer
    "learning_rate": 3e-4,       # was 0.001 — lower LR for stock data
    "tau": 0.005,                # soft-update coefficient
    "train_freq": 4,             # update every 4 env steps (not every step)
    "gradient_steps": 4,         # 4 gradient updates per train call
}

# ── TD3: tune buffer, LR, decay noise setup ───────────────────────────────────
TD3_PARAMS = {
    "batch_size": 256,           # was 100
    "buffer_size": 1_000_000,    # unchanged — already 1M
    "learning_rate": 3e-4,       # was 0.001
    "tau": 0.005,
    "policy_delay": 2,           # actor updates every 2 critic steps (default)
    "target_policy_noise": 0.2,  # smoothing noise on target actions
    "target_noise_clip": 0.5,
    "train_freq": 4,
    "gradient_steps": 4,
    "action_noise": "decaying",  # automatically decays sigma 0.1 -> 0.01 over 500k steps
}

# ── SAC: tăng buffer 100K→1M, use_sde, warmup lâu hơn ────────────────────────
SAC_PARAMS = {
    "batch_size": 256,           # was 64
    "buffer_size": 1_000_000,    # was 100000 — 10x bigger
    "learning_rate": 3e-4,       # was 0.0001
    "learning_starts": 5_000,    # was 100 — collect more data before training
    "ent_coef": "auto",          # was "auto_0.1" — fully automatic temperature
    "tau": 0.005,
    "use_sde": True,             # gSDE: adaptive exploration
    "sde_sample_freq": 8,        # re-sample noise every 8 steps
}

# ── TQC: Advanced Distributional version of SAC (requires sb3-contrib) ───────
TQC_PARAMS = {
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "learning_rate": 3e-4,
    "learning_starts": 5_000,
    "tau": 0.005,
    "policy_kwargs": {
        "n_quantiles": 25,                   # number of quantiles per critic
        "n_critics": 2,                      # twin critics
        "top_quantiles_to_drop_per_net": 2,  # drop top 2 to be conservative
    },
    "use_sde": True,
}

# ── RPPO: Recurrent PPO with LSTM (requires sb3-contrib) ──────────────────────
RPPO_PARAMS = {
    "n_steps": 2048,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "gae_lambda": 0.95,
    # RPPO uses a different policy architecture specifically for LSTM
    "policy_kwargs": {
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    }
}

ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
