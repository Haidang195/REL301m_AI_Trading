import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class StockAttentionExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for FinRL using Multi-Head Attention.
    Instead of treating 30 stocks as a flat vector, this architecture:
    1. Encodes each stock's features (price, shares, MACD, RSI, etc.) independently.
    2. Uses Self-Attention across the 30 stocks to learn market correlations.
    3. Combines the result with global market indicators (cash, turbulence).
    """
    
    def __init__(self, observation_space: gym.Space, n_stocks: int = 30, n_indicators: int = 8, embed_dim: int = 32, n_heads: int = 4):
        # We calculate the final feature dimension to inform the RL policy
        features_dim = embed_dim * n_stocks + 3  # +3 for global features: cash, turbulence, turbulence_bool
        super().__init__(observation_space, features_dim=features_dim)

        self.n_stocks = n_stocks
        self.n_indicators = n_indicators

        # 1. Per-stock encoder
        # Features per stock: price (1), shares (1), cooldown (1) + tech indicators (n_indicators)
        # Total per stock: 3 + n_indicators
        self.stock_encoder = nn.Sequential(
            nn.Linear(3 + n_indicators, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 2. Cross-stock Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )

        # 3. Market context encoder
        # Features: cash, turbulence, turbulence_bool
        self.market_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]

        # In FinRL, the default state space is ordered as:
        # [cash (1)] + [prices (30)] + [shares (30)] + [MACD (30), RSI (30), ...] + [turbulence (1)] + [turbulence_bool (1)]
        # We must carefully slice this according to how env_stocktrading.py builds the state.
        # Assuming standard FinRL observation structure:
        # cash: index 0
        cash = obs[:, 0:1]
        
        # prices: 1 to 1+n_stocks
        prices = obs[:, 1 : 1 + self.n_stocks]
        
        # shares: 1+n_stocks to 1+2*n_stocks
        shares = obs[:, 1 + self.n_stocks : 1 + 2 * self.n_stocks]
        
        # In recent FinRL versions, the state vector length is 1 + 2*stock_dim + tech_indicator_len
        # Let's dynamically find the indicators part
        tech_start = 1 + 2 * self.n_stocks
        tech_end = tech_start + self.n_stocks * self.n_indicators
        tech_features = obs[:, tech_start:tech_end]
        
        # Market features (turbulence) are usually at the end, if included
        # If the state space doesn't have turbulence, we'll just pad it for safety
        if obs.shape[1] > tech_end:
            market_extra = obs[:, tech_end:]
            # Ensure it is exactly 2 elements (turb, turb_bool)
            if market_extra.shape[1] < 2:
                market_extra = torch.cat([market_extra, torch.zeros(batch_size, 2 - market_extra.shape[1], device=obs.device)], dim=1)
            elif market_extra.shape[1] > 2:
                market_extra = market_extra[:, :2]
        else:
            market_extra = torch.zeros(batch_size, 2, device=obs.device)

        market = torch.cat([cash, market_extra], dim=1)  # [B, 3]

        # Reshape tech features to [B, n_stocks, n_indicators]
        tech_reshaped = tech_features.view(batch_size, self.n_stocks, self.n_indicators)

        # Stack per-stock features: [B, 30, 3 + n_indicators]
        per_stock = torch.cat([
            prices.unsqueeze(-1), 
            shares.unsqueeze(-1), 
            torch.zeros_like(prices).unsqueeze(-1), # padding for 'cooldown' or extra feature
            tech_reshaped
        ], dim=-1)

        # Encode: [B, 30, embed_dim]
        stock_embeds = self.stock_encoder(per_stock)

        # Attention: [B, 30, embed_dim]
        attn_out, _ = self.attention(stock_embeds, stock_embeds, stock_embeds)
        
        # Residual connection
        attn_out = attn_out + stock_embeds

        # Flatten: [B, 30 * embed_dim]
        stock_feat = attn_out.reshape(batch_size, -1)

        # Market context: [B, 3]
        market_feat = self.market_encoder(market)

        # Concat: [B, 30*embed_dim + 3]
        final_features = torch.cat([stock_feat, market_feat], dim=-1)
        
        return final_features

# Example usage in training script:
# from finrl.agents.stablebaselines3.custom_policy import StockAttentionExtractor
# policy_kwargs = dict(
#     features_extractor_class=StockAttentionExtractor,
#     features_extractor_kwargs=dict(n_stocks=30, n_indicators=8),
#     net_arch=[256, 128],
#     activation_fn=nn.GELU
# )
