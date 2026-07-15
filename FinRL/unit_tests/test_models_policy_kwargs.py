import types

from finrl.agents.stablebaselines3 import models


class DummyEnv:
    def __init__(self):
        self.observation_space = types.SimpleNamespace()
        self.action_space = types.SimpleNamespace(shape=(2,))


def test_get_model_merges_policy_kwargs(monkeypatch):
    captured = {}

    class DummyModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(models, "MODELS", {"tqc": DummyModel})
    monkeypatch.setattr(models, "MODEL_KWARGS", {"tqc": {"policy_kwargs": {"n_quantiles": 25}}})
    monkeypatch.setattr(models, "VecNormalize", lambda env, **kwargs: env)

    agent = models.DRLAgent(DummyEnv())
    agent.get_model("tqc", policy_kwargs={"net_arch": [256, 256]})

    assert captured["policy_kwargs"]["net_arch"] == [256, 256]
    assert captured["policy_kwargs"]["n_quantiles"] == 25
