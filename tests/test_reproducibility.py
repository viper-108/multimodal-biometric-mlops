from src.bioml.utils.reproducibility import seed_everything


def test_seed_everything_sets_hashseed():
    rep = seed_everything(123, deterministic=True)
    assert rep.seed == 123
    assert rep.pythonhashseed == "123"
