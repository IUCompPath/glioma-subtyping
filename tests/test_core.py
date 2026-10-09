import numpy as np
import pandas as pd

from ensemble_script import average_predictions, class_auc
from utils.utils import safe_auc


def test_safe_auc_handles_missing_classes():
    labels = [0, 2, 0, 2]
    probs = np.array([[.8, .1, .1], [.1, .1, .8], [.7, .2, .1], [.2, .1, .7]])
    assert safe_auc(labels, probs, 3) == 1.0
    assert np.isnan(safe_auc([1, 1], probs[:2], 3))


def test_average_predictions_fuses_by_slide():
    a = pd.DataFrame({"slide_id": ["s"], "Y": [1], "Y_hat": [0], "p_0": [.9], "p_1": [.1], "p_2": [0.]})
    b = pd.DataFrame({"slide_id": ["s"], "Y": [1], "Y_hat": [1], "p_0": [.1], "p_1": [.9], "p_2": [0.]})
    out = average_predictions([a, b])
    assert out.loc[0, "p_0"] == .5 and out.loc[0, "Y"] == 1
    assert class_auc(out) != 2  # runs without error


def test_mamba_forward():
    import pytest, torch
    if not torch.cuda.is_available():
        pytest.skip("Mamba kernels need a GPU")
    from models.MambaMIL import MambaMIL
    m = MambaMIL(in_dim=32, n_classes=3, dropout=0.1, act="gelu").cuda()
    logits = m(torch.randn(1, 50, 32).cuda())[0]
    assert logits.shape == (1, 3)
