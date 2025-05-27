import numpy as np
import torch
import pytest

from peak_detection.curve_fitting import (
    LinearModel, ExpModel, BiExpModel, BBModel, model_fitting
)

@pytest.mark.parametrize("ModelClass", [LinearModel, ExpModel, BiExpModel, BBModel])
def test_model_forward(ModelClass):
    y = np.linspace(1, 10, 10)
    model = ModelClass(y)
    x = torch.linspace(1, 10, 10).unsqueeze(1)
    out = model(x)
    assert out.shape[0] == 10

def test_model_fitting_runs():
    x = np.linspace(0, 1, 30)
    y = np.exp(-x) + 0.1 * np.random.randn(30)
    model = ExpModel(y)
    device = torch.device("cpu")
    x_fit, y_fit, params = model_fitting(x, y, model, device, epoch=5, lr=0.01)
    assert x_fit.shape[0] == 100
    assert y_fit.shape[0] == 100
    assert isinstance(params, list)
