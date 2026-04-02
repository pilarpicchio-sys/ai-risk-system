import numpy as np


def compute_drawdown(equity):
    """
    Max drawdown
    """
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown, drawdown.min()


def buy_and_hold(returns):
    """
    Benchmark semplice
    """
    equity = (1 + returns).cumprod()
    return equity