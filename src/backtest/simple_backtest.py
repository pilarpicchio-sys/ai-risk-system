
import numpy as np


def run_backtest(returns, sizes):
    """
    Backtest con position sizing
    """

    strategy_returns = []

    for i in range(len(returns)):

        if i == 0:
            strategy_returns.append(0)
            continue

        size = sizes[i - 1]  # shift!

        strategy_returns.append(size * returns[i])

    strategy_returns = np.array(strategy_returns)

    equity = (1 + strategy_returns).cumprod()

    return strategy_returns, equity