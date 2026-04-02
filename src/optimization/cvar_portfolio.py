import numpy as np


def compute_cvar_weights(returns_matrix, alpha=0.1):
    """
    returns_matrix: shape (T, N assets)
    """

    n_assets = returns_matrix.shape[1]

    best_score = -np.inf
    best_weights = None

    # grid search semplice
    for w1 in np.linspace(0, 1, 10):
        for w2 in np.linspace(0, 1 - w1, 10):

            w3 = 1 - w1 - w2

            weights = np.array([w1, w2, w3])

            portfolio_returns = returns_matrix @ weights

            threshold = np.quantile(portfolio_returns, alpha)
            tail = portfolio_returns[portfolio_returns <= threshold]

            cvar = np.mean(tail) if len(tail) > 0 else threshold

            expected = np.mean(portfolio_returns)

            score = expected / abs(cvar + 1e-6)

            if score > best_score:
                best_score = score
                best_weights = weights

    return best_weights