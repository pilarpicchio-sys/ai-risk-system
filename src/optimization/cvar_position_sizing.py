import numpy as np


def compute_cvar_sizes(signal, returns, alpha=0.1):
    """
    CVaR-based position sizing

    signal → expected return (array)
    returns → historical returns
    alpha → tail level (0.1 = worst 10%)
    """

    sizes = []

    for i in range(len(signal)):

        # finestra rolling (ultimi 100 giorni)
        if i < 100:
            sizes.append(0.5)
            continue

        window = returns[i-100:i]

        # downside (worst alpha%)
        threshold = np.quantile(window, alpha)
        tail_losses = window[window <= threshold]

        if len(tail_losses) == 0:
            cvar = abs(threshold)
        else:
            cvar = abs(np.mean(tail_losses))

        # 🔥 rischio → scala la posizione
        risk_adjustment = 0.02 / (cvar + 1e-6)

        size = signal[i] * 50 * risk_adjustment

        size = max(0.0, min(size, 1.5))

        sizes.append(size)

    return np.array(sizes)