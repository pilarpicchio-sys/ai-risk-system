import numpy as np


def compute_var(quantile_preds, alpha=0.1):
    """
    VaR = quantile inferiore
    """
    return quantile_preds[alpha]


def compute_cvar(quantile_preds, alpha=0.1):
    """
    CVaR approssimato usando quantili disponibili
    """
    q_low = quantile_preds[alpha]
    q_mid = quantile_preds[0.5]

    return (q_low + q_mid) / 2


def compute_expected_return(quantile_preds):
    """
    Expected return (approssimazione)
    """
    q_low = quantile_preds[0.1]
    q_mid = quantile_preds[0.5]
    q_high = quantile_preds[0.9]

    return (q_low + q_mid + q_high) / 3