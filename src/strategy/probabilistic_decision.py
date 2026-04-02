import numpy as np


def make_prob_decision(probs, up_threshold=0.55, down_threshold=0.55):
    """
    Decisione basata su probabilità
    probs shape: (n_samples, n_classes)
    """

    decisions = []

    for p in probs:

        p_down = p[0] + p[1]
        p_up = p[4] + p[5]

        if p_up > up_threshold:
            decisions.append("LONG")

        elif p_down > down_threshold:
            decisions.append("FLAT")

        else:
            decisions.append("FLAT")

    return decisions