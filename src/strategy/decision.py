def make_decision(var, cvar, exp_ret,
                  var_threshold=-0.01,
                  cvar_threshold=-0.02):
    """
    Decisione conservativa:
    LONG solo se rischio controllato
    """

    decisions = []

    for v, c, e in zip(var, cvar, exp_ret):

        if (e > 0) and (v > var_threshold) and (c > cvar_threshold):
            decisions.append("LONG")
        else:
            decisions.append("FLAT")

    return decisions