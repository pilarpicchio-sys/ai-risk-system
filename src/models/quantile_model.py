import lightgbm as lgb


def train_quantile_model(X, y, quantile):
    """
    Allena modello per un singolo quantile
    """

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1
    )

    model.fit(X, y)

    return model


def train_multiple_quantiles(X, y, quantiles=None):
    """
    Allena più modelli (uno per quantile)
    """

    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    models = {}

    for q in quantiles:
        models[q] = train_quantile_model(X, y, q)

    return models


def predict_quantiles(models, X):
    """
    Predice tutti i quantili
    """

    preds = {}

    for q, model in models.items():
        preds[q] = model.predict(X)

    return preds