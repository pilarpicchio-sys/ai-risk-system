import lightgbm as lgb


def train_model(X, y):
    """
    Allena un modello base LightGBM
    """

    
    model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    min_child_samples=10
    )

    model.fit(X, y)

    return model


def predict(model, X):
    """
    Predizione probabilità
    """
    probs = model.predict_proba(X)
    return probs