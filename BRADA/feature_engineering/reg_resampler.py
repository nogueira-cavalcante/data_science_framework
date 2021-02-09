"""Reamostragem de variáveis contínuas para a amostra de treino."""


def resampler_regression(X_train, y_train, target, bins=5,
                         balanced_binning=False, random_state=10):
    """
    Reamostragem de variáveis contínuas para a amostra de treino.

    Esta função utiliza o SMOTE para a reamostragem dos dados do
    DataFrame de entrada.

    Parameters
    ----------
    X_train : pandas DataFrame
        DataFrame contendo as features de treino.
    y_train : pandas DataFrame
        DataFrame contendo os targets de treino.
    target : string
        Nome do target no DataFrame de entrada.
    bins : integer, opcional
        Quantidade de bins para a reamostragem. O valor padrão é 5.
    balanced_binning : boolean, opcional
        Balanceamento nos bins. O valor padrão é False.
    random_state : integer ou RandomState, opcional
        Gerador numérico para ser usado para a geração da amostra
        aleatória. O valor padrão é 10.

    Returns
    -------
    X_train_res : pandas DataFrame
        DataFrame contendo as features reamostradas.
    y_train_res : pandas DataFrame
        DataFrame contendo o target reamostrada.

    """
    import pandas as pd
    from reg_resampler import resampler
    from imblearn.over_sampling import SMOTE

    base = pd.concat([X_train, y_train], axis=1)

    rs = resampler()

    Y_classes = rs.fit(base, target=target, bins=bins,
                       balanced_binning=balanced_binning)

    smote = SMOTE(random_state=random_state)

    X_train_res, y_train_res = rs.resample(smote, base, Y_classes)

    return X_train_res, y_train_res
