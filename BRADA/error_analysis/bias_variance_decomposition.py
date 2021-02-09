"""Decomposição Bias-Variance."""


def bias_variance_decomposition(modelo, X_train, y_train, X_test, y_test,
                                loss='mse', num_rounds=3, random_seed=10):
    """
    Decomposição bias-variance.

    Parameters
    ----------
    modelo : object
        Um objeto do tipo classifier ou regressor ou uma classe implementando
        ambos os métidos fit e predict, similares ao de API scikit-learn.
    X_train : array-like
        Amostra de treino para as amostras bootstrap para a decomposição
        bias-variance.
    y_train : array-like
        Targets associado à amostra de treino X_train.
    X_test : array-like
        Amostra de teste para computar o loss average, bias e variance.
    y_test : array-like
        Targets associados à amostra de teste X_test.
    loss : string, optional
         Função loss para a decompisição bias-variance. Atualmente os valores
         permitidos são '0-1_loss' e 'mse'. O valor padrão é 'mse'.
    num_rounds : integer, optional
        Número de rodadas bootstrap para a decomposição bias-variance.
        O valor padrão é 3.
    random_seed : integer, optional
        Semente aleatória para a amostragem bootstrap. O valor padrão é 10.

    Returns
    -------
    bias : flot
        Valor de bias.

    var : float
        Valor de variância.

    """
    from mlxtend.evaluate import bias_variance_decomp

    mse, bias, var = bias_variance_decomp(modelo, X_train.values,
                                          y_train.ravel(), X_test.values,
                                          y_test.ravel(),
                                          loss=loss,
                                          num_rounds=num_rounds,
                                          random_seed=random_seed)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f' % var)

    return bias, var
