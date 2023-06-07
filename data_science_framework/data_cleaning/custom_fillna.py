"""Preenchimento personalizado de nulos."""


def custom_fillna(X_train, lista_variaveis, forma, valor=None):
    """
    Preenchimento personalisado de valores nulos.

    Preenchimento de valores nulos de acordo com a forma informada.

    Parameters
    ----------
    X_train : pandas DataFrame
        DataFrame contendo as features.
    lista_variaveis : list
        Lista de strings contendo os nomes das colunas para o preenchimento.
    forma : string
        Método para a imputação:
            - "mean";
            - "median";
            - "arbitrary";
            - "endtail";
            - "categorical";
            - "RandomSample"
            - "missing_indicator".
    valor : string, optional
        Valor para ser imputado (usado apenas quando forma="arbitrary").
        O valor padrão é None.

    Raises
    ------
    Exception
        - Quando a forma informado é inválida.

    Returns
    -------
    train_t : numpy array
        Amostra já tratada com a imputação.
    imputer : objeto feature_engine
        Objeto usado na imputação.

    """
    from feature_engine.imputation import MeanMedianImputer
    from feature_engine.imputation import ArbitraryNumberImputer
    from feature_engine.imputation import EndTailImputer
    from feature_engine.imputation import CategoricalImputer
    from feature_engine.imputation import RandomSampleImputer
    from feature_engine.imputation import AddMissingIndicator

    print('Método aplicado:')
    print(forma)
    print('Imputação nas variáveis:')
    print(lista_variaveis)

    if forma == 'mean':
        imputer = MeanMedianImputer(imputation_method='mean',
                                    variables=lista_variaveis)
    elif forma == 'median':
        imputer = MeanMedianImputer(imputation_method='median',
                                    variables=lista_variaveis)
    elif forma == 'arbitrary':
        imputer = ArbitraryNumberImputer(arbitrary_number=valor,
                                         variables=lista_variaveis)
    elif forma == 'endtail':
        imputer = EndTailImputer(imputation_method='gaussian', tail='right',
                                 fold=3, variables=lista_variaveis)
    elif forma == 'categorical':
        imputer = CategoricalImputer(variables=lista_variaveis)

    elif forma == 'RandomSample':
        imputer = RandomSampleImputer(random_state=lista_variaveis,
                                      seed='observation',
                                      seeding_method='add')
    elif forma == 'missing_indicator':
        imputer = AddMissingIndicator(variables=lista_variaveis)

    else:
        raise Exception('O método de imputação informado é inválido (' +
                        forma + ')')

    imputer.fit(X_train)
    train_t = imputer.transform(X_train)

    return train_t, imputer
