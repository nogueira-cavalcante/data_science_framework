def separacao_dados_treino_teste(df, sep_type, target_col, features_cols=[],
                                 cat_cont_var_bins=5, test_size=0.2,
                                 random_state=10):

    """
    Separa o DataFrame de entrada em treino e teste, de acordo com o tipo de
    separação escolhido (parâmetro sep_type).

    Args:
        df (pandas DataFrame): DataFrame contendo as features e o target.
        sep_type (string): Tipo de separação treino-teste. É possível
                           escolher os seguintes tipos:
                           - "time-series": O índice do DataFrame de entrada
                                            precisa estar no formato datetime.
                                            Logo, o DaFrame é ordenado pelo
                                            índice, a parte de treino recebe
                                            a parte inicial e o treino a parte
                                            restante, de acordo com o valor em
                                            test_size.
                           - "regression": É criada uma varível categórica
                                           temporárica baseada no target e no
                                           valor em cat_cont_var_bins. Logo,
                                           a separação treino-teste é
                                           estratificada baseada nessa variável
                                           categórica temporária.
                           - "classification": A separação treino-teste é
                                               estratificada baseada no target.
        target_col (string): Nome do target.
        features_cols (list): Lista contendo os nomes das features.
        cat_cont_var_bins (integer): Quantidade de bins para a divisão do
                                     target contínuo e assim criar a separação
                                     treino-teste estratificada. Usada apenas
                                     quando sep_type="regression".
        test_size (float): Tamanho da amostra de teste, dado em fração do total
                           de registros do DataFrame de entrada.
        random_state (integer): Semente usada para a separação treino-teste
                                (apenas para os modos "regression" e
                                "classification").

    Return:
        Total de quatro DataFrames (X_train, y_train, X_test, y_test).

    Imports:
        import pandas as pd
        from sklearn.model_selection import train_test_split
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split

    if (test_size <= 0) or (test_size >= 0.5):
        raise Exception('A fração da amostra de teste deve ser maior que' +
                        ' 0.0 e menor que 0.5.')

    if len(features_cols) == 0:
        features_cols = list(df.drop(columns=target_col).columns)

    df_temp = df[[target_col] + features_cols].copy()

    if sep_type == 'classification':

        train, test = train_test_split(df_temp,
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=True,
                                       stratify=df_temp[target_col])

    elif sep_type == 'regression':

        labels_cat = []
        for i in range(cat_cont_var_bins):
            labels_cat.append(i)
        df_temp.loc[:, 'categorical_target'] = pd.cut(df_temp[target_col],
                                                      bins=cat_cont_var_bins,
                                                      labels=labels_cat)
        train, test = train_test_split(df_temp,
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=True,
                                       stratify=df_temp['categorical_target'])

        train = train[[target_col] + features_cols]
        test = test[[target_col] + features_cols]

    elif sep_type == 'time_series':

        index_type = str(df_temp.index.dtype)
        if ('datetime' not in index_type) and ('period' not in index_type):
            raise Exception('O indice do DataFrame deve estar no ' +
                            'formato datetime ou period.')

        df_temp.sort_index(inplace=True)

        train_size = int(df_temp.shape[0] * (1.-test_size))
        train = df_temp.iloc[:train_size, :]
        test = df_temp.iloc[train_size:, :]

    else:
        raise Exception('Valor inválido no parâmetro sep_type ' +
                        '(somente "classification", "regression" e ' +
                        '"time_series" são permitidos)')

    X_train = train[features_cols].copy()
    y_train = train[[target_col]].copy()
    X_test = test[features_cols].copy()
    y_test = test[[target_col]].copy()

    return X_train, y_train, X_test, y_test
