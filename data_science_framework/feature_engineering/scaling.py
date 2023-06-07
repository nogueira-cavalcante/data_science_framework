"""Escalonamento dos dados."""


def scaling(df, scaling_methods=['min_max'], kwargs=None):
    """
    Escalonamento dos dados do DataFrame de entrada.

    Esta função utiliza os métodos de escalonamento do scikit-learn.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo os dados para serem escalanados.
    scaling_methods : list, optional
        Lista de string dos métodos de escalonamento. O valor padrão é
        ['min_max']. Atualmente esta função comporta os seguintes métodos:
            - 'min_max';
            - 'max_abs':
            - 'standard';
            - 'robust';
            - 'normalizer';
            - 'quantile';
            - 'power_transform'.
    kwargs : list, optional
        Lista de dicionários de parâmetros para ser passado para os métodos
        escolhidos. Caso não seja informado, serão considerados os valores
        padrões para os parâmetros dos métodos escolhidos. O valor padrão é
        None.

    Raises
    ------
    Exception
        - Quando o método escolhido não se encontra implementado ou não existe.

    Returns
    -------
    df_temp : pandas DataFrame
        DataFrame escalonado.
    pipe : object
        Objeto pipe contendo os objetos dos método escolhidos.

    """
    import pandas as pd
    from sklearn.pipeline import Pipeline

    list_scaler = []
    for i in range(len(scaling_methods)):

        scaling_method = scaling_methods[i]

        if scaling_method == 'min_max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler

        elif scaling_method == 'max_abs':
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler

        elif scaling_method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler

        elif scaling_method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler

        elif scaling_method == 'normalizer':
            from sklearn.preprocessing import Normalizer
            scaler = Normalizer

        elif scaling_method == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            scaler = QuantileTransformer

        elif scaling_method == 'power_transform':
            from sklearn.preprocessing import PowerTransformer
            scaler = PowerTransformer

        else:
            raise Exception('Método inválido (' + scaling_method + ')')

        if (kwargs is None) or (kwargs[i] is None):
            scaler = scaler()
        else:
            scaler = scaler(kwargs[i])

        list_scaler.append((scaling_method, scaler))

    pipe = Pipeline(list_scaler)

    df_temp = pipe.fit_transform(df)
    df_temp = pd.DataFrame(data=df_temp, columns=df.columns)

    return df_temp, pipe
