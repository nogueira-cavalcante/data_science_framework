"""Tratamento de outliers."""


def outlier_treatment(df, columns=None, method='removal',
                      remove_outlier_column=True, constant_value=0):
    """
    Tratamento de outliers.

    Função de tratamento de outliers, sendo necessário o uso da função
    outlier_detection antes.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo as colunas já identificadas como outliers.
    columns : list, optional
        Lista de colunas onde os outliers já foram identificados.
        O valor padrão é None. Caso não seja informado a função buscará
        automaticamente as colunas onde os outliers já foram identificados.
    method : string, optional
        Nome do método a ser usado para tratar os outliers. O valor padrão é
       'removal'. Os métodos implementados nesta função são:
           - 'removal': Remoção das linhas do outliers;
           - 'constant': Subtituição do outlier pelo valor definido em
                         constant_value;
           - 'min': Subtituição do outlier pelo valor mínimo da coluna;
           - 'max': Subtituição do outlier pelo valor máximo da coluna;
           - 'mean': Subtituição do outlier pelo valor médio da coluna;
           - 'median': Subtituição do outlier pelo valor da medianda da coluna;
           - 'mode': Subtituição do outlier pela moda da coluna;
           - 'bfill': Subtituição do outlier pelo valor posterior ao outlier;
           - 'ffill': Subtituição do outlier pelo valor anterior ao outlier;
           - 'interpolate': Interpolação linear (somente para séries temporais
                            onde o index é datetime).
    remove_outlier_column : boolena, optional
        Caso True, irá remover do DataFrame de entrada as colunas que
        identificam os outliers. O valor padrão é True.
    constant_value : float, optional
        Valor constante para ser usado no método 'constant'. O valor padrão é
        0.

    Raises
    ------
    Exception
        Quando o método de imputação informado não se encontra implementado ou
        não existe.

    Returns
    -------
    df_temp : pandas DataFrame
        DataFrame contendo todas as colunas do DataFrame de entrada com os
        outliers tratados, confome o método escolhido.

    """
    import numpy as np

    print('Tratamento de outliers...')
    print('Método: ' + method)

    df_temp = df.copy()

    if columns is None:
        outlier_columns = df_temp.filter(regex='_outlier_flag').columns

    else:
        outlier_columns = []
        for col in df_temp[columns].columns:
            outlier_columns.append(col + '_outlier_flag')

    print('Total de colunas a serem analisadas:',
          df_temp[outlier_columns].shape[1])

    for outlier_column in outlier_columns:
        col = outlier_column.replace('_outlier_flag', '')

        df_temp.loc[df_temp[outlier_column] == 1, col] = np.nan

        if method == 'removal':
            df_temp.dropna(subset=[col], inplace=True)

        elif method == 'constant':
            df_temp[col] = df_temp[col].fillna(constant_value)

        elif method == 'min':
            df_temp[col] = df_temp[col].fillna(df_temp[col].min())

        elif method == 'max':
            df_temp[col] = df_temp[col].fillna(df_temp[col].max())

        elif method == 'mean':
            df_temp[col] = df_temp[col].fillna(df_temp[col].mean())

        elif method == 'median':
            df_temp[col] = df_temp[col].fillna(df_temp[col].median())

        elif method == 'mode':
            df_temp[col] = df_temp[col].fillna(df_temp[col].mode().iloc[0])

        elif method == 'bfill':
            df_temp[col] = df_temp[col].fillna(method='bfill')

        elif method == 'ffill':
            df_temp[col] = df_temp[col].fillna(method='ffill')

        elif method == 'interpolate':
            index_type = str(df_temp.index.dtype)
            if 'datetime' not in index_type:
                raise Exception('O indice do DataFrame deve estar no ' +
                                'formato datetime.')
            df_temp.sort_index(inplace=True)

            df_temp[col] = df_temp[col].interpolate(method='linear')

        else:
            raise Exception('Método inválido.')

        if remove_outlier_column:
            df_temp.drop(columns=[outlier_column], inplace=True)

    return df_temp
