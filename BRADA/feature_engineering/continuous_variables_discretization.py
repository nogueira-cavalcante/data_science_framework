"""Discretização de variáveis contínuas."""


def continuous_variables_discretization(df, continuos_columns=None, bins=5):
    """
    Discretização de variáveis contínuas.

    Esta função cria novas colunas categóricas a partir de colunas contínuas.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo as colunas contínuas pare serem discretizdas.
    continuos_columns : list, optional
        Lista de strings contendo os nomes das colunas contínuas para serem
        discretizados. Caso não seja informado serão consideradas todas as
        colunas do DataFrame. O valor padrão é None.
    bins : integer, optional
        Quantidade de bins para gerar as novas colunas categóricas. O valor
        padrão é 5.

    Returns
    -------
    df_temp : pandas DataFrame
        DataFrame contendo as colunas contínuas discretizadas, além das colunas
        do DataFrame de entrada.

    """
    import numpy as np
    import pandas as pd

    df_temp = df.copy()

    if continuos_columns is None:
        continuos_columns = df_temp.columns

    for col in continuos_columns:

        cat_col = pd.cut(df_temp[col], bins=bins)
        cat_col = cat_col.apply(lambda x: np.mean([x.left, x.right]))
        new_col_name = col + '_categorical'
        df_temp.loc[:, new_col_name] = cat_col

    return df_temp
