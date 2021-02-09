"""Remoção de duplicados."""


def remove_duplicates(df, y):
    """
    Remoção de duplicados no DataFrame de entrada.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo os dados parserem tratados.
    y : pandas Series
        Series contendo o target.

    Returns
    -------
    df : pandas DataFrame
        DataFrame original.
    dfy: pandas Series
        Series concatenada com a DataFrame de entrada removido os duplicados.

    """
    import pandas as pd

    print('Número de registros antes da filtragem:' + str(len(df)))
    df.drop_duplicates(keep='first', inplace=True)
    print('Número de registros depois da filtragem:' + str(len(df)))
    dfy = pd.concat([df, y], axis=1, join='inner')

    return df, dfy.iloc[:, -1]
