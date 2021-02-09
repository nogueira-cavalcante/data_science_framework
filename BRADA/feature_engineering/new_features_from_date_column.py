"""Criação de novas colunas a partir do campo data."""


def new_features_from_date_column(df, new_columns, date_col=None):
    """
    Criação de novas colunas a partir do campo data.

    Esta função usando os recursos datetime (dt) do pandas.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame contendo a coluna data.
    new_columns : list
        Lista de strings contendo os nomes das novas colunas. Atualmente, é
        possível criar as seguintes colunas a partir do campo datetime:
            - 'dia': integer, correspondente ao dia do mês;
            - 'mes': integer, correspondente ao mês do ano;
            - 'nome_mes': string, correspondente ao mês do ano;
            - 'semana_do_ano': integer, correspondente à semana do ano;
            - 'ano': integer, correspondente ao ano;
            - 'nome_do_dia': string, correspondente ao dia da semana;
            - 'ordem_do_dia_na_semana': integer, onde 0 começa na Segunda;
            - 'ordem_do_dia_no_ano': integer, onde 1 começa no primeiro dia do
                                     ano;
            - 'hora': time, parte da hora do campo datetime;
            - 'normalizacao': date, parte da data do campo datetime;
            - 'ano_bissexto': boolean, se o ano no campo datetime é bissexto;
            - 'primeiro_dia_mes': boolean, se a data é o primeiro dia do mês;
            - 'ultimo_dia_mes': boolean, se a data é o último dia do mês;
    date_col : string, optional
        Nome do campo datetime no DataFrame de entrada. Se não for informado
        será considerado o índice do DataFrame. O valor padrão é None.

    Raises
    ------
    Exception
        - Quando o coluna informada ou o índice do DataFrame não estiverem no
          formato datetime;
        - Quando o nome para a nova coluna não estiver implementada na função.

    Returns
    -------
    df_temp : pandas DataFrame
        DataFrame contendo as colunas do DataFrame de entrada junto com as
        novas colunas criadas especificadas em new_columns.

    """
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    df_temp = df.copy()

    is_index_reseted = False

    if date_col is None:
        if not is_datetime(df_temp.index):
            raise Exception('O índice do DataFrame deve estar no ' +
                            'formato datetime.')
        else:
            date_col = df_temp.index.name
            df_temp.reset_index(inplace=True)
            is_index_reseted = True
    else:
        if not is_datetime(df_temp[date_col]):
            raise Exception('A coluna especificada deve estar no ' +
                            'formato datetime.')

    if new_columns is None:
        new_columns = []

    for new_col in new_columns:

        if new_col == 'dia':
            df_temp.loc[:, 'dia'] = df_temp[date_col].dt.day

        elif new_col == 'mes':
            df_temp.loc[:, 'mes'] = df_temp[date_col].dt.month

        elif new_col == 'nome_mes':
            df_temp.loc[:, 'nome_mes'] = df_temp[date_col].dt.month_name()

        elif new_col == 'semana_do_ano':
            semana_do_ano = df_temp[date_col].dt.isocalendar().week
            df_temp.loc[:, 'semana_do_ano'] = semana_do_ano

        elif new_col == 'ano':
            df_temp.loc[:, 'ano'] = df_temp[date_col].dt.year

        elif new_col == 'nome_do_dia':
            df_temp.loc[:, 'nome_do_dia'] = df_temp[date_col].dt.day_name()

        elif new_col == 'ordem_do_dia_na_semana':
            ordem_do_dia_na_semana = df_temp[date_col].dt.dayofweek
            df_temp.loc[:, 'ordem_do_dia_na_semana'] = ordem_do_dia_na_semana

        elif new_col == 'ordem_do_dia_no_ano':
            ordem_do_dia_no_ano = df_temp[date_col].dt.dayofyear
            df_temp.loc[:, 'ordem_do_dia_no_ano'] = ordem_do_dia_no_ano

        elif new_col == 'hora':
            df_temp.loc[:, 'hora'] = df_temp[date_col].dt.hour

        elif new_col == 'normalizacao':
            df_temp.loc[:, 'normalizacao'] = df_temp[date_col].dt.normalize()

        elif new_col == 'ano_bissexto':
            df_temp.loc[:, 'ano_bissexto'] = df_temp[date_col].dt.is_leap_year

        elif new_col == 'primeiro_dia_mes':
            primeiro_dia_mes = df_temp[date_col].dt.is_month_start
            df_temp.loc[:, 'primeiro_dia_mes'] = primeiro_dia_mes

        elif new_col == 'ultimo_dia_mes':
            ultimo_dia_mes = df_temp[date_col].dt.is_month_end
            df_temp.loc[:, 'ultimo_dia_mes'] = ultimo_dia_mes

        else:
            raise Exception('Não é possível criar uma coluna nova a partir ' +
                            'no parâmetro informado (' + new_col + '). ' +
                            'Veja a documentação da função para revisar ' +
                            'quais novas colunas podem ser criadas a partir ' +
                            'de um campo data.')

    if is_index_reseted:
        df_temp.set_index(date_col, inplace=True)

    return df_temp
