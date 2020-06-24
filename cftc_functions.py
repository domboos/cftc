import pandas as pd
import sqlalchemy as sq

engine1 = sq.create_engine(
    "postgresql+psycopg2://pren@iwa-backtest:pren@iwa-backtest.postgres.database.azure.com:5432/postgres")

def gets(engine, type, data_tab='data', desc_tab='cot_desc', series_id=None, bb_tkr=None, bb_ykey='COMDTY',
         start_dt='1900-01-01', end_dt='2100-01-01', constr=None):

    if constr is None:
        constr = ''
    else:
        constr = ' AND ' + constr

    if series_id is None:
        if desc_tab == 'cot_desc':
            series_id = pd.read_sql_query("SELECT cot_id FROM cftc.cot_desc WHERE bb_tkr = '" + bb_tkr +
                                          "' AND bb_ykey = '" + bb_ykey + "' AND cot_type = '" + type + "'", engine1)
        else:
            series_id = pd.read_sql_query("SELECT px_id FROM cftc.fut_desc WHERE bb_tkr = '" + bb_tkr +
                                          "' AND bb_ykey = '" + bb_ykey + "' AND data_type = '" + type + "'", engine1)
        series_id = str(series_id.values[0][0])
    else:
        series_id = str(series_id)

    print(series_id)

    h_1 = " WHERE px_date >= '" + str(start_dt) +  "' AND px_date <= '" + str(end_dt) + "' AND px_id = "
    h_2 = series_id + constr + " order by px_date"
    fut = pd.read_sql_query('SELECT px_date, qty FROM cftc.' + data_tab + h_1 + h_2, engine, index_col='px_date')
    return fut

# test
hh = gets(engine1, 'agg_open_interest', data_tab='vw_data', bb_tkr='KC')
print(hh)