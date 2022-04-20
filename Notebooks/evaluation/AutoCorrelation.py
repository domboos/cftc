import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from Notebooks.evaluation.getR2 import getResiduals
from cfunctions import engine1


# Autocorrelation result: x_t = a + b* x_(t-1):
# Source_ https://openstax.org/books/introductory-business-statistics/pages/13-2-testing-the-significance-of-the-correlation-coefficient


def autoCorrelationStatistics(model_type_ids: list, cftcVariableName: str, fcastVariableName: str) -> pd.DataFrame:
    for model_type_id in model_type_ids:
        residuals = getResiduals(model_type_id, cftcVariableName, fcastVariableName, fixedStartdate=None,
                                 fixedEndDate=None,
                                 type_='diff')
        bb_tkrs = list(residuals)[1:]

        temp_result: pd.DataFrame = pd.DataFrame(index=bb_tkrs,
                                                 columns=['corr_emp_lag1', 'lag1-tstat', 'corr_emp_lag2', 'lag2-tstat'])
        for bb_tkr in bb_tkrs:
            df = pd.DataFrame(data=residuals[bb_tkr], columns=['diff_'])
            df[f"lag_1"] = df.diff_.shift(1)
            df[f"lag_2"] = df.diff_.shift(2)

            # get Autocorr values:
            corr = df.corr()
            temp_result.loc[bb_tkr, 'corr_emp_lag1'] = corr.loc['diff_', 'lag_1']
            temp_result.loc[bb_tkr, 'corr_emp_lag2'] = corr.loc['diff_', 'lag_2']

            df = df.dropna()
            x1 = sm.add_constant(df['lag_1']).values
            x2 = sm.add_constant(df['lag_2']).values
            y = df['diff_'].values

            mod_ac1 = sm.OLS(y, x1).fit()
            mod_ac2 = sm.OLS(y, x2).fit()

            temp_result.loc[bb_tkr, 'lag1-tstat'] = mod_ac1.tvalues[1]
            temp_result.loc[bb_tkr, 'lag2-tstat'] = mod_ac2.tvalues[1]

        temp_result.to_excel(exclWriter, sheet_name=f"{model_type_id}")
    return temp_result


def autoCorrelationPlots(model_type_ids: list, cftcVariableName: str, fcastVariableName: str):
    for model_type_id in model_type_ids:
        residuals = getResiduals(model_type_id, cftcVariableName, fcastVariableName, type_='diff',
                                 fixedStartdate=None, fixedEndDate=None)

        bb_tkrs = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things", engine1).bb_tkr
        oot = pd.read_sql_query(f"SELECT * FROM cftc.order_of_things", engine1)
        fig, axs = plt.subplots(8, 3, figsize=(15, 20))
        fig.tight_layout()
        sns.set(font_scale=1.2)
        sns.set_style('white')
        sns.set_style('white', {'font.family': 'serif', 'font.serif': 'Times New Roman'})

        # fig.text(0.5, 0.00, 'Return Lag', ha='center', fontsize = 20)
        # fig.text(0.00, 0.5, 'Beta', va='center', rotation='vertical', fontsize = 20)

        plot_matrix = np.arange(24).reshape(8, -1)
        for col in range(len(plot_matrix[0])):
            # print(f"Row: {row}")print(f"Row: {row}")
            for row in range(len(plot_matrix)):
                try:
                    bb_tkr = bb_tkrs[plot_matrix[row][col]]
                    # print(bb_tkr)
                except:
                    break

                ax_curr = axs[row, col]
                plot_acf(residuals[bb_tkr], lags=np.arange(100)[1:], ax=ax_curr)

                ax_curr.set_xlabel('')
                ax_curr.set_ylabel('')
                ax_curr.set_title(f"Autocorr: {oot[oot.bb_tkr == bb_tkr].name.values[0]}")

        fig.delaxes(axs[7, 2])
        plt.savefig(f"Autocorr-{model_type_id}.png", dpi=100)  # './reports/figures/'+
        plt.show()


if __name__ == '__main__':
    cftcVariableName = 'cftc'  # * OR cftc_adj
    fcastVariableName = 'forecast'  # *OR 'forecast_adj'

    writer = pd.ExcelWriter(f'Autocorr.xlsx', engine='xlsxwriter')
    autoCorrelationPlots(model_type_ids=[153],
                         cftcVariableName='cftc',
                         fcastVariableName='forecast'
                         )
    cftcVar = 'cftc'  # * OR cftc_adj
    fcastVar = 'forecast'  # *OR 'forecast_adj'

    exclWriter = pd.ExcelWriter(f'Autocorr_.xlsx', engine='xlsxwriter')
    stats = autoCorrelationStatistics(model_type_ids=[153], cftcVariableName=cftcVariableName,
                                      fcastVariableName=fcastVariableName)
    exclWriter.save()
