
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from cfunctions import engine1, gets, getexposure


# for xlim
def getDates(model_type_id):
    query = ("select b.model_id, min(b.px_date) minDate, max(b.px_date) maxDate \n"
             "from cftc.beta b where b.model_id in \n"
             f"(SELECT model_id from cftc.model_desc where model_type_id = {model_type_id}) \n"
             "group by b.model_id  order by min(b.px_date) asc")


    aggBetaStats = pd.read_sql_query(query,engine1)

    model_id = aggBetaStats.model_id[0]
    dates_all = pd.read_sql_query(f"SELECT px_date from cftc.beta where model_id = {model_id}", engine1)

    return dates_all


def getRatioOIvsPosExposure(model_id, bb_tkr):
    query = (" SELECT b.px_date,\n"
             "    b.model_id,\n"
             "    avg(b.qty) AS average\n"
             "   FROM cftc.vw_beta b \n"
             f"where model_id = {model_id} \n"
             "  GROUP BY b.px_date, b.model_id")
    exposure_avg = pd.read_sql_query(query, engine1)
    oi = getexposure(type_of_trader='agg_open_interest', norm='exposure', bb_tkr=bb_tkr, start_dt='1900-01-01',
                     end_dt='2100-01-01', bb_ykey='COMDTY')
    oi.columns = oi.columns.droplevel(0)
    oi.columns = ['oi']

    df = pd.merge(exposure_avg, oi, how='inner', on='px_date')
    return df


# * get Evolution of Betas:
def getEvolutionofBetaPeak(betas_per_model):
    temp = betas_per_model.set_index('return_lag')
    n10 = temp.groupby('px_date').qty.nlargest(10)
    n10 = n10.reset_index()

    n10_lowerb = n10.set_index('qty').groupby('px_date').return_lag.nsmallest(1).reset_index()
    n10_upperb = n10.set_index('qty').groupby('px_date').return_lag.nlargest(1).reset_index()
    n10_max = n10.set_index('return_lag').groupby('px_date').qty.nlargest(1).reset_index()
    return n10_lowerb, n10_max, n10_upperb


# *do plots:
def getPlots(model_type_id: int, showme: str, saveFig: bool):
    """[Plots for one model Betas, and alpha]

    Args:
        model_type_id ([int]): defines for which model_type the plots will be generated
        showme ([str()]): defines what will be plottet: args can be one of: ['EvolutionOfBetaPeak','ExposureAndOI','RatioExposureOI','weighted_beta','alpha']
    """

    models = pd.read_sql_query(f"SELECT * from cftc.model_desc where model_type_id = {model_type_id}", engine1)
    bb_tkrs = list(pd.read_sql_query(f"SELECT  bb_tkr from cftc.order_of_things order by ranking asc", engine1).bb_tkr)
    # * general Layout
    fig, axs = plt.subplots(8, 3, figsize=(15, 20))
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    sns.set(font_scale=0.9)
    sns.set_style('white')
    sns.set_style('white', {
        'font.family': 'serif',
        'font.serif': 'Times New Roman'
    }
                  )

    plot_matrix = np.arange(24).reshape(8, -1)
    dates_all = getDates(model_type_id=model_type_id)
    for col in range(len(plot_matrix[0])):
        for row in range(len(plot_matrix)):
            try:
                bb_tkr = bb_tkrs[plot_matrix[row][col]]
                print(bb_tkr)
            except:
                break

            ax_curr = axs[row, col]
            model_id = models[models.bb_tkr == bb_tkr].model_id.values[0]

            print(model_id)

            if showme == 'EvolutionOfBetaPeak':
                betas_per_model = pd.read_sql_query(f"SELECT * from cftc.beta where model_id = {model_id}", engine1)
                betas_per_model = betas_per_model.sort_values(['px_date', 'return_lag'], ascending=True)

                n10_lowerb, n10_max, n10_upperb = getEvolutionofBetaPeak(betas_per_model)
                if n10_lowerb.shape[0] != dates_all.shape[0]:
                    n10_lowerb = pd.merge(n10_lowerb, dates_all, on='px_date', how='outer').sort_values('px_date')
                    n10_max = pd.merge(n10_max, dates_all, on='px_date', how='outer').sort_values('px_date')
                    n10_upperb = pd.merge(n10_upperb, dates_all, on='px_date', how='outer').sort_values('px_date')

                sns.lineplot(x=n10_lowerb.px_date, y=n10_lowerb.return_lag, ax=ax_curr, linewidth=2, legend=False,
                             color='crimson')
                sns.lineplot(x=n10_max.px_date, y=n10_max.return_lag, ax=ax_curr, linewidth=2, legend=False,
                             color='cyan')
                sns.lineplot(x=n10_upperb.px_date, y=n10_upperb.return_lag, ax=ax_curr, linewidth=2, legend=False,
                             color='crimson')
                # ax_curr.set_xlim((dates_all[0],dates_all.iloc[-1]))

            elif showme == 'ExposureAndOI':
                ax2 = ax_curr.twinx()
                df = getRatioOIvsPosExposure(model_id=model_id, bb_tkr=bb_tkr)
                # plot some data on each axis.
                lns1 = ax_curr.plot(df.px_date, df.average, 'r')
                lns2 = ax2.plot(df.px_date, df.oi, 'b')
                # ax_curr.set_xlim((dates_all[0],dates_all.iloc[-1]))
                # sns.lineplot(x = df.px_date,y = df.qty,ax=ax_curr, linewidth = 2, legend = False,color ='crimson')
                # sns.lineplot(x = df.px_date,y = df.oi,ax=ax_curr, linewidth = 2, legend = False,color ='cyan')
            elif showme == 'RatioExposureOI':
                print('in')
                df = getRatioOIvsPosExposure(model_id=model_id, bb_tkr=bb_tkr)
                df['ratio'] = df.average / df.oi

                sns.lineplot(x=df.px_date, y=df.ratio, ax=ax_curr, linewidth=2, legend=False, color='blue')
                ax_curr.axhline(0, ls='--', color='black')

            elif showme == 'weighted_beta':
                query = ("SELECT b.px_date, b.model_id,sum(b.qty * b.return_lag) / sum(abs(b.qty)) average \n"
                         "FROM cftc.beta b \n"
                         f"where model_id = {model_id} \n"
                         "group by b.px_date, b.model_id")
                df = pd.read_sql_query(query, engine1)
                sns.lineplot(x=df.px_date, y=df.average, ax=ax_curr, linewidth=2, legend=False, color='darkblue')
                ax_curr.axhline(0, ls='--', color='black')
            elif showme == 'alpha':
                df = pd.read_sql_query(f"Select * from cftc.alpha where model_id = {model_id}", engine1)
                sns.lineplot(x=df.px_date, y=df.qty, ax=ax_curr, linewidth=2, legend=False, color='darkblue')
                # ax_curr.set_xlim((dates_all[0],dates_all.iloc[-1]))
                ax_curr.axhline(0, ls='--', color='black')
            else:
                print(f"wrong definition of showme: {showme}")

            ax_curr.set_xlabel('')
            ax_curr.set_ylabel('')
            title = pd.read_sql_query(f"SELECT name from cftc.order_of_things where bb_tkr = '{bb_tkr}'", engine1).name[0]
            ax_curr.set_title(title)

        # plt.xlim(dates_all[0],dates_all.iloc[-1])

    fig.delaxes(axs[7, 2])
    if saveFig == True:
        plt.savefig(f"./temp/{showme}_model_type_id_{model_type_id}.png", dpi=100)  # './reports/figures/'+

    plt.show()
    # os.chdir('/home/jovyan/work/')


def main():
    #todo: fix showme= ExposureAndOI
    showmelist = ['alpha','weighted_beta','EvolutionOfBetaPeak']
    modelTypeIds = [153, 152, 151, 150, 149, 148, 147, 146,145]

    for model_type_id in modelTypeIds:
        for plottype in showmelist:
            getPlots(model_type_id=model_type_id, showme=plottype, saveFig=True)



if __name__ == '__main__':
    main()
