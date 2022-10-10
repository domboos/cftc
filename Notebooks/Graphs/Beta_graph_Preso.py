import numpy as np
from matplotlib.ticker import FormatStrFormatter

from cfunctions import engine1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

query_flat = """
select md.model_id,md.bb_tkr , oot."name" from cftc.model_desc md
left join cftc.order_of_things oot on oot.bb_tkr = md.bb_tkr
where cot_norm = 'exposure'
and gamma_type = 'flat'
and cot_type = 'net_non_commercials'
and md.bb_tkr in ('S','KC','FC','SI')
and model_type_id =182
"""
query_sqrt = """
select md.model_id,md.bb_tkr , oot."name" from cftc.model_desc md
left join cftc.order_of_things oot on oot.bb_tkr = md.bb_tkr
where cot_norm = 'exposure'
and gamma_type = 'sqrt'
and cot_type = 'net_non_commercials'
and md.bb_tkr in ('S','KC','FC','SI')
and model_type_id =139;
"""


def plot_flat_betas():
    # Flat:
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))  # sharex=False, sharey= False ,
    fig.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98, hspace=0.45,
                        wspace=0.3)

    sns.set_style('white', {'font.family': 'serif', 'font.serif': 'Times New Roman'})

    flat_models = pd.read_sql_query(query_flat, engine1)

    plot_matrix = np.arange(4).reshape(2, -1)
    counter = 0
    for col in range(len(plot_matrix[0])):
        for row in range(len(plot_matrix)):
            print(f"{col};{row} ")

            ax_curr = axs[row, col]
            MODEL_ID_Flat = flat_models[flat_models.index == counter].model_id.values[0]
            q_betas_flat = f"select * from cftc.beta b \
                      where 1=1 \
                      and b.px_date = '2019-12-31' \
                      and model_id ={MODEL_ID_Flat}"
            beta = pd.read_sql_query(q_betas_flat, engine1)
            sns.lineplot(x=beta.return_lag, y=beta.qty, ax=ax_curr, linewidth=2, legend=False)

            ax_curr.set_xlabel('')
            ax_curr.set_ylabel('')

            ax_curr.set_title(flat_models[flat_models.index == counter].name.values[0], fontsize=12)
            ax_curr.tick_params(axis='both', labelsize=10)
            ax_curr.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

            counter = counter + 1
            plt.savefig(f"./flat_betas_preso.png", dpi=100)  # './reports/figures/'+
    plt.show()


def plot_sqrt_and_flat_betas():
    flat_models = pd.read_sql_query(query_flat, engine1)
    sqrt_models = pd.read_sql_query(query_sqrt, engine1)
    # Flat:
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))  # sharex=False, sharey= False ,
    fig.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98, hspace=0.45,
                        wspace=0.3)

    sns.set_style('white', {'font.family': 'serif', 'font.serif': 'Times New Roman'})

    plot_matrix = np.arange(4).reshape(2, -1)
    counter = 0
    for col in range(len(plot_matrix[0])):
        for row in range(len(plot_matrix)):
            print(f"{col};{row} ")

            ax_curr = axs[row, col]
            MODEL_ID_Flat = flat_models[flat_models.index == counter].model_id.values[0]
            MODEL_ID_Sqrt = sqrt_models[sqrt_models.index == counter].model_id.values[0]

            q_betas_flat = f"select * from cftc.beta b \
                      where 1=1 \
                      and b.px_date = '2019-12-31' \
                      and model_id ={MODEL_ID_Flat}"
            q_betas_sqrt = f"select * from cftc.beta b \
                                  where 1=1 \
                                  and b.px_date = '2019-12-31' \
                                  and model_id ={MODEL_ID_Sqrt}"

            beta_sqrt = pd.read_sql_query(q_betas_sqrt, engine1)
            beta_flat = pd.read_sql_query(q_betas_flat, engine1)
            sns.lineplot(x=beta_flat.return_lag, y=beta_flat.qty, ax=ax_curr, linewidth=2, legend=False)
            sns.lineplot(x=beta_sqrt.return_lag, y=beta_sqrt.qty, ax=ax_curr, linewidth=2, legend=False, color='red')

            ax_curr.set_xlabel('')
            ax_curr.set_ylabel('')

            ax_curr.set_title(flat_models[flat_models.index == counter].name.values[0], fontsize=12)
            ax_curr.tick_params(axis='both', labelsize=10)
            ax_curr.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

            counter = counter + 1
            plt.savefig(f"./sqrt_and_flat_betas_preso.png", dpi=100)  # './reports/figures/'+
    plt.show()


if __name__ == '__main__':
    plot_sqrt_and_flat_betas()