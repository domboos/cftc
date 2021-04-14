#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cfunctions import getGamma

gamma_sqrt = getGamma(260, regularization='d1', gammatype='sqrt', gammapara=1, naildownvalue0=0, naildownvalue1=0)
gamma_flat = getGamma(260, regularization='d1', gammatype='flat', gammapara=1, naildownvalue0=0, naildownvalue1=0)
#%%

fig, (s1) = plt.subplots(1, sharex=False, sharey= False ,figsize=(7,4))
color = ['crimson', 'cyan'] # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
fig.tight_layout()

sns.set(font_scale = 1.5)
sns.set_style('white')
sns.set_style('white', {'font.family':'serif', 'font.serif':'Times New Roman'})


sns.lineplot(data = gamma_flat.diagonal(),ax = s1, color = "#53777a")
sns.lineplot(data = gamma_sqrt.diagonal(),ax = s1, color = "#c02942")
s1.lines[0].set_linestyle("--")

plt.legend(frameon = False, labels=['y = const','y = sqrt(x)'])
sns.despine()


# # df_gamma_res = pd.read_excel('Choice_Gamma.xlsx',sheet_name  = 'nonc_expo')
# df_gamma_res = df_gamma_res.round(3)
# a = df_gamma_res.to_latex(index = False)

# %%
