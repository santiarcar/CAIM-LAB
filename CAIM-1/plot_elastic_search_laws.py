import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

import warnings

warnings.filterwarnings('error')

cwd = os.getcwd()

archive = 'news'


def func(x, a, b, c):
    try:
        return c / pow(x+b, a)
    except:
        return 0

def func_a_over_x(x, a):
    return a/x

def func_ln(x, a, b):
    return -a * np.log(x*b)

def func_invexp(x, a, b):
    return a * np.exp(b/x)


df = pd.read_csv('{0}_clean.txt'.format(archive), names=['freq', 'word'])
df.sort_values(by=['freq'], ascending=[False], inplace=True)
df['rank'] = df['freq'].rank(ascending=False, method='dense')

df_fr = df[['freq', 'rank']].drop_duplicates().reset_index(drop=True)

df_fr_main = df_fr[df_fr.shape[0]//10:].reset_index(drop=True)

f = df_fr_main['freq'].sum()
df_fr_main['freq'] = df_fr_main['freq'] / f

xdata = df_fr_main['rank'].array
ydata = df_fr_main['freq'].array

# DATA
plt.plot(xdata, ydata, 'b-', label='data')

# FIT ZIPFS
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [np.inf, np.inf, np.inf]))
plt.plot(xdata, func(xdata, *popt), 'g--', label='fit-with-bounds\nfit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

# FIT A/X
popt, pcov = curve_fit(func_a_over_x, xdata, ydata, bounds=(0, [np.inf]))
plt.plot(xdata, func_a_over_x(xdata, *popt), 'r--', label='fit-with-bounds\nfit: a=%5.3f' % tuple(popt))

# FIT A/X
popt, pcov = curve_fit(func_ln, xdata, ydata, bounds=(0, [np.inf, np.inf]))
plt.plot(xdata, func_ln(xdata, *popt), 'y--', label='fit-with-bounds\nfit: a=%5.3f, b=%5.3f' % tuple(popt))

# FIT inverse exponential
popt, pcov = curve_fit(func_invexp, xdata, ydata, bounds=(0, [np.inf, np.inf]))
plt.plot(xdata, func_invexp(xdata, *popt), 'k--', label='fit-with-bounds\nfit: a=%5.3f, b=%5.3f' % tuple(popt))

# PLOT

plt.xlabel('x')
plt.ylabel('y')
# plt.yscale('log')
plt.legend()
plt.show()

pause = True
