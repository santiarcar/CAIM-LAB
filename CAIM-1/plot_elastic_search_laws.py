import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

import warnings

warnings.filterwarnings('error')

cwd = os.getcwd()

files_zipf = ['news', 'novels']

files_heap = ["novels" + str(n) for n in range(1, 11)]


def func_zipf(x, a, b, c):
    try:
        return c / pow(x + b, a)
    except:
        return 0


def func_a_over_x(x, a):
    return a / x


def func_ln(x, a, b):
    return -a * np.log(x * b)


def func_invexp(x, a, b):
    return a * np.exp(b / x)


def func_heap(x, k, b):
    return k * pow(x, b)


def zipf_checker(files: list):
    for f in files:
        df = pd.read_csv('{0}_clean.txt'.format(f), names=['freq', 'word'])
        df.sort_values(by=['freq'], ascending=[False], inplace=True)
        df['rank'] = df['freq'].rank(ascending=False, method='dense')

        df_fr = df[['freq', 'rank']].drop_duplicates().reset_index(drop=True)

        df_fr_main = df_fr[df_fr.shape[0] // 10:].reset_index(drop=True)

        # f = df_fr_main['freq'].sum()
        # df_fr_main['freq'] = df_fr_main['freq'] / f

        xdata = df_fr_main['rank'].array
        ydata = df_fr_main['freq'].array

        # DATA
        plt.plot(xdata, ydata, 'b-', label='data')

        # FIT ZIPFS
        popt, pcov = curve_fit(func_zipf, xdata, ydata, bounds=(0, [np.inf, np.inf, np.inf]))
        plt.plot(xdata, func_zipf(xdata, *popt), 'g--', label='fit-with-bounds\nfit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

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
        plt.title("ZIPF'S: {0}".format(f))
        plt.show()

        plt.clf()



def heaps_checker(files: list):

    data = {}

    for f in files:
        df = pd.read_csv('{0}_clean.txt'.format(f), names=['freq', 'word'])
        total_words = df['freq'].sum()
        distinct_words = df.shape[0]

        data[total_words] = distinct_words

    xdata = sorted(data)

    ydata = []

    for x in xdata:
        ydata.append(data[x])

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # DATA
    plt.plot(xdata, ydata, 'bo-', label='data')

    # FIT HEAPS
    popt, pcov = curve_fit(func_heap, xdata, ydata)
    plt.plot(xdata, func_heap(xdata, *popt), 'go--', label='fit-with-bounds\nfit: k=%5.3f, b=%5.3f' % tuple(popt))

    # # FIT A/X
    # popt, pcov = curve_fit(func_a_over_x, xdata, ydata, bounds=(0, [np.inf]))
    # plt.plot(xdata, func_a_over_x(xdata, *popt), 'r--', label='fit-with-bounds\nfit: a=%5.3f' % tuple(popt))
    #
    # # FIT A/X
    # popt, pcov = curve_fit(func_ln, xdata, ydata, bounds=(0, [np.inf, np.inf]))
    # plt.plot(xdata, func_ln(xdata, *popt), 'y--', label='fit-with-bounds\nfit: a=%5.3f, b=%5.3f' % tuple(popt))
    #
    # # FIT inverse exponential
    # popt, pcov = curve_fit(func_invexp, xdata, ydata, bounds=(0, [np.inf, np.inf]))
    # plt.plot(xdata, func_invexp(xdata, *popt), 'k--', label='fit-with-bounds\nfit: a=%5.3f, b=%5.3f' % tuple(popt))

    # PLOT

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.yscale('log')
    plt.legend()
    plt.title("HEAP'S: {0}".format(f))
    plt.show()


# a = zipf_checker(files=files_zipf)

b = heaps_checker(files=files_heap)



