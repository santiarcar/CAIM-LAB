import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from delete_noise import noise_cleaner

import warnings

warnings.filterwarnings('error')

cwd = os.getcwd()

files_zipf = ['news', 'novels']


def func_zipf(x, a, b, c):
    try:
        return c /((x + b)**a)
    except:
        return 0


def func_a_over_x(x, a, b):
    return a / (x+b)


def func_ln(x, a, b, c):
    return -a * np.log((x+b) * c)


def func_invexp(x, a, b, c):
    return a * np.exp(c / (x+b))


def func_heap(x, k, b):
    return k * pow(x, b)


def sq_root(x, a, b, c):
    return a*np.sqrt(c*(x+b))


def zipf_checker(files: list):
    for f in files:
        nc = noise_cleaner(f)
        nc.clean_noise()
        del nc

        df = pd.read_csv('{0}_clean.txt'.format(f), names=['freq', 'word'])
        df.sort_values(by=['freq'], ascending=[False], inplace=True)

        df = df[df.shape[0] // 20:].reset_index(drop=True)

        # df['rank'] = df['freq'].rank(ascending=False, method='dense')
        df.sort_values("freq", ascending=[False])
        df.reset_index(drop=True, inplace=True)
        df.reset_index(drop=False, inplace=True)
        df.rename(columns={"index": "rank"}, inplace=True)

        df_fr = df[['freq', 'rank']].drop_duplicates().reset_index(drop=True)
        df_fr.sort_values("rank", ascending=False, inplace=True)
        df_fr.drop_duplicates(subset="freq", inplace=True)

        df_fr_main = df_fr.copy()

        # f = df_fr_main['freq'].sum()
        # df_fr_main['freq'] = df_fr_main['freq'] / f

        xdata = df_fr_main['rank'].array
        ydata = df_fr_main['freq'].array

        # DATA

        plt.plot(xdata, ydata, 'bo-', label='data')

        # FIT ZIPFS
        popt, pcov = curve_fit(func_zipf, xdata, ydata, bounds=([0, 0, 100000], [3, 20000, np.inf]))
        plt.plot(xdata, func_zipf(xdata, *popt), 'gx--',
                 label='ZIPF\S FIT\nfit: a=%5.3f, b=%5.3f, c=%5.3f\nf = c / (x+b)^a' % tuple(popt))

        # FIT A/X
        popt, pcov = curve_fit(func_a_over_x, xdata, ydata, bounds=(0, [np.inf, np.inf]))
        plt.plot(xdata, func_a_over_x(xdata, *popt), 'r--',
                label='fit: a=%5.3f, b=%5.3f\nf = a/(x+b)' % tuple(popt))

        # FIT LN(X)
        # popt, pcov = curve_fit(func_ln, xdata, ydata, bounds=(0, [np.inf, np.inf, np.inf]))
        # plt.plot(xdata, func_ln(xdata, *popt), 'y--',
        #         label='fit: a=%5.3f, b=%5.3f, c=%5.3f\nf =-a * ln((x+b) * c)' % tuple(popt))

        # FIT inverse exponential
        popt, pcov = curve_fit(func_invexp, xdata, ydata, bounds=(0, [np.inf, np.inf, np.inf]))
        plt.plot(xdata, func_invexp(xdata, *popt), 'k--',
                label='fit: a=%5.3f, b=%5.3f, c=%5.3f\nf = a * e^(c/(x+b))' % tuple(popt))

        # PLOT

        plt.xlabel('x')
        plt.ylabel('y')
        # plt.yscale('log')
        plt.legend()
        plt.title("ZIPF'S: {0}".format(f))

        plt.savefig("zipfs_{0}.png".format(f))

        plt.clf()


def heaps_checker(file_basic: str, num_of_files: int, index_and_count: bool = False, zoomed: bool = False):
    if index_and_count:
        for i in range(1, num_of_files + 1):
            os.system(
                "python IndexFiles.py --index {1}{0} --path /home/santi_arcar/PycharmProjects/CAIM-LAB/CAIM-1/{1}{0}".format(
                    str(i), file_basic)
            )

        for i in range(1, num_of_files + 1):
            os.system("python CountWords.py --index {1}{0} > {1}{0}_output.txt".format(str(i),
                                                                                       file_basic)
                      )

    for i in range(1, num_of_files + 1):
        nc = noise_cleaner("{0}{1}".format(file_basic, str(i)))
        nc.clean_noise()
        del nc

    files = ["{0}".format(file_basic) + str(n) for n in range(1, num_of_files + 1)]

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

    xmedian = xdata[len(xdata) // 2]
    ymedian = sorted(ydata)[len(ydata) // 2]

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    # DATA
    plt.plot(xdata, ydata, 'bo-', label='data')

    # FIT HEAPS
    popt, pcov = curve_fit(func_heap, xdata, ydata)
    plt.plot(xdata, func_heap(xdata, *popt), 'gx--', label='HEAP\'S FIT\nfit: k=%5.3f, b=%5.3f\nf = k * x^b' % tuple(popt))

    # FIT SQRT
    popt, pcov = curve_fit(sq_root, xdata, ydata, bounds=(0, [np.inf, np.inf, np.inf]))
    plt.plot(xdata, sq_root(xdata, *popt), 'r--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f\nf = a + sqrt(c * (x+b))' % tuple(popt))


    # PLOT

    plt.xlabel('x')
    plt.ylabel('y')
    # plt.yscale('log')
    if zoomed:
        plt.xlim(right=xmedian*2)
        plt.ylim(top=ymedian*2)
    plt.legend()
    plt.title("HEAP'S: {0}".format(file_basic))
    plt.savefig("heaps_{0}{1}.png".format(file_basic, "_zoomed"*zoomed))


zipf_checker(files=files_zipf)
heaps_checker(file_basic='novels', num_of_files=32, zoomed=False)
heaps_checker(file_basic='novels', num_of_files=32, zoomed=True)



