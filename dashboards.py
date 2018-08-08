import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import scipy.stats as st
import datetime


def plot_error(me, aemo):
    fig, ax = plt.subplots()
    service = 'ENERGY'
    me_service = me[(me['Service'] == service)]
    aemo_service = aemo.loc[:, ('REGIONID', 'SETTLEMENTDATE', 'RRP')]
    comp = pd.merge(me_service, aemo_service, 'inner', left_on=['DateTime', 'State'],
                    right_on=['SETTLEMENTDATE', 'REGIONID'])
    comp['ERROR'] = comp['Price'] - comp['RRP']
    error = np.asarray(comp['ERROR'])
    #sns.set(color_codes=True)
    plt.hist(error, label='Error Frequency', bins=10000)
    ci = st.t.interval(0.95, len(error) - 1, loc=np.mean(error), scale=st.sem(error))
    mean_error = np.mean(abs(error))
    print(max(error))
    print(min(error))
    plt.axvline(np.percentile(error, 5), color='r', linestyle= '--', linewidth=0.5,
                label='$5^{th}$ / $95^{th}$ Percentile' + '\n ({} / {}) {}'.format(round(np.percentile(error, 5),1),
                                                                                   round(np.percentile(error, 95),1),
                                                                                   round(mean_error,1)))
    plt.axvline(np.percentile(error, 95), color='r', linestyle= '--', linewidth=0.5)
    plt.legend()
    #plt.xlim(-60, 100)
    plt.xlabel('Error \$ ($P_{estimate} - P_{actual}$)')
    plt.ylabel('Normalised Frequency')
    plt.title('Nemlite Backcast \n Energy Price Error Distribution \n {} to {}'.
              format(me_service['DateTime'].min()[:10], me_service['DateTime'].max()[:10]),
              fontdict = {'fontsize': 10})
    plt.xlim([-100, 100])

    #error_and_flags = pd.merge(comp, flags, 'inner', 'DateTime')
    #error = np.asarray(error_and_flags['ERROR'])
    #sns.distplot(error, rug=False, kde=False, hist=True, ax=ax, rug_kws={"color": 'r'})

    return fig


def plot_comp(me, aemo, region):
    fig, ax = plt.subplots(1, 1)
    me_service = me[(me['State'] == region) & (me['Service'] == 'ENERGY')]
    aemo_service = aemo[(aemo['REGIONID'] == region)]
    dates = [pd.to_datetime(elem, format='%Y/%m/%d %H:%M:%S') for elem in list(me_service['DateTime'])]
    plt.plot(dates, list(aemo_service['RRP']), 'ro', markersize=1, label='AEMO')
    plt.plot(dates, list(me_service['Price']), 'b.', label='Nemlite', markersize=1)
    plt.title('Nemlite Backcast \n  {} {}'.format(region, 'Energy'))
    plt.ylabel('Price ($ {\$}/{MWh} $)')
    plt.yscale('log')
    plt.legend()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    ax.tick_params(labelsize=7)
    return fig


def plot_objective_error(me, aemo):
    fig, ax = plt.subplots()
    me = me.loc[:, ('DateTime', 'Objective')]
    me.columns = ['SETTLEMENTDATE', 'Objective']
    me['SETTLEMENTDATE'] = pd.to_datetime(me['SETTLEMENTDATE'])
    comp = pd.merge(me, aemo, 'inner', ['SETTLEMENTDATE'])
    comp['ERROR'] = comp['Objective'] - comp['TOTALOBJECTIVE']
    error = np.asarray(comp['ERROR'])
    sns.set(color_codes=True)
    sns.distplot(error, bins=100, kde=False, norm_hist=False, label='Error Frequency', ax=ax)
    ci = st.t.interval(0.95, len(error) - 1, loc=np.mean(error), scale=st.sem(error))
    plt.axvline(np.percentile(error, 5), color='r', linestyle= '--', linewidth=0.5,
                label='$5^{th}$ / $95^{th}$ Percentile' + '\n ({} / {})'.format(round(np.percentile(error, 5),1),
                                                                            round(np.percentile(error, 95),1)))
    plt.axvline(np.percentile(error, 95), color='r', linestyle= '--', linewidth=0.5)
    plt.legend()
    plt.xlabel('Error \$ ($P_{estimate} - P_{actual}$)')
    plt.ylabel('Normalised Frequency')
    plt.title('Nemlite Backcast \n Objective value Error Distribution \n {} to {}'.
              format(me['SETTLEMENTDATE'].min(), me['SETTLEMENTDATE'].max()),
              fontdict = {'fontsize': 10})

    return fig


def construct_pdf(me, aemo, save_as):
    pp = PdfPages(save_as)
    #pp.savefig(plot_objective_error(objective_me, objective_aem))
    if not me.empty:
        pp.savefig(plot_error(me, aemo))
        pp.savefig(plot_comp(me, aemo, 'NSW1'))
        pp.savefig(plot_comp(me, aemo, 'VIC1'))
        pp.savefig(plot_comp(me, aemo, 'QLD1'))
        pp.savefig(plot_comp(me, aemo, 'SA1'))
        pp.savefig(plot_comp(me, aemo, 'TAS1'))
    pp.close()
    return

