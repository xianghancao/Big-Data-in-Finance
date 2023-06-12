# encoding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from scipy import stats
import seaborn as sns
import matplotlib.cm as cm
import statsmodels.api as sm

from functools import wraps


class AnalysisPlot(): 
    def __init__(self, data, process, original_perform, alpha_perform):
        self.resample_dates = process.resample_dates
        self.top_bottom_quintile = alpha_perform.top_bottom_quintile
        self.alpha_pnl = alpha_perform.alpha_pnl
        self.ticker_names = data.ticker_names
        self.ts_cpnl = alpha_perform.ts_cpnl
        self.quantile_turnover = alpha_perform.quantile_turnover
        self.IC_arr = original_perform.IC_arr


    def build(self):
        resample_dates = pd.to_datetime(self.resample_dates)

        gf = GridFigure(rows=11, cols=2)

        pnl_df = pd.DataFrame(self.alpha_pnl, index=resample_dates)
        monthly_cpnl_df = pnl_df.resample('1M').sum()
        plot_monthly_cpnl_heatmap(monthly_cpnl_df, ax=gf.next_row())


        alpha_wgts_df = pd.DataFrame(self.top_bottom_quintile, columns=self.ticker_names, index=resample_dates)    
        plot_abs_alpha_wgts_hist(alpha_wgts_df, ax=gf.next_row())
        plot_long_short_alpha_wgts_hist(alpha_wgts_df, ax=gf.next_row())


        ts_cpnl_df = pd.DataFrame(self.ts_cpnl, columns=self.ticker_names, index=resample_dates)
        plot_time_series_cpnl(ts_cpnl_df, ax=gf.next_row())
        plot_time_series_cpnl_hist(ts_cpnl_df, ax=gf.next_row())

        alpha_pnl = pd.DataFrame(self.alpha_pnl, index=resample_dates)
        plot_alpha_pnl(alpha_pnl, ax=gf.next_row())

        quantile_turnover = pd.DataFrame(self.quantile_turnover, index=resample_dates)

        plot_top_bottom_quantile_turnover(quantile_turnover, ax=gf.next_row())

        ic = pd.DataFrame(self.IC_arr, index=resample_dates)    
        fr_cols = ic.shape[1]
        ax_plot_ic_ts = [gf.next_row() for x in range(fr_cols)]
        plot_ic_ts(ic, ax_plot_ic_ts)

        ic = pd.DataFrame(self.IC_arr, index=resample_dates)
        monthly_ic = ic.resample('1M').mean()
        plot_monthly_ic_heatmap(monthly_ic, ax=gf.next_row())

        ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
        plot_ic_hist(ic, ax=ax_ic_hqq[::2])
        plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

        gf.show()
        gf.close()



class GridFigure(object):
    """
    It makes life easier with grid plots
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def show(self):
        plt.show(self.fig)
        
    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None



def plot_alpha_pnl(alpha_pnl, ax=None):
    title = ('Alpha PNL')
    alpha_pnl.plot(alpha=0.8, ax=ax, lw=1, color='forestgreen')
    alpha_pnl.rolling(window=22).mean().plot(
        color='orangered',
        alpha=0.8,
        ax=ax
    )
    ax.legend(['mean returns spread', '1 month moving avg'], loc='upper right')
    ax.set(ylabel='alpha pnl',
           xlabel='',
           title=title)
    ax.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
    return ax




def plot_top_bottom_quantile_turnover(quantile_turnover, ax=None):
    """
    Plots period wise top and bottom quantile factor turnover.

    Parameters
    ----------
    quantile_turnover: pd.Dataframe
        Quantile turnover (each DataFrame column a quantile).
    period: int, optional
        Period over which to calculate the turnover
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    max_quantile = quantile_turnover.columns.max()
    min_quantile = quantile_turnover.columns.min()
    turnover = pd.DataFrame()
    turnover['top quantile turnover'] = quantile_turnover[max_quantile]
    turnover['bottom quantile turnover'] = quantile_turnover[min_quantile]
    turnover.plot(title='Top and Bottom Quantile Turnover', ax=ax, alpha=0.8, lw=1)
    ax.set(ylabel='Proportion Of Names New To Quantile', xlabel="")

    return ax


def plot_ic_ts(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient and IC moving
    average for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    ic = ic.copy()

    num_plots = len(ic.columns)
    if ax is None:
        f, ax = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        ax = np.asarray([ax]).flatten()

    ymin, ymax = (None, None)
    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        ic.plot(alpha=0.8, ax=a, lw=0.7, color='steelblue')
        ic.rolling(window=22).mean().plot(
            ax=a,
            color='forestgreen',
            lw=2,
            alpha=0.8
        )

        a.set(ylabel='IC', xlabel="")
        a.set_title("Information Coefficient (IC)")
        a.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        a.legend(['IC', '1 month moving avg'], loc='upper right')
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')

        curr_ymin, curr_ymax = a.get_ylim()
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

    for a in ax:
        a.set_ylim([ymin, ymax])

    return ax


def plot_ic_hist(ic, ax=None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=a)
        a.set(title="IC" , xlabel='IC')
        a.set_xlim([-1, 1])
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')
        a.axvline(ic.mean(), color='w', linestyle='dashed', linewidth=2)

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax


def plot_ic_qq(ic, theoretical_dist=stats.norm, ax=None):
    """
    Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
    a theoretical distribution.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats._continuous_distns
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = 'Normal'
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = 'T'
    else:
        dist_name = 'Theoretical'

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sm.qqplot(ic.replace(np.nan, 0.).values, theoretical_dist, fit=True,
                  line='45', ax=a)
        a.set(title="IC {} Dist. Q-Q".format(dist_name),
              ylabel='Observed Quantile',
              xlabel='{} Distribution Quantile'.format(dist_name))

    return ax



def plot_monthly_ic_heatmap(mean_monthly_ic, ax=None):
    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month],
        names=["year", "month"])

    sns.heatmap(
        mean_monthly_ic.unstack(),
        annot=True,
        alpha=0.8,
        center=0.0,
        annot_kws={"size": 7},
        linewidths=0.01,
        linecolor='white',
        cmap=cm.RdYlGn,
        cbar=True,
        ax=ax)
    ax.set(ylabel='year', xlabel='month')

    ax.set_title("Monthly Mean IC")

    return ax




def plot_monthly_cpnl_heatmap(monthly_cpnl_df, ax=None): 
    new_index_year = []
    new_index_month = []
    for date in monthly_cpnl_df.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    monthly_cpnl_df.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month],
        names=["year", "month"])

    sns.heatmap(
        monthly_cpnl_df.unstack(),
        annot=True,
        alpha=0.8,
        center=0.0,
        annot_kws={"size": 7},
        linewidths=0.01,
        linecolor='white',
        cmap=cm.RdYlGn,
        cbar=True,
        ax=ax)
    ax.set(ylabel='year', xlabel='month')

    ax.set_title("Monthly Alpha PNL")

    return ax



def plot_time_series_cpnl(ts_cpnl_df, ax=None): 
    ts_cpnl_df.plot(title='Time Series cpnl', ax=ax, alpha=0.8, lw=1)
    ax.set(ylabel='30 instruments', xlabel="")
    ax.legend(loc='upper left')
    return ax


def plot_time_series_cpnl_hist(ts_cpnl_df, ax=None):
    df = ts_cpnl_df.iloc[-1,:].sort_values(ascending=False)
    df.plot.bar(width=0.7, ax=ax, alpha=0.8, color='b')
    ax.set(ylabel='30 instruments', xlabel="")
    ax.set_title('Time Series cpnl histogram')
    ax.legend(loc='upper right')
    return ax


def plot_abs_alpha_wgts_hist(alpha_wgts_df, ax=None):
    df = alpha_wgts_df.abs().mean()    #.sort_values(ascending=False)
    df.plot.bar(width=0.7, ax=ax, alpha=0.8, color='b')
    ax.set(ylabel='%', xlabel="")
    ax.set_title('average abs(alpha wgts) histogram')
    #ax.legend(loc='upper right')
    return ax


def plot_long_short_alpha_wgts_hist(alpha_wgts_df, ax=None):
    df = alpha_wgts_df.iloc[-500:]
    long_df = (df.mean() + df.abs().mean())/2./df.abs().mean()
    #df = long_df.sort_values(ascending=False)
    long_df.plot.bar(width=0.7, ax=ax, alpha=0.35, color='red')
    df = alpha_wgts_df.iloc[-500:]
    short_df = (-df.mean() + df.abs().mean())/2./df.abs().mean() 
    #df = long_df.sort_values(ascending=False)
    short_df.plot.bar(width=0.7, ax=ax, alpha=0.35, color='forestgreen', bottom=long_df)
    ax.set(ylabel='long + short = 1', xlabel="")
    ax.set_title("long & short of alpha wgts histogram")
    plt.legend(['long', 'short'], loc='upper right')
    #ax.legend(loc='upper right')
    return ax







