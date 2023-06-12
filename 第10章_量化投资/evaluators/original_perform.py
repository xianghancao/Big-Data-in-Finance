# encoding: utf-8
import numpy as np
import pandas as pd 

def drawdown(cpnl):
    _cpnl = cpnl * np.ones((cpnl.shape[0], cpnl.shape[0]))
    _cpnl = np.tril(_cpnl)
    max_cpnl = np.nanmax(_cpnl, axis=1)
    return cpnl - max_cpnl



def drawdown_period(cpnl):
    _cpnl = cpnl * np.ones((cpnl.shape[0], cpnl.shape[0]))
    _cpnl = np.tril(_cpnl)
    _cpnl = np.hstack((np.zeros((cpnl.shape[0],1)), _cpnl))
    max_cpnl = np.argmax(_cpnl, axis=1)
    return np.arange(len(cpnl)) - max_cpnl + 1



class OriginalPerform():
    def __init__(self, process, cost, cycle='DAY'):
        self.scaled_resample_wgts = process.scaled_resample_wgts
        self.cfg = {'Cost':cost, "Cycle":cycle}
        self.resample_return = process.resample_return

    def build(self):
        self.stat_turnover()
        self.stat_pnl()
        self.stat_net_pnl()
        self.stat_sharpe()
        self.stat_drawdown()
        self.stat_drawdown_period()
        self.stat_anual()
        self.stat_IC()


    # ============================================ raw signal =================================================
    def stat_turnover(self):
        # raw signal turnover
        shift = np.zeros_like(self.scaled_resample_wgts) * np.nan
        shift[1:] = self.scaled_resample_wgts[:-1]
        self.turnover = np.nansum(np.nan_to_num(np.abs(self.scaled_resample_wgts - shift)), axis=1)
        self.avg_turnover = np.nanmean(self.turnover)



    #----------------------------------------------------------------------
    def stat_pnl(self):
        # raw signal pnl
        self.ret = self.resample_return * self.scaled_resample_wgts
        self.pnl = np.nan_to_num(np.nansum(self.ret, axis=1))
        self.cpnl = np.cumsum(self.pnl)


    #----------------------------------------------------------------------
    def stat_net_pnl(self):
        self.cost = self.cfg['Cost'] * self.turnover
        self.net_pnl = self.pnl - self.cost
        self.net_cpnl = np.cumsum(self.net_pnl)


        #----------------------------------------------------------------------
    def stat_anual(self):
        self.net_anual_pnl = np.nanmean(self.net_pnl) * 252


    #----------------------------------------------------------------------
    def stat_sharpe(self):
        if self.cfg['Cycle'] == '15MIN':
            self.sharpe =  np.sqrt(252 * 24) * np.nanmean(self.net_pnl) / np.nanstd(self.net_pnl)  
        elif self.cfg['Cycle'] == '60MIN':
            self.sharpe =  np.sqrt(252 * 6) * np.nanmean(self.net_pnl) / np.nanstd(self.net_pnl) 
        elif self.cfg['Cycle'] == '2HOUR':
            self.sharpe =  np.sqrt(252 * 3) * np.nanmean(self.net_pnl) / np.nanstd(self.net_pnl) 
        elif self.cfg['Cycle'] == 'DAY' or self.cfg['Cycle'] == '1DAY':
            self.sharpe =  np.sqrt(252) * np.nanmean(self.net_pnl) / np.nanstd(self.net_pnl)  
        else:
            raise Exception('stat_sharpe error! no match time frame')


    #----------------------------------------------------------------------
    def stat_drawdown(self):
        self.drawdown = drawdown(self.cpnl)
        self.max_drawdown = round(np.abs(np.min(self.drawdown)), 4)


    #----------------------------------------------------------------------
    def stat_drawdown_period(self):
        self.drawdown_period = drawdown_period(self.cpnl)
        self.max_drawdown_period = str(int(np.max(self.drawdown_period))) + self.cfg["Cycle"]


    #----------------------------------------------------------------------
    def stat_IC(self):
        # alpha 的 Infomation Coefficience(这里用RankIC，spearman rank) 即某时点某因子在全部股票因子暴露值排名与其下期回报排名的截面相关系数。
        
        def rankdata(x):
            import scipy.stats as st
            tmp_x = x.copy() 
            idx = np.sum(~np.isnan(tmp_x), axis=1) == 0  #某行全为nan值
            tmp_x[idx, :] = 0
            #sz = np.sum(~np.isnan(x), axis=1)
            res = st.mstats.rankdata(tmp_x, axis=1)   #.T/sz.astype(float)).T
            res[np.isnan(x)] = np.nan
            return res

        if np.isnan(self.sharpe) or self.sharpe == 0: 
            self.IC = np.nan
            self.IC_IR = np.nan
            self.IC_arr = np.nan * np.zeros_like(self.resample_dates)
            return 
        IC_arr = []
        rank_wgts = rankdata(self.scaled_resample_wgts)
        rank_return = rankdata(self.resample_return)
        #from scipy.stats import linregress
        for i in range(self.scaled_resample_wgts.shape[0]):
            #index_ = ~np.isnan(rank_wgts[i]) * ~np.isnan(rank_return[i])
            if np.isnan(rank_wgts[i]).all() or np.isnan(rank_return[i]).all():
                ic_val = np.nan
            else:
                ic_val = np.corrcoef(rank_wgts[i], rank_return[i])[0, 1]
            ##x = rank_wgts[i][index_] 
            ##y = rank_return[i][index_]
            ##slope, intercept, r_value, p_value, std_err = linregress(x, y) 
            IC_arr.append(ic_val)

        self.IC_arr = np.nan_to_num(IC_arr)
        self.IC = np.nanmean(IC_arr)
        self.IC_IR = np.nanmean(np.nan_to_num(IC_arr))/np.nanstd(np.nan_to_num(IC_arr))

