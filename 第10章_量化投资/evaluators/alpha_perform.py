import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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



def Rsquared(y):
    # return R^2 where x and y are array-like
    from scipy.stats import linregress
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2, x*slope + intercept

def scale_one(x):
    #归一化处理
    x[np.isinf(x)] = np.nan
    res = (x.T / (np.nansum(np.abs(x), axis=1) + 1e-20)).T
    return res


def equal_wgts(x):
    x = np.nan_to_num(x)
    x[x > 0] = 1
    x[x < 0] = -1
    return x


class AlphaPerform():
    def __init__(self, dealObj, cost, cycle, quintiles_num, figure=True, stat_info=True):
        self.scaled_resample_wgts = dealObj.scaled_resample_wgts
        self.quintiles_num = quintiles_num
        self.cfg = {'Cycle': cycle, 'Quintiles': quintiles_num, 'Cost': cost, 'figure': figure, 'stat_info': stat_info}
        self.resample_return = dealObj.resample_return
        self.resample_dates = dealObj.resample_dates

    def build(self):
        self.stat_quintiles()
        self.stat_quintiles_pnl()
        self.stat_alpha_turnover()
        self.stat_alpha_pnl()
        self.stat_net_alpha_pnl()
        self.stat_alpha_sharpe()
        self.stat_alpha_drawdown()
        self.stat_alpha_drawdown_period()
        self.stat_net_alpha_sharpe()
        self.stat_net_alpha_drawdown()
        self.stat_net_alpha_drawdown_period()
        self.stat_alpha_Rsquared()
        self.stat_alpha_time_series_cpnl()
        self.stat_info()
        if self.cfg['figure']: self.plot()
        
    #----------------------------------------------------------------------
    def sort_quintiles(self, wgts, bottom, up):
        # 排序选择
        not_nan_num = - np.sum(np.isnan(wgts), axis=1) + wgts.shape[1]
        bottom_num = (np.round(bottom/100. * not_nan_num).astype(np.int) * np.ones_like(wgts).T).T   
        up_num = (np.round(up/100. * not_nan_num).astype(np.int) * np.ones_like(wgts).T).T               # 四舍五入, 然后进行类型转换 9.5 ---> 10. ---> 10
        rank_wgts = np.argsort(np.argsort(wgts, axis=1), axis=1).astype(np.float) + 1                #这里加1
        rank_wgts[np.isnan(wgts)] = np.nan
        res = np.ones_like(wgts)
        res[rank_wgts <= bottom_num] = np.nan
        res[rank_wgts > up_num] = np.nan
        return res



    #----------------------------------------------------------------------
    def stat_quintiles(self):
        quintiles_num=self.cfg['Quintiles']
        # 多分位测试
        if  quintiles_num == 10:
            self.quintiles_1 = self.sort_quintiles(self.scaled_resample_wgts, 0, 10)
            self.quintiles_2 = self.sort_quintiles(self.scaled_resample_wgts, 10, 20)
            self.quintiles_3 = self.sort_quintiles(self.scaled_resample_wgts, 20, 30)
            self.quintiles_4 = self.sort_quintiles(self.scaled_resample_wgts, 30, 40)
            self.quintiles_5 = self.sort_quintiles(self.scaled_resample_wgts, 40, 50)
            self.quintiles_6 = self.sort_quintiles(self.scaled_resample_wgts, 50, 60)
            self.quintiles_7 = self.sort_quintiles(self.scaled_resample_wgts, 60, 70)
            self.quintiles_8 = self.sort_quintiles(self.scaled_resample_wgts, 70, 80)
            self.quintiles_9 = self.sort_quintiles(self.scaled_resample_wgts, 80, 90)
            self.quintiles_10 = self.sort_quintiles(self.scaled_resample_wgts, 90, 100)
        elif quintiles_num == 5:
            """
            五分位测试
            多头值和空头值最大的20%， 20%-40%， 40%-60%，60-80%，80%-100%
            """
            self.quintiles_1 = self.sort_quintiles(self.scaled_resample_wgts, 0, 20)
            self.quintiles_2 = self.sort_quintiles(self.scaled_resample_wgts, 20, 40)
            self.quintiles_3 = self.sort_quintiles(self.scaled_resample_wgts, 40, 60)
            self.quintiles_4 = self.sort_quintiles(self.scaled_resample_wgts, 60, 80)
            self.quintiles_5 = self.sort_quintiles(self.scaled_resample_wgts, 80, 100)

        elif quintiles_num == 4:
            """
            4分位测试
            多头值和空头值最大的25%， 25%-50%， 50%-75%, 75%-100%
            """
            self.quintiles_1 = self.sort_quintiles(self.scaled_resample_wgts, 0, 2)
            self.quintiles_2 = self.sort_quintiles(self.scaled_resample_wgts, 25, 50)
            self.quintiles_3 = self.sort_quintiles(self.scaled_resample_wgts, 50, 75)
            self.quintiles_4 = self.sort_quintiles(self.scaled_resample_wgts, 75, 100)

        elif quintiles_num == 3:
            """
            3分位测试
            多头值和空头值最大的33%， 33%-66%, 66%-100%
            """
            self.quintiles_1 = self.sort_quintiles(self.scaled_resample_wgts, 0, 33)
            self.quintiles_2 = self.sort_quintiles(self.scaled_resample_wgts, 33, 67)
            self.quintiles_3 = self.sort_quintiles(self.scaled_resample_wgts, 67, 100)
        
        elif quintiles_num == 2:
            """
            2分位测试
            多头值和空头值最大的33%， 33%-66%, 66%-100%
            """
            self.quintiles_1 = self.sort_quintiles(self.scaled_resample_wgts, 0, 50)
            self.quintiles_2 = self.sort_quintiles(self.scaled_resample_wgts, 50, 100)

        else:
            raise Exception('')



    #----------------------------------------------------------------------
    def stat_quintiles_pnl(self):
        quintiles_num=self.cfg['Quintiles']
        #setattr(self, 'quintiles_%s_return' %i, scale_one(getattr(self, 'quintiles_%s' %i)) * self.OOS_resample_return)
        #setattr(self, 'quintiles_%s_pnl' %i, np.nan_to_num(np.nansum(getattr(self, 'quintiles_%s_return' %i), axis=1)) )
        if quintiles_num == 10:
            self.quintiles_1_return = scale_one(self.quintiles_1) * self.resample_return
            self.quintiles_2_return = scale_one(self.quintiles_2) * self.resample_return
            self.quintiles_3_return = scale_one(self.quintiles_3) * self.resample_return
            self.quintiles_4_return = scale_one(self.quintiles_4) * self.resample_return
            self.quintiles_5_return = scale_one(self.quintiles_5) * self.resample_return
            self.quintiles_6_return = scale_one(self.quintiles_6) * self.resample_return
            self.quintiles_7_return = scale_one(self.quintiles_7) * self.resample_return
            self.quintiles_8_return = scale_one(self.quintiles_8) * self.resample_return
            self.quintiles_9_return = scale_one(self.quintiles_9) * self.resample_return
            self.quintiles_10_return = scale_one(self.quintiles_10) * self.resample_return

            self.quintiles_1_pnl = np.nan_to_num(np.nansum(self.quintiles_1_return, axis=1)) 
            self.quintiles_2_pnl = np.nan_to_num(np.nansum(self.quintiles_2_return, axis=1)) 
            self.quintiles_3_pnl = np.nan_to_num(np.nansum(self.quintiles_3_return, axis=1)) 
            self.quintiles_4_pnl = np.nan_to_num(np.nansum(self.quintiles_4_return, axis=1))
            self.quintiles_5_pnl = np.nan_to_num(np.nansum(self.quintiles_5_return, axis=1)) 
            self.quintiles_6_pnl = np.nan_to_num(np.nansum(self.quintiles_6_return, axis=1)) 
            self.quintiles_7_pnl = np.nan_to_num(np.nansum(self.quintiles_7_return, axis=1)) 
            self.quintiles_8_pnl = np.nan_to_num(np.nansum(self.quintiles_8_return, axis=1)) 
            self.quintiles_9_pnl = np.nan_to_num(np.nansum(self.quintiles_9_return, axis=1))
            self.quintiles_10_pnl = np.nan_to_num(np.nansum(self.quintiles_10_return, axis=1)) 

        # 多分位测试
        elif quintiles_num == 5:
            """
            五分位测试
            多头值和空头值最大的20%， 20%-40%， 40%-60%，60-80%，80%-100%
            """
            self.quintiles_1_return = scale_one(self.quintiles_1) * self.resample_return
            self.quintiles_2_return = scale_one(self.quintiles_2) * self.resample_return
            self.quintiles_3_return = scale_one(self.quintiles_3) * self.resample_return
            self.quintiles_4_return = scale_one(self.quintiles_4) * self.resample_return
            self.quintiles_5_return = scale_one(self.quintiles_5) * self.resample_return

            self.quintiles_1_pnl = np.nan_to_num(np.nansum(self.quintiles_1_return, axis=1)) 
            self.quintiles_2_pnl = np.nan_to_num(np.nansum(self.quintiles_2_return, axis=1)) 
            self.quintiles_3_pnl = np.nan_to_num(np.nansum(self.quintiles_3_return, axis=1)) 
            self.quintiles_4_pnl = np.nan_to_num(np.nansum(self.quintiles_4_return, axis=1))
            self.quintiles_5_pnl = np.nan_to_num(np.nansum(self.quintiles_5_return, axis=1)) 

        elif quintiles_num == 4:
            """
            4分位测试
            多头值和空头值最大的25%， 25%-50%， 50%-75%, 75%-100%
            """
            self.quintiles_1_return = scale_one(self.quintiles_1) * self.resample_return
            self.quintiles_2_return = scale_one(self.quintiles_2) * self.resample_return
            self.quintiles_3_return = scale_one(self.quintiles_3) * self.resample_return
            self.quintiles_4_return = scale_one(self.quintiles_4) * self.resample_return

            self.quintiles_1_pnl = np.nan_to_num(np.nansum(self.quintiles_1_return, axis=1)) 
            self.quintiles_2_pnl = np.nan_to_num(np.nansum(self.quintiles_2_return, axis=1)) 
            self.quintiles_3_pnl = np.nan_to_num(np.nansum(self.quintiles_3_return, axis=1)) 
            self.quintiles_4_pnl = np.nan_to_num(np.nansum(self.quintiles_4_return, axis=1))
        elif quintiles_num == 3:
            """
            3分位测试
            多头值和空头值最大的33%， 33%-66%, 66%-100%
            """
            self.quintiles_1_return = scale_one(self.quintiles_1) * self.resample_return
            self.quintiles_2_return = scale_one(self.quintiles_2) * self.resample_return
            self.quintiles_3_return = scale_one(self.quintiles_3) * self.resample_return

            self.quintiles_1_pnl = np.nan_to_num(np.nansum(self.quintiles_1_return, axis=1)) 
            self.quintiles_2_pnl = np.nan_to_num(np.nansum(self.quintiles_2_return, axis=1)) 
            self.quintiles_3_pnl = np.nan_to_num(np.nansum(self.quintiles_3_return, axis=1)) 

        elif quintiles_num == 2:
            """
            2分位测试
            多头值和空头值最大的33%， 33%-66%, 66%-100%
            """
            self.quintiles_1_return = scale_one(self.quintiles_1) * self.resample_return
            self.quintiles_2_return = scale_one(self.quintiles_2) * self.resample_return

            self.quintiles_1_pnl = np.nan_to_num(np.nansum(self.quintiles_1_return, axis=1)) 
            self.quintiles_2_pnl = np.nan_to_num(np.nansum(self.quintiles_2_return, axis=1)) 

        else:
            raise Exception('')




    #----------------------------------------------------------------------
    def stat_alpha_pnl(self):
        quintiles_num=self.cfg['Quintiles']
        top_pnl = getattr(self, 'quintiles_' + str(quintiles_num) + '_pnl')
        bottom_pnl = getattr(self, 'quintiles_' + str(1) + '_pnl')
        self.alpha_pnl = 0.5*top_pnl - 0.5*bottom_pnl
        self.alpha_cpnl = np.cumsum(self.alpha_pnl)

    #----------------------------------------------------------------------
    def stat_net_alpha_pnl(self):
        self.alpha_cost = self.cfg['Cost'] * self.alpha_turnover_arr
        self.net_alpha_pnl = self.alpha_pnl - self.alpha_cost
        self.net_alpha_cpnl = np.cumsum(self.net_alpha_pnl)


    #----------------------------------------------------------------------
    def stat_alpha_turnover(self):
        quintiles_num=self.cfg['Quintiles']
        top_quintile = getattr(self, 'quintiles_' + str(quintiles_num))
        bottom_quintile = getattr(self, 'quintiles_' + str(1))


        self.top_bottom_quintile = scale_one(np.nan_to_num(top_quintile) - np.nan_to_num(bottom_quintile))
        shift = np.zeros_like(self.top_bottom_quintile) * np.nan
        shift[1:] = self.top_bottom_quintile[:-1]
        self.alpha_turnover_arr = np.nansum(np.nan_to_num(np.abs(self.top_bottom_quintile - shift)), axis=1)
        self.alpha_turnover = np.nanmean(self.alpha_turnover_arr)


        self.quantile_turnover = {}
        top_quintile = scale_one(equal_wgts(np.abs(top_quintile)))
        shift = np.zeros_like(top_quintile) * np.nan
        shift[1:] = np.nan_to_num(top_quintile)[:-1]
        self.quantile_turnover[quintiles_num] = np.nansum(np.abs(np.nan_to_num(top_quintile) - shift), axis=1)

        bottom_quintile = scale_one(equal_wgts(np.abs(bottom_quintile)))
        shift = np.zeros_like(bottom_quintile) * np.nan
        shift[1:] = np.nan_to_num(bottom_quintile)[:-1]
        self.quantile_turnover[1] = np.nansum(np.abs(np.nan_to_num(bottom_quintile) - shift), axis=1)
        #raise Exception()


    #----------------------------------------------------------------------
    def stat_alpha_sharpe(self):
        if self.cfg['Cycle'] == '15MIN' :
            self.alpha_sharpe =  np.sqrt(252 * 16) * np.nanmean(self.alpha_pnl)/np.nanstd(self.alpha_pnl)  
        elif self.cfg['Cycle'] == '60MIN':
            self.alpha_sharpe =  np.sqrt(252 * 4) * np.nanmean(self.alpha_pnl)/np.nanstd(self.alpha_pnl) 
        elif self.cfg['Cycle'] == '2HOUR':
            self.alpha_sharpe =  np.sqrt(252 * 2) * np.nanmean(self.alpha_pnl)/np.nanstd(self.alpha_pnl) 
        elif self.cfg['Cycle'] == 'DAY' or self.cfg['Cycle'] == '1DAY':
            self.alpha_sharpe =  np.sqrt(252) * np.nanmean(self.alpha_pnl)/np.nanstd(self.alpha_pnl)  
        else:
            raise Exception('Cycle error')


    #----------------------------------------------------------------------
    def stat_net_alpha_sharpe(self):
        if self.cfg['Cycle'] == '15MIN' :
            self.net_alpha_sharpe =  np.sqrt(252 * 16) * np.nanmean(self.net_alpha_pnl)/np.nanstd(self.net_alpha_pnl)  
        elif self.cfg['Cycle'] == '60MIN':
            self.net_alpha_sharpe =  np.sqrt(252 * 4) * np.nanmean(self.net_alpha_pnl)/np.nanstd(self.net_alpha_pnl) 
        elif self.cfg['Cycle'] == '2HOUR':
            self.net_alpha_sharpe =  np.sqrt(252 * 2) * np.nanmean(self.net_alpha_pnl)/np.nanstd(self.net_alpha_pnl) 
        elif self.cfg['Cycle'] == 'DAY' or self.cfg['Cycle'] == '1DAY':
            self.net_alpha_sharpe =  np.sqrt(252) * np.nanmean(self.net_alpha_pnl)/np.nanstd(self.net_alpha_pnl)  
        else:
            raise Exception('Cycle error')

            

    #----------------------------------------------------------------------
    def stat_alpha_drawdown(self):
        self.alpha_drawdown = drawdown(self.alpha_cpnl)
        self.alpha_max_drawdown = round(np.abs(np.min(self.alpha_drawdown)), 3)

    
    #----------------------------------------------------------------------
    def stat_alpha_drawdown_period(self):
        self.alpha_drawdown_period = drawdown_period(self.alpha_cpnl)
        self.alpha_max_drawdown_period = int(np.max(self.alpha_drawdown_period))


    #----------------------------------------------------------------------
    def stat_net_alpha_drawdown(self):
        self.net_alpha_drawdown = drawdown(self.net_alpha_cpnl)
        self.net_alpha_max_drawdown = round(np.abs(np.min(self.net_alpha_drawdown)), 3)

    
    #----------------------------------------------------------------------
    def stat_net_alpha_drawdown_period(self):
        self.net_alpha_drawdown_period = drawdown_period(self.net_alpha_cpnl)
        self.net_alpha_max_drawdown_period = int(np.max(self.net_alpha_drawdown_period))


    #----------------------------------------------------------------------
    def stat_alpha_Rsquared(self):
        self.alpha_Rsquared, self.alpha_regress = Rsquared(self.alpha_cpnl)


    #----------------------------------------------------------------------
    def stat_alpha_time_series_cpnl(self):
        quintiles_num=self.cfg['Quintiles']
        top_return = getattr(self, 'quintiles_' + str(quintiles_num) + '_return')
        bottom_return = getattr(self, 'quintiles_' + str(1) + '_return')
        ts_return = 0.5 * np.nan_to_num(top_return) - 0.5 * np.nan_to_num(bottom_return)
        self.ts_cpnl = np.cumsum(np.nan_to_num(ts_return), axis=0)



    def stat_info(self):
        self.alpha_return = self.alpha_cpnl[-1]

        self.indicators = {}
        # ===== alpha =====
        self.indicators["Alpha Turnover"] = round(self.alpha_turnover, 3)
        self.indicators["Alpha PNL"] = round(self.alpha_cpnl[-1], 3)
        self.indicators["Net Alpha PNL"] = round(self.net_alpha_cpnl[-1], 3)
        self.indicators["Alpha Sharpe"] = round(self.alpha_sharpe, 3)
        self.indicators["Net Alpha Sharpe"] = round(self.net_alpha_sharpe, 3)
        self.indicators["Alpha Max Drawdown"] = round(self.alpha_max_drawdown, 3)
        self.indicators["Alpha Max DrawdownPeriod"] = self.alpha_max_drawdown_period
        self.indicators["Net Alpha Max Drawdown"] = round(self.net_alpha_max_drawdown, 3)
        self.indicators["Net Alpha Max DrawdownPeriod"] = self.net_alpha_max_drawdown_period
        self.indicators["Alpha Rsquared"] = round(self.alpha_Rsquared, 3)

        print('[AlphaPerform] start:%s end:%s' %(self.resample_dates[0], self.resample_dates[-1]))
        self.stat_df = pd.DataFrame(self.indicators, index=[" "])
        if self.cfg['stat_info']: print(self.stat_df)


    def plot(self):
        figure = plt.figure(figsize=(18,10))
        quintiles_num = self.cfg["Quintiles"]
        for i in range(1, quintiles_num+1):
            q_pnl = getattr(self, 'quintiles_' + str(i) + '_pnl')
            tmp = np.zeros(len(q_pnl)+1)
            tmp[1:] = np.cumsum(q_pnl)
            signal_line = plt.plot(tmp, '-', linewidth=1, label='quintiles_' + str(i))

        alpha = np.zeros(len(self.alpha_cpnl)+1)
        alpha[1:] = self.alpha_cpnl
        alpha_line = plt.plot(alpha, color='r', linewidth=2, label='alpha')

        dates = self.resample_dates
        step = len(dates)/8
        space = [i for i in np.arange(len(dates)) if i%step==0]
        dates_str = [i.split(' ')[0] for i in dates[space]]
        if len(np.unique(dates_str)) <= 3:
            step = int(len(dates)/5)
            space = [i for i in np.arange(len(dates)) if i%step==0]
            dates_str = [i for i in dates[space]]
        plt.xticks(space, dates_str)
        #plt.xlabel('Date')
        plt.ylabel('PNL')
        plt.title(str(quintiles_num) + ' quintiles profits & loss')
        plt.legend(loc=2)
        plt.grid()
        plt.show()


