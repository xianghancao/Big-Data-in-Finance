# encoding: utf-8
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

class RealPerform():
    def __init__(self, traders, capital, cost):
        self.cfg = {'Captical': capital, 'Cost':cost}
        self.scaled_resample_wgts = traders.scaled_resample_wgts
        self.resample_tradeprice = traders.resample_tradeprice
        self.resample_return = traders.resample_return
        self.resample_dates = traders.resample_dates
        
    def build(self):
        self.positionize()
        self.stat_pnl()
        self.stat_net_pnl()
        self.plot()
        
    def positionize(self):
        cap = self.cfg['Captical'] * np.ones((1,500)) * np.abs(self.scaled_resample_wgts) * np.sign(self.scaled_resample_wgts)
        stock = cap/self.resample_tradeprice/100
        self.position = stock.astype(np.int) * 100
        print()

    def stat_pnl(self):
        self.ret = self.resample_return * self.position
        self.pnl = np.nan_to_num(np.nansum(self.ret, axis=1))
        self.cpnl = np.cumsum(self.pnl)
        
    def stat_net_pnl(self):
        shift = np.zeros_like(self.position) * np.nan
        shift[1:] = self.position[:-1]
        turnover = np.abs(np.nan_to_num((self.position - shift) * self.resample_tradeprice))
        cost = self.cfg['Cost'] * turnover
        self.net_pnl = np.nan_to_num(np.nansum(self.ret - cost, axis=1))
        self.net_cpnl = np.cumsum(self.net_pnl)
        
    def plot(self):
        figure = plt.figure(figsize=(22,13))
        pnl = np.zeros(len(self.cpnl)+1)
        pnl[1:] = self.cpnl
        pnl_line = plt.plot(pnl, color='b', linewidth=2, label='pnl')
        
        net_pnl = np.zeros(len(self.net_cpnl)+1)
        net_pnl[1:] = self.net_cpnl
        net_pnl_line = plt.plot(net_pnl, color='r', linewidth=2, label='net_pnl')
        
        print(pnl.shape, net_pnl.shape)
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
        plt.title('real perfomance')
        plt.legend(loc=2)
        plt.grid()
        plt.show()
