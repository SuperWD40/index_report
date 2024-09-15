import pandas as pd
import numpy as np
from mytoolkit.finance import finance
import scipy.stats as stats
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.ticker

import warnings
warnings.filterwarnings("ignore")

class BaseClass:
    _date = None
    _range_dict = None
    _freq_dict = None
    _range_keys = None
    _freq_keys = None
    
    def __init__(self, date):
        self.date = pd.to_datetime(date, dayfirst=True).tz_localize(None)
        if BaseClass._date is None:
            BaseClass._initialize_shared_data(self.date)

    @classmethod
    def _initialize_shared_data(cls, date):
        cls._range_dict = {
            '1W'    : date - pd.Timedelta(days=7),
            '1M'    : date - pd.Timedelta(days=30),
            'YTD'   : pd.Timestamp(year=date.year, month=1, day=1),
            '1Y'    : date - pd.Timedelta(days=365),
            '5Y'    : date - pd.Timedelta(days=365*5),
            '10Y'   : date - pd.Timedelta(days=365*10),
        }
        cls._freq_dict = {
            'D' : 365,
            'B' : 255,
            'W' : 52,
            'M' : 12
        }
        cls._range_keys = list(cls._range_dict.keys())
        cls._freq_keys = list(cls._freq_dict.keys())
        cls._date = date
    
    @classmethod
    def get_range_dict(cls):
        return cls._range_dict
    
    @classmethod
    def get_freq_dict(cls):
        return cls._freq_dict
    
    @classmethod
    def get_range_keys(cls):
        return cls._range_keys
    
    @classmethod
    def get_freq_keys(cls):
        return cls._freq_keys
    
    @classmethod
    def get_date(cls):
        return cls._date

# Calculating asset valuations and allocation
class index_table(BaseClass):
    def __init__(self, history, allocation, date):
        super().__init__(date)
        # Assigning input parameters to instance variables
        self.history = history
        self.date = self.get_date()
        self.allocation = allocation

    def compute(self):
        index = pd.DataFrame()
        index.index = self.allocation.index
        index['Price'] = self.history.loc[self.date]
        index['Share'] = self.allocation['Share']
        index['Valorisation'] = self.allocation['Share'] * index['Price']
        index['Allocation'] = index['Valorisation'] / index['Valorisation'].sum()
        index['Date'] = self.date
        index['Type'] = self.allocation['Type']
        index['Account'] = self.allocation['Account']
        return index
    
    def historization(self):
        
    
    def show(self):
        display = self.compute()
        display = display.drop(columns=['Date'])
        display['Allocation'] = display['Allocation'].apply(lambda x: f"{x * 100:.2f}%")

        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.float_format', '{:,.1f}'.format,
            'display.max_colwidth', None,
            'display.colheader_justify', 'center'
        ):
            print(display)

# Class Composition
class stats_table(BaseClass):
    def __init__(self, history, market, riskfree, allocation, date):
        super().__init__(date)
        # Assigning input parameters to instance variables
        self.history = history
        self.date = self.get_date()
        self.range_dict = self.get_range_dict()
        self.freq_dict = self.get_freq_dict()
        self.range_keys = self.get_range_keys()
        self.freq_keys = self.get_freq_keys()
        self.market = market
        self.riskfree = riskfree
        self.allocation = allocation
    
    def stats(self, freq, range): 
        """Calculating statistics based on frequency and time range"""
        # Fetching historical data and resampling based on frequency and time range
        df_data = self.history
        df_data = df_data.loc[self.range_dict[range]:]
        df_data = df_data.resample(freq).ffill()

        # Fetching market data and risk-free rate for the specified time range and frequency
        market = self.market
        market = market.loc[self.range_dict[range]:]
        market = market.resample(freq).ffill()
        riskfree = self.riskfree

        # Get the correct freq for a given freq
        freq = self.freq_dict[freq]

        # Calculating various statistics for each asset in the data
        df_stats = pd.DataFrame() 
        for ticker in df_data.columns:
            history = df_data[ticker]
            if history.isna().any():
                # Handling missing values
                df_stats[ticker] = np.nan  
            else:
                # Calculating and formatting various statistics
                df_stats.at['Last price', ticker]       = "{:.1f}".format(history.loc[self.date])
                df_stats.at['Highest price', ticker]    = "{:.1f}".format(history.max())
                df_stats.at['Lowest price', ticker]     = "{:.1f}".format(history.min())
                df_stats.at['Last return', ticker]      = "{:.1f}%".format(history.loc[:self.date].pct_change().iloc[-1] * 100)
                df_stats.at['Cumulative return', ticker]= "{:.1f}%".format(((history.iloc[-1] - history.iloc[0]) / history.iloc[0]) * 100)
                df_stats.at['Avg returns', ticker]      = "{:.1f}%".format(finance.avg_return(history, timeperiod=freq) * 100)
                df_stats.at['Avg volatility', ticker]   = "{:.1f}%".format(finance.avg_volatility(history, timeperiod=freq) * 100)
                df_stats.at['Alpha', ticker]            = "{:.1f}%".format(finance.alpha(history, market) * 100)
                df_stats.at['Beta', ticker]             = "{:.1f}".format(finance.beta(history, market))
                df_stats.at['Sharpe ratio', ticker]     = "{:.1f}".format(finance.sharpe_ratio(history, riskfree, timeperiod=freq))
                df_stats.at['Calmar ratio', ticker]     = "{:.1f}".format(finance.calmar_ratio(history, riskfree, timeperiod=freq))
                df_stats.at['Max drawdown', ticker]     = "{:.1f}%".format(finance.max_drawdown(history) * 100)
                df_stats.at['Skewness', ticker]         = "{:.1f}".format(stats.kurtosis(history))
                df_stats.at['Kurtosis', ticker]         = "{:.1f}".format(stats.skew(history))
                #df_stats.at['3% VAR', ticker]           = "{:.1f}%".format(finance.var(history, feq='B', conf_level=0.03) * 100)
                df_stats.at['Allocation', ticker]       = "{:.1f}%".format(self.allocation[ticker] * 100)

        # Displaying the statistics table
        with pd.option_context('expand_frame_repr', False):
            print(df_stats)
        
    def show(self):
        """Displaying interactive controls for selecting frequency and time range"""
        # Creating interactive controls for selecting frequency and time range
        controls = widgets.interactive(
            self.stats,
            freq=widgets.Select(options=self.freq_keys, value=self.freq_keys[0]),
            range=widgets.Select(options=self.range_keys, value=self.range_keys[0])
        )

        # Displaying the interactive controls
        display(controls)

# Class Comparaison
class chart_comparison(BaseClass):
    def __init__(self, history, date):
        super().__init__(date)
        # Assigning input parameters to instance variables
        self.history = history
        self.date = self.get_date()
        self.range_dict = self.get_range_dict()
        self.freq_dict = self.get_freq_dict()
        self.range_keys = self.get_range_keys()
        self.freq_keys = self.get_freq_keys()
    
    def plot(self, freq, range, indexing=True):
        """Plot data from the DataFrame"""
        history = self.history
        history = history.loc[self.range_dict[range]:]
        history = history.resample(freq).ffill()
        if indexing:
            history = finance.indexing(history)
        history.plot(figsize=(12,6), title='Components returns comparison')
    
    def show(self, title="", indexing=True):
        """Displaying interactive controls for plotting"""
        controls = widgets.interactive(
            self.plot,
            title=title,
            indexing=indexing,
            freq=widgets.Select(options=self.freq_keys, value=self.freq_keys[0]),
            range=widgets.Select(options=self.range_keys, value=self.range_keys[0])
        )
        display(controls)

class top_n_flop(BaseClass):
    def __init__(self, history, date):
        super().__init__(date)
        # Assigning input parameters to instance variables
        self.date = pd.to_datetime(date, dayfirst=True).tz_localize(None)
        self.history = history
        self.date = self.get_date()
        self.range_dict = self.get_range_dict()
        self.range_keys = self.get_range_keys()
    
    def plot(self, range, n_stock): 
        df = self.history.resample('D').ffill()
        df = (df.loc[self.date] - df.loc[self.range_dict[range]]) / df.loc[self.range_dict[range]]
        df = df.dropna()

        df_first = (df.sort_values().tail(n_stock) * 100).round(2)
        df_last = (df.sort_values().head(n_stock) * 100).round(2)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        df_first.sort_values(ascending=True).plot(kind='barh', color='g', ax=ax2)
        df_last.sort_values(ascending=False).plot(kind='barh', color='r', ax=ax1)

        # Représenter à droite l'axe y du graphique 2
        ax2.yaxis.set_ticks_position('right')

        ax1.set_xlim([df_last.min()*1.2, 0])
        ax2.set_xlim([0, df_first.max()*1.2])

        # Afficher les variations directement sur le graphique
        for i, v in enumerate(df_last.sort_values(ascending=False)):
            ax1.text(df_last.min()*0.2+v, i, f"{v:.2f}%", color='r')
        for i, v in enumerate(df_first):
            ax2.text(df_first.min()*0.05+v, i, f"{v:.2f}%", color='g')

        # Afficher l'échelle x uniquement en entier
        ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        # Insérer un titre
        fig.suptitle('Componentd top and flop', y=0.95)

        # Ajuster l'écart entre les deux graphiques
        fig.subplots_adjust(wspace=0.05)

        plt.show()
        
    def show(self):
        """Displaying interactive controls for selecting frequency and time range"""
        # Creating interactive controls for selecting frequency and time range
        controls = widgets.interactive(
            self.plot,
            range=widgets.Select(options=self.range_keys, value=self.range_keys[0]),
            n_stock=widgets.IntText(value=5, description='N stocks:', disabled=False)
        )

        # Displaying the interactive controls
        display(controls)

