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
    def __init__(self, input_date): 
        self.date = pd.to_datetime(input_date, dayfirst=True).tz_localize(None)

    def get_range_dict(self):  
        date = self.date 
        return {
            '1W': date - pd.Timedelta(days=7),
            '1M': date - pd.Timedelta(days=30),
            'YTD': pd.Timestamp(year=date.year, month=1, day=1),
            '1Y': date - pd.Timedelta(days=365),
            '5Y': date - pd.Timedelta(days=365 * 5),
            '10Y': date - pd.Timedelta(days=365 * 10),
        }


    def get_freq_dict(self):
        return {
            'D': 365,
            'B': 255,
            'W': 52,
            'M': 12
        }

    def get_date(self): 
        return self.date

    def get_history(self, input_index, input_history, freq='B', range='1Y', by='All', indexing=False):
        #Index
        index = pd.DataFrame()
        index.index = input_index.index
        index['Price'] = input_history.iloc[-1]
        index['Share'] = input_index['Share']
        index['Valorisation'] = input_index['Share'] * index['Price']
        index['Allocation'] = index['Valorisation'] / index['Valorisation'].sum()
        if by not in ['All', 'Overall']:
            index[by] = input_index[by]
            by_index = index.groupby(by)['Allocation'].transform('sum')
            index['Allocation'] = index['Allocation'] / by_index

        #History
        input_history = input_history.loc[range:]
        input_history = input_history.resample(freq).ffill()
        if by == 'All' and indexing == False:
            history = input_history
        elif by == 'All' and indexing == True:
            history = 100 * (input_history.pct_change() + 1).cumprod()
            history.loc[history.index[0]] = 100
        elif by == 'Overall':
            history = (index['Allocation'] * input_history.pct_change()).T.sum()
            history = 100 * (history + 1).cumprod()
            history.name = 'Overall'
            history = pd.DataFrame(history)
        else:
            history = pd.DataFrame()
            for n in index[by].unique():
                allocation = index[index[by] == n]['Allocation']
                by_history = input_history[index[index[by] == n].index]
                by_history = (allocation * input_history.pct_change()).T.sum()
                by_history = 100 * (by_history + 1).cumprod()
                by_history.name = n
                history = pd.concat([history, by_history], axis=1)
                    
        return history
    
    def get_index(self, input_index, input_history):
        index = pd.DataFrame()
        index.index = input_index.index
        index['Price'] = input_history.iloc[-1]
        index['Share'] = input_index['Share']
        index['Valorisation'] = input_index['Share'] * index['Price']
        index['Allocation'] = index['Valorisation'] / index['Valorisation'].sum()
        index['Account'] = input_index['Account']
        index['Type'] = input_index['Type']
        index['Industry'] = input_index['Industry']
        return index
    
class index_table(BaseClass):
    def __init__(self, index, history):  
        super().__init__(history.index[-1]) 
        self.input_index = index
        self.input_history = history

    def compute(self):
        index = self.get_index(
            input_index=self.input_index,
            input_history=self.input_history
        )
        return index

    def show(self):
        index = self.compute()
        index['Allocation'] = index['Allocation'].apply(lambda x: f"{x * 100:.2f}%")
        with pd.option_context(
            'expand_frame_repr', None,
            'display.float_format', '{:,.1f}'.format,
            'display.colheader_justify', 'center'
        ):
            print(index)

# Class Composition
class stats_table(BaseClass):
    def __init__(self, index, history, market, riskfree):
        super().__init__(history.index[-1]) 
        self.range_dict = self.get_range_dict()
        self.freq_dict = self.get_freq_dict() 

        self.input_index = index
        self.input_history = history
        self.input_market = market
        self.input_riskfree = riskfree

    def compute(self, freq, range, by): 
        """Calculating statistics based on frequency and time range"""
        # Fetching historical data and resampling based on frequency and time range
        df_history = self.get_history(
            input_index=self.input_index, 
            input_history=self.input_history, 
            freq=freq,
            range=self.range_dict[range],
            by=by
        )

        # Fetching market data and risk-free rate for the specified time range and frequency
        market = self.input_market
        market = market.loc[self.range_dict[range]:]
        market = market.resample(freq).ffill()
        riskfree = self.input_riskfree

        # Get the correct freq for a given freq
        freq = self.freq_dict[freq]

        # Calculating various statistics for each asset in the data
        df_stats = pd.DataFrame() 
        for ticker in df_history.columns:
            history = df_history[ticker]
            if history.isna().any():
                # Handling missing values
                df_stats[ticker] = np.nan  
            else:
                # Calculating and formatting various statistics
                df_stats.at['Last price', ticker]       = "{:.1f}".format(history.iloc[-1])
                df_stats.at['Highest price', ticker]    = "{:.1f}".format(history.max())
                df_stats.at['Lowest price', ticker]     = "{:.1f}".format(history.min())
                df_stats.at['Last return', ticker]      = "{:.1f}%".format(history.pct_change().iloc[-1] * 100)
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
                #df_stats.at['Allocation', ticker]       = "{:.1f}%".format(self.allocation[ticker] * 100)

        # Displaying the statistics table
        with pd.option_context(
            'expand_frame_repr', None,
            'display.colheader_justify', 'center'
        ):
            print(df_stats)
        
    def show(self):
        """Displaying interactive controls for selecting frequency and time range"""
        # Creating interactive controls for selecting frequency and time range
        controls = widgets.interactive(
            self.compute,
            freq=widgets.Select(options=list(self.freq_dict.keys()), value='B'),
            range=widgets.Select(options=list(self.range_dict.keys()), value='1M'),
            by=widgets.Select(options=['All', 'Overall', 'Type', 'Account', 'Industry'], value='All'),
        )

        # Displaying the interactive controls
        display(controls)

# Class Comparaison
class chart_comparison(BaseClass):
    def __init__(self,index, history):
        super().__init__(history.index[-1]) 
        self.range_dict = self.get_range_dict()
        self.freq_dict = self.get_freq_dict() 
        self.input_history = history
        self.input_index = index
    
    def plot(self, freq, range, by='All', indexing=True):
        """Plot data from the DataFrame"""
        history = self.get_history(
            input_index=self.input_index, 
            input_history=self.input_history, 
            freq=freq,
            range=self.range_dict[range],
            by=by,
            indexing=indexing
        )

        ax = history.plot(figsize=(16, 8), title='Components returns comparison')
        ax.legend().remove()

        for column in history.columns:
            last_value = history[column].iloc[-1]
            last_date = history.index[-1]
            
            ax.text(last_date, last_value, f'{column}: {last_value:.2f}', 
                    verticalalignment='center', fontsize=10, 
                    bbox=dict(facecolor='white', edgecolor='none', pad=2))
        
        plt.show()
    
    def show(self, indexing=True):
        """Displaying interactive controls for plotting"""
        controls = widgets.interactive(
            self.plot,
            freq=widgets.Select(options=list(self.freq_dict.keys()), value='B'),
            range=widgets.Select(options=list(self.range_dict.keys()), value='1M'),
            by=widgets.Select(options=['All', 'Overall', 'Type', 'Account', 'Industry'], value='All'),
            indexing=indexing
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

