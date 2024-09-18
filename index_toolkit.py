import pandas as pd
import numpy as np
import finance_toolkit as fin
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
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
            '1Y': date - pd.Timedelta(days=360),
            '5Y': date - pd.Timedelta(days=360 * 5),
            '10Y': date - pd.Timedelta(days=360 * 10),
        }


    def get_freq_dict(self):
        return {
            'D': 360,
            'B': 255,
            'W': 52,
            'M': 12
        }

    def get_date(self): 
        return self.date
    
    def get_index(self, input_index, input_history, by='All'):
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

        return index

    def get_history(self, input_index, input_history, freq='B', range='1Y', by='All', indexing=False):
        index = self.get_index(input_index=input_index, input_history=input_history, by=by)

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
    
    def get_by_list(self, index):
        by_list = ['All', 'Overall']
        by_list += [e for e in index.columns.to_list() if e not in ('Ticker', 'Component', 'Isin', 'Share')]

        return by_list
    
class index_table(BaseClass):
    def __init__(self, index, history):  
        super().__init__(history.index[-1]) 
        self.input_index = index
        self.input_history = history
        self.by_list = self.get_by_list(index)

    def compute(self, by):
        input_index = self.get_index(
            input_index=self.input_index,
            input_history=self.input_history,
            by=by
        )
        input_index['Allocation'] = input_index['Valorisation'] / input_index['Valorisation'].sum()
        if by == 'All':
            index = input_index
        elif by == 'Overall':
            index = pd.DataFrame()
            index.at['Overall', 'Valorisation'] = input_index['Valorisation'].sum()
            index.at['Overall', 'Allocation'] = 1
            index.at['Overall', 'Components'] = input_index['Valorisation'].count()
        else:
            index = pd.DataFrame(index=input_index[by].unique())
            index['Valorisation'] = input_index.groupby(by)[['Valorisation']].sum()
            index['Allocation'] = input_index.groupby(by)[['Allocation']].sum()
            index['Components'] = input_index.groupby(by)['Share'].count()

        index['Allocation'] = index['Allocation'].apply(lambda x: f"{x * 100:.2f}%")
        with pd.option_context(
            'expand_frame_repr', None,
            'display.float_format', '{:,.1f}'.format,
            'display.colheader_justify', 'center'
        ):
            print(index)

    def show(self):
        """Displaying interactive controls for selecting frequency and time range"""
        # Creating interactive controls for selecting frequency and time range
        controls = widgets.interactive(
            self.compute,
            by=widgets.Select(options=self.by_list, value='All'),
        )

        # Displaying the interactive controls
        display(controls)
        
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
        self.by_list = self.get_by_list(index)

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
                df_stats.at['Last price', ticker]       = "{:.2f}".format(history.iloc[-1])
                df_stats.at['Highest price', ticker]    = "{:.2f}".format(history.max())
                df_stats.at['Lowest price', ticker]     = "{:.2f}".format(history.min())
                df_stats.at['Last return', ticker]      = "{:.2f}%".format(history.pct_change().iloc[-1] * 100)
                df_stats.at['Cumulative return', ticker]= "{:.2f}%".format(((history.iloc[-1] - history.iloc[0]) / history.iloc[0]) * 100)
                df_stats.at['Avg 1y returns', ticker]      = "{:.2f}%".format(fin.avg_return(history, timeperiod=freq) * 100)
                df_stats.at['Avg 1y volatility', ticker]   = "{:.2f}%".format(fin.avg_volatility(history, timeperiod=freq) * 100)
                df_stats.at['Alpha', ticker]            = "{:.2f}%".format(fin.alpha(history, market) * 100)
                df_stats.at['Beta', ticker]             = "{:.2f}".format(fin.beta(history, market))
                df_stats.at['Sharpe ratio', ticker]     = "{:.2f}".format(fin.sharpe_ratio(history, riskfree, timeperiod=freq))
                df_stats.at['Calmar ratio', ticker]     = "{:.2f}".format(fin.calmar_ratio(history, riskfree, timeperiod=freq))
                df_stats.at['Max drawdown', ticker]     = "{:.2f}%".format(fin.max_drawdown(history) * 100)
                #df_stats.at['Skewness', ticker]         = "{:.1f}".format(stats.kurtosis(history))
                #df_stats.at['Kurtosis', ticker]         = "{:.1f}".format(stats.skew(history))
                #df_stats.at['3% VAR', ticker]           = "{:.2f}%".format(fin.var(history, freq='B', conf_level=0.03) * 100)

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
            by=widgets.Select(options=self.get_by_list(self.input_index), value='All'),
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
        self.by_list = self.get_by_list(index)
    
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
            by=widgets.Select(options=self.get_by_list(self.input_index), value='All'),
            indexing=indexing
        )
        display(controls)

