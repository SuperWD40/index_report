import pandas as pd
import numpy as np

import finance_toolkit as fin

import ipywidgets as widgets
from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

class BaseClass:
    def __init__(self, input_date): 
        self.date = pd.to_datetime(input_date, dayfirst=True).tz_localize(None)

class BaseClass:
    def __init__(self, input_date): 
        # Initialize the class with the input date, parsing it to datetime and removing timezone info
        self.date = pd.to_datetime(input_date, dayfirst=True).tz_localize(None)

    def get_range_dict(self):  
        # Create a dictionary of time ranges relative to the current date
        date = self.date 
        return {
            '1W': date - pd.Timedelta(days=7),                  # 1 Week ago
            '1M': date - pd.Timedelta(days=30),                 # 1 Month ago
            'YTD': pd.Timestamp(year=date.year, month=1, day=1),# Year-to-date
            '1Y': date - pd.Timedelta(days=360),                # 1 Year ago
            '5Y': date - pd.Timedelta(days=360 * 5),            # 5 Years ago
            '10Y': date - pd.Timedelta(days=360 * 10),          # 10 Years ago
        }

    def get_freq_dict(self):
        # Dictionary for different resampling frequencies and their respective annualization factors
        return {
            'D': 360,  # Daily frequency, 360 days/year
            'B': 255,  # Business days frequency, 255 days/year
            'W': 52,   # Weekly frequency, 52 weeks/year
            'M': 12    # Monthly frequency, 12 months/year
        }

    def get_date(self): 
        # Return the current date
        return self.date
    
    def get_index(self, input_index, input_history, by='All'):
        # Prepare a DataFrame with price, shares, and valorisation (market value)
        index = pd.DataFrame()
        index.index = input_index.index
        index['Price'] = input_history.iloc[-1]  # Use the last available price
        index['Share'] = input_index['Share']    # Number of shares
        index['Valorisation'] = input_index['Share'] * index['Price']  # Market value
        index['Allocation'] = index['Valorisation'] / index['Valorisation'].sum()  # Proportional allocation

        # If a specific grouping ('by') is provided, normalize allocations by group
        if by not in ['All', 'Overall']:
            index[by] = input_index[by]
            by_index = index.groupby(by)['Allocation'].transform('sum')
            index['Allocation'] = index['Allocation'] / by_index

        return index

    def get_history(self, input_index, input_history, freq='B', range='1Y', by='All', indexing=False, comparing=None):
        # Retrieve historical data for the selected range, frequency, and grouping
        index = self.get_index(
            input_index=input_index, 
            input_history=input_history, 
            by=by
        )

        # Trim history data based on the provided date range
        input_history = input_history.loc[range:]
        # Resample data to the desired frequency
        input_history = input_history.resample(freq).ffill()
        
        # Handle different cases for grouping and indexing
        if by == 'All' and not indexing:
            history = input_history
        elif by == 'All' and indexing:
            # Compute cumulative returns if indexing is requested
            history = 100 * (input_history.pct_change() + 1).cumprod()
            history.loc[history.index[0]] = 100  # Set the first value to 100
        elif by == 'Overall':
            # Compute weighted cumulative returns for the entire portfolio
            history = (index['Allocation'] * input_history).T.sum()
            history = 100 * (history.pct_change() + 1).cumprod()
            history.loc[history.index[0]] = 100  # Set the first value to 100
            history.name = 'Overall'
            history = pd.DataFrame(history)
        else:
            # Compute historical returns by specific grouping (e.g., sectors, regions)
            history = pd.DataFrame()
            for n in index[by].unique():
                allocation = index[index[by] == n]['Allocation']
                by_history = input_history[index[index[by] == n].index]
                by_history = (allocation * input_history).T.sum()
                by_history = 100 * (by_history.pct_change() + 1).cumprod()
                by_history.loc[by_history.index[0]] = 100
                by_history.name = n
                history = pd.concat([history, by_history], axis=1)
        
        # If a comparison series is provided, compute its historical performance as well
        if isinstance(comparing, pd.Series):
            comparing = comparing.loc[range:]
            comparing = comparing.resample(freq).ffill()
            if indexing:
                comparing = 100 * (comparing.pct_change() + 1).cumprod()
                comparing.loc[comparing.index[0]] = 100
            history[comparing.name] = comparing
        
        history = history.dropna()  # Remove any NaN values from the history
        
        return history
    
    def get_by_list(self, index):
        # Create a list of valid 'by' categories for grouping (excluding default columns like 'Ticker', 'Share')
        by_list = ['All', 'Overall']
        by_list += [e for e in index.columns.to_list() if e not in ('Ticker', 'Component', 'Isin', 'Share')]
        return by_list



class risk_chart(BaseClass):
    def __init__(self, index, history, market):
        """Initializes the risk_chart class with provided index, historical data, and market info."""
        # Calls the parent class constructor, passing the last available date
        super().__init__(history.index[-1])  
        self.input_index = index  
        self.input_history = history
        self.by_list = self.get_by_list(index)
        self.range_dict = self.get_range_dict()
        self.freq_dict = self.get_freq_dict()
        self.market = market

    def compute(self, freq='B', range='1M', by='All'):
        """Computes statistics such as average returns, volatility, and valuation based on user inputs."""
        # Gets historical data resampled to the chosen frequency and time range, with optional grouping
        history = self.get_history(
            input_index=self.input_index, 
            input_history=self.input_history, 
            freq=freq, 
            range=self.range_dict[range], 
            by=by,  
        )

        # Gets the index data to be used for valuation, based on the selected grouping
        index = self.get_index(
            input_index=self.input_index,
            input_history=self.input_history,
            by=by  # Grouping criterion for index data
        )

        # Handles different cases for grouping to calculate total valuation (e.g., by 'All', 'Overall', or specific groups)
        if by == 'All':
            valo = index['Valorisation']  # No grouping, take the entire valuation
        elif by == 'Overall':
            valo = pd.Series(index['Valorisation'].sum(), name='Valorisation', index=['Overall'])  # Sum of all valuations
        else:
            valo = index.groupby([by])['Valorisation'].sum()  # Group by the specified category and sum valuations

        # Initialize an empty DataFrame to store computed statistics for each ticker (asset)
        stats = pd.DataFrame()
        for ticker in history.columns:
            if history[ticker].isna().any():
                stats[ticker] = np.nan  # If data contains NaN values, assign NaN for this asset
            else:
                # Compute and store 1-year average return and volatility, and the valuation for each asset
                stats.at[ticker, 'Avg 1y returns'] = fin.avg_return(history[ticker], timeperiod=self.freq_dict[freq]) * 100
                stats.at[ticker, 'Avg 1y volatility'] = fin.avg_volatility(history[ticker], timeperiod=self.freq_dict[freq]) * 100
                stats.at[ticker, 'Valorisation'] = valo[ticker]

        return stats  # Return the DataFrame containing the computed statistics

    def plot(self, freq='B', range='1M', by='All'):
        """Plots a scatter plot of 1-year average returns vs volatility, sized by valuation."""
        # Compute the necessary statistics (returns, volatility, valuation) based on selected options
        df = self.compute(freq=freq, range=range, by=by)

        # Set up the figure size for the plot
        plt.figure(figsize=(16, 8))

        # Create a scatter plot where each point represents an asset; its size reflects the asset's valuation
        plt.scatter(
            df['Avg 1y volatility'],  # x-axis: 1-year average volatility
            df['Avg 1y returns'],  # y-axis: 1-year average returns
            s=df['Valorisation'],  # Point size represents valuation
            alpha=0.75,  # Transparency level
            edgecolors='w',  # White edge color for points
            linewidth=1  # Width of the point edge
        )

        # Add labels for each point (ticker names)
        for label, x, y in zip(df.index, df['Avg 1y volatility'], df['Avg 1y returns']):
            plt.text(x, y, label, fontsize=9, ha='right')  # Label positioning and font size

        # Set the plot title and labels for the x and y axes
        plt.title('1y average Returns vs Volatility with Valorisation')
        plt.xlabel('Avg 1y volatility')
        plt.ylabel('Avg 1y returns')

        # Disable the grid for a cleaner look
        plt.grid(False)

        # Format the x-axis and y-axis ticks as percentages
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}%'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}%'))

        # Display the plot
        plt.show()

    def show(self):
        """Creates interactive controls for dynamically adjusting the plot."""
        # Create interactive widgets for frequency, time range, and grouping selection
        controls = widgets.interactive(
            self.plot,
            freq=widgets.Select(options=list(self.freq_dict.keys()), value='B'),
            range=widgets.Select(options=list(self.range_dict.keys()), value='1M'),
            by=widgets.Select(options=self.get_by_list(self.input_index), value='All'),
        )

        # Display the interactive controls for user interaction
        display(controls)
    


class index_table(BaseClass):
    def __init__(self, index, history):
        """Initializes the index_table class with an index and historical data."""
        super().__init__(history.index[-1])
        self.input_index = index
        self.input_history = history
        self.by_list = self.get_by_list(index)

    def compute(self, by='All'):
        """Computes and displays a table summarizing the index allocation and valuation."""
        # Retrieves the index data, potentially grouping it by the specified criterion
        input_index = self.get_index(
            input_index=self.input_index,
            input_history=self.input_history,
            by=by 
        )

        # Calculates the allocation of each component as a percentage of the total valuation
        input_index['Allocation'] = input_index['Valorisation'] / input_index['Valorisation'].sum()

        # Conditional handling based on the grouping option ('All', 'Overall', or specific groups)
        if by == 'All':
            # No grouping, take the entire index
            index = input_index  
        elif by == 'Overall':
            # Create a DataFrame summarizing the overall values for the index
            index = pd.DataFrame()
            index.at['Overall', 'Valorisation'] = input_index['Valorisation'].sum()  # Total valuation
            index.at['Overall', 'Allocation'] = 1  # Overall allocation is 100%
            index.at['Overall', 'Components'] = input_index['Valorisation'].count()  # Count of components
        else:
            # Create a DataFrame to summarize the index by the specified grouping
            index = pd.DataFrame(index=input_index[by].unique())
            index['Valorisation'] = input_index.groupby(by)[['Valorisation']].sum()
            index['Allocation'] = input_index.groupby(by)[['Allocation']].sum() 
            index['Components'] = input_index.groupby(by)['Share'].count() 

        # Format the Allocation column to display percentages
        index['Allocation'] = index['Allocation'].apply(lambda x: f"{x * 100:.2f}%")

        # Set display options for the DataFrame output
        with pd.option_context(
            'expand_frame_repr', None,
            'display.float_format', '{:,.1f}'.format,
            'display.colheader_justify', 'center'
        ):
            print(index)  # Print the resulting index summary table

    def show(self):
        """Creates interactive controls for dynamically adjusting the index summary display."""
        # Create interactive widget for selecting the grouping criterion
        controls = widgets.interactive(
            self.compute,
            by=widgets.Select(options=self.by_list, value='All'),
        )

        # Display the interactive controls for user interaction
        display(controls)

        


class stats_table(BaseClass):
    def __init__(self, index, history, market, riskfree):
        """Initializes the stats_table class with an index, historical data, market data, and a risk-free rate."""
        super().__init__(history.index[-1])
        self.range_dict = self.get_range_dict()
        self.freq_dict = self.get_freq_dict()
        self.input_index = index
        self.input_history = history
        self.input_market = market
        self.input_riskfree = riskfree
        self.by_list = self.get_by_list(index)

    def compute(self, freq='B', range='1M', by='All', comparing=True):
        """Computes and displays statistical metrics for the index, optionally comparing to market data."""
        # If comparing is True, use the input market data
        if comparing:
            comparing = self.input_market
        
        # Retrieve historical data for the specified index
        df_history = self.get_history(
            input_index=self.input_index, 
            input_history=self.input_history, 
            freq=freq,
            range=self.range_dict[range],
            by=by,
            comparing=comparing
        )

        # Process the market data for the specified range and frequency
        market = self.input_market
        market = market.loc[self.range_dict[range]:]
        market = market.resample(freq).ffill()
        riskfree = self.input_riskfree

        freq = self.freq_dict[freq]  # Retrieve the frequency for calculations

        df_stats = pd.DataFrame()  # Create an empty DataFrame to hold the statistics
        for ticker in df_history.columns:  # Iterate over each asset in the historical data
            history = df_history[ticker]  # Get the historical data for the current asset
            if history.isna().any():  # Check for missing values
                df_stats[ticker] = np.nan  # If any value is missing, assign NaN
            else:
                # Calculate various statistics for the asset and store in the DataFrame
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

                # Optional calculations (commented out)
                # df_stats.at['Skewness', ticker]         = "{:.1f}".format(stats.kurtosis(history))
                # df_stats.at['Kurtosis', ticker]         = "{:.1f}".format(stats.skew(history))
                # df_stats.at['3% VAR', ticker]           = "{:.2f}%".format(fin.var(history, freq='B', conf_level=0.03) * 100)

        # Set display options for the DataFrame output
        with pd.option_context(
            'expand_frame_repr', None,  # Don't limit frame representation
            'display.colheader_justify', 'center'  # Center-align column headers
        ):
            print(df_stats)  # Print the resulting statistics table

    def show(self, comparing=True):
        """Creates interactive controls for dynamically adjusting the statistical metrics display."""
        # Create interactive widget for selecting frequency, range, grouping, and comparison
        controls = widgets.interactive(
            self.compute,
            freq=widgets.Select(options=list(self.freq_dict.keys()), value='B'),
            range=widgets.Select(options=list(self.range_dict.keys()), value='1M'),
            by=widgets.Select(options=self.get_by_list(self.input_index), value='All'),
            comparing=widgets.Checkbox(value=comparing)
        )

        # Display the interactive controls for user interaction
        display(controls)




class chart_comparison(BaseClass):
    def __init__(self, index, history, market):
        """Initializes the chart_comparison class with an index, historical data, and market data."""
        super().__init__(history.index[-1])  
        self.range_dict = self.get_range_dict()  
        self.freq_dict = self.get_freq_dict()  
        self.input_history = history  
        self.input_index = index  
        self.by_list = self.get_by_list(index) 
        self.market = market  

    def plot(self, freq='B', range='1M', by='All', indexing=True, comparing=True):
        """Plots the returns of the index components, optionally comparing to market data."""
        # If comparing is True, use the input market data
        if comparing:
            comparing = self.market
        
        # Retrieve historical data for the specified index
        history = self.get_history(
            input_index=self.input_index, 
            input_history=self.input_history, 
            freq=freq,
            range=self.range_dict[range],
            by=by,
            indexing=indexing, 
            comparing=comparing  
        )

        # Create a plot for the historical data
        ax = history.plot(figsize=(16, 8), title='Components returns comparison') 
        ax.legend().remove()  # Remove the default legend for cleaner visualization

        # Annotate the last value of each column on the plot
        for column in history.columns:
            last_value = history[column].iloc[-1]
            last_date = history.index[-1]
            
            # Add a text box with the last value on the plot
            ax.text(last_date, last_value, f'{column}: {last_value:.2f}', 
                    verticalalignment='center', fontsize=10, 
                    bbox=dict(facecolor='white', edgecolor='none', pad=2))  # Customize text box appearance
        
        plt.show() 

    def show(self, indexing=True, comparing=True):
        """Creates interactive controls for dynamically adjusting the chart display."""
        # Create interactive widget for selecting frequency, range, grouping, indexing, and comparison
        controls = widgets.interactive(
            self.plot,
            freq=widgets.Select(options=list(self.freq_dict.keys()), value='B'), 
            range=widgets.Select(options=list(self.range_dict.keys()), value='1M'),
            by=widgets.Select(options=self.get_by_list(self.input_index), value='All'), 
            indexing=widgets.Checkbox(value=indexing),
            comparing=widgets.Checkbox(value=comparing) 
        )

        # Display the interactive controls for user interaction
        display(controls)



class return_table(BaseClass):
    def __init__(self, index, history, market):
        """Initializes the corr_table class with an index, historical data, and market data."""
        super().__init__(history.index[-1])  
        self.range_dict = self.get_range_dict()  
        self.freq_dict = self.get_freq_dict()  
        self.by_list = self.get_by_list(index)  
        self.input_index = index  
        self.input_history = history  
        self.market = market  

    def plot(self, freq='B', range='1M', by='All', comparing=True):
        """Plots the correlation matrix of the returns for the index components."""
        # If comparing is True, use the input market data
        if comparing:
            comparing = self.market
        
        # Retrieve historical data for the specified index
        history = self.get_history(
            input_index=self.input_index, 
            input_history=self.input_history, 
            freq=freq,
            range=self.range_dict[range],
            by=by,
            comparing=comparing  # Market data to compare against
        )

        # Calculate the correlation matrix based on percentage changes in historical data
        returns = history.pct_change().T
        returns.columns = returns.columns.strftime('%Y-%m-%d')
        returns = returns.dropna(axis=1)

        # Set up the heatmap for visualization
        plt.figure(figsize=(16, len(returns.T)))  # Set figure size
        sns.heatmap(returns, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5, center=0, cbar=False)  
        # Draw the heatmap with annotations, color map, and formatting options

        plt.title("Component's returns")  

        plt.tight_layout()  
        plt.show() 

    def show(self, comparing=True):
        """Creates interactive controls for dynamically adjusting the correlation matrix display."""
        # Create interactive widget for selecting frequency, range, grouping, and comparison
        controls = widgets.interactive(
            self.plot,
            freq=widgets.Select(options=list(self.freq_dict.keys()), value='B'),  
            range=widgets.Select(options=list(self.range_dict.keys()), value='1M'),  
            by=widgets.Select(options=self.by_list, value='All'),  
            comparing=widgets.Checkbox(value=comparing)  
        )

        # Display the interactive controls for user interaction
        display(controls)
