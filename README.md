# Index Report Project

This Python script provides a set of classes for performing various financial analysis tasks. The classes are designed to work with pandas DataFrames and are intended to be used in a Jupyter notebook environment.

## Classes

### BaseClass

The `BaseClass` is the base class that provides common functionalities for the other classes. It includes the following methods:

- `__init__(self, input_date)`: Initializes the class with a date.
- `get_range_dict(self)`: Returns a dictionary with date ranges for different time periods.
- `get_freq_dict(self)`: Returns a dictionary with frequency values for different time periods.
- `get_date(self)`: Returns the date.
- `get_history(self, input_index, input_history, freq='B', range='1Y', by='All', indexing=False)`: Returns a DataFrame with historical data based on the input index, history, frequency, range, and other parameters.
- `get_index(self, input_index, input_history)`: Returns a DataFrame with index data based on the input index and history.

### index_table

The `index_table` class inherits from `BaseClass` and is used to compute and display an index table. It includes the following methods:

- `__init__(self, index, history)`: Initializes the class with an index and history.
- `compute(self)`: Computes the index table.
- `show(self)`: Displays the index table.

### stats_table

The `stats_table` class also inherits from `BaseClass` and is used to compute and display a statistics table. It includes the following methods:

- `__init__(self, index, history, market, riskfree)`: Initializes the class with an index, history, market, and risk-free rate.
- `compute(self, freq, range, by)`: Computes the statistics table based on the frequency, range, and other parameters.
- `show(self)`: Displays the statistics table with interactive controls for selecting the frequency and time range.

### chart_comparison

The `chart_comparison` class inherits from `BaseClass` and is used to plot and compare data. It includes the following methods:

- `__init__(self, index, history)`: Initializes the class with an index and history.
- `plot(self, freq, range, by='All', indexing=True)`: Plots the data based on the frequency, range, and other parameters.
- `show(self, indexing=True)`: Displays the plot with interactive controls for selecting the frequency and time range.

## Libraries

The script uses the following libraries:

- pandas
- numpy
- mytoolkit.finance
- scipy.stats
- ipywidgets
- IPython.display
- matplotlib.pyplot
- matplotlib.ticker
- warnings

The script also ignores all warnings using `warnings.filterwarnings("ignore")`.

## Usage

To use the classes, you can import them into your Jupyter notebook and create an instance of the class with the appropriate input data. Then, you can call the `show` method to display the results of the analysis. For example:

```python
from my_script import stats_table

# Create an instance of the stats_table class
stats = stats_table(index, history, market, riskfree)

# Display the statistics table
stats.show()
```

You can replace `stats_table` with `index_table` or `chart_comparison` to use the other classes.
