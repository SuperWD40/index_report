import pandas as pd
import numpy as np

def beta(asset: pd.Series, market: pd.Series) -> float:
    """
    #### Description:
    Beta is a measure of a financial instrument's sensitivity to market movements. A beta of 1 indicates the asset tends
    to move in line with the market, a beta greater than 1 suggests higher volatility, and a beta less than 1 indicates
    lower volatility compared to the market.

    #### Parameters:
    - asset (pd.Series): Time series data representing the returns of the asset.
    - market (pd.Series): Time series data representing the returns of the market.

    #### Returns:
    - float: Beta coefficient, which measures the asset's sensitivity to market movements.
    """
    # Combine asset and market returns, calculate covariance and market variance
    df = pd.concat([asset, market], axis=1).dropna().pct_change().dropna()
    covariance = df.cov().iloc[1, 0]
    market_variance = df.iloc[:, 1].var()

    # Calculate beta as the ratio of covariance to market variance
    return covariance / market_variance


def avg_return(history: pd.Series, timeperiod: int = 255) -> float:
    """
    #### Description:
    Calculate the average return of a financial instrument over a specified time period.

    #### Parameters:
    - history (pd.Series): Time series data representing the historical prices or returns of the financial instrument.
    - timeperiod (int, optional): The number of periods to consider for calculating the average return. Default is 255.

    #### Returns:
    - float: Average return over the specified time period.
    """
    returns = history.pct_change().dropna()
    return (1 + returns.sum()) ** (timeperiod/returns.count()) - 1

def avg_volatility(history: pd.Series, timeperiod: int = 255) -> float:
    """
    #### Description:
    Calculate the average volatility of a financial instrument over a specified time period.

    #### Parameters:
    - history (pd.Series): Time series data representing the historical prices or returns of the financial instrument.
    - timeperiod (int, optional): The number of periods to consider for calculating the average volatility. Default is 255.

    #### Returns:
    - float: Average volatility over the specified time period.
    """
    returns = history.pct_change(fill_method=None).dropna()
    return np.std(returns) * np.sqrt(timeperiod)

def max_drawdown(history: pd.Series) -> float:
    """
    #### Description:
    Max Drawdown is a measure of the largest loss from a peak to a trough in a financial instrument's historical performance.
    It is calculated as the percentage decline relative to the highest cumulative value.

    #### Parameters:
    - history (pd.Series): Time series data representing the historical prices or returns of the financial instrument.

    #### Returns:
    - float: Maximum drawdown, representing the largest percentage decline from a peak to a trough.
    """
    # Calculate the cumulative percentage change in price or returns
    index = 100 * (1 + history.pct_change()).cumprod()

    # Identify the peaks and calculate drawdowns
    peaks = index.cummax()
    drawdowns = (index - peaks) / peaks

    # Return the maximum drawdown
    return drawdowns.min()


def alpha(asset: pd.Series, market: pd.Series) -> float:
    """
    #### Description:
    Alpha measures the asset's performance relative to the market. A positive alpha indicates that the asset has outperformed
    the market, while a negative alpha suggests underperformance.

    #### Parameters:
    - asset (pd.Series): Time series data representing the historical performance of the asset.
    - market (pd.Series): Time series data representing the historical performance of the market.

    #### Returns:
    - float: Alpha, representing the excess return of the asset over the market.
    """
    # Calculate percentage change in asset and market
    asset_return = (asset.iloc[-1] - asset.iloc[0]) / asset.iloc[0]
    market_return = (market.iloc[-1] - market.iloc[0]) / market.iloc[0]

    # Calculate alpha as the difference in returns
    return asset_return - market_return


def sharpe_ratio(history: pd.Series, risk_free: float, timeperiod: int = 255) -> float:
    """
    #### Description:
    Calculate the Sharpe Ratio for a given financial instrument based on its historical performance.
    The Sharpe Ratio helps assess the investment's return per unit of risk taken.

    #### Parameters:
    - history (list or numpy array): Historical price or return data of the financial instrument.
    - risk_free (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and volatility. Default is 255.

    #### Returns:
    - float: The Sharpe Ratio, a measure of the instrument's risk-adjusted performance.
    """
    # Calculate average return and volatility using helper functions
    returns = avg_return(history=history, timeperiod=timeperiod)
    volatility = avg_volatility(history=history, timeperiod=timeperiod)
    
    # Calculate Sharpe Ratio using the formula
    return (returns - risk_free) / volatility

def calmar_ratio(history: pd.Series, risk_free: float, timeperiod: int = 255) -> float:
    """
    #### Description:
    Calculate the Calmar Ratio for a given financial instrument based on its historical performance.
    The Calmar ratio uses a financial instrument’s maximum drawdown as its sole measure of risk

    #### Parameters:
    - history (list or numpy array): Historical price or return data of the financial instrument.
    - risk_free (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and volatility. Default is 255.

    #### Returns:
    - float: The Sharpe Ratio, a measure of the instrument's risk-adjusted performance.
    """
    # Calculate average return and volatility using helper functions
    returns = avg_return(history=history, timeperiod=timeperiod)
    maxdrawdown = max_drawdown(history=history)
    
    if abs(maxdrawdown) != 0:
        # Calculate Sharpe Ratio using the formula
        return (returns - risk_free) / abs(maxdrawdown)
    else:
        return np.nan

def indexing(data: pd.Series, base: int = 100, weight: pd.Series = None) -> pd.Series:
    """
    #### Description:
    The function calculates the indexed values based on the percentage change and cumulative product of the input data.
    Optionally, it supports weighting the components if a weight Series is provided.

    #### Parameters:
    - data (pandas.Series or pandas.DataFrame): Time series data representing the value of a financial instrument.
    - base (float, optional): Initial base value for indexing. Default is 100.
    - weight (pandas.Series, optional): Weighting factor for different components in the data. Default is None.

    #### Returns:
    - pandas.Series: Indexed time series data.
    """
    # Calculate percentage change and cumulative product for indexing
    data = base * (data.pct_change() + 1).cumprod()

    # Set the initial base value for indexing
    data.loc[data.index[0]] = base

    # Apply weights if provided
    if isinstance(weight, pd.Series):
        data = (data * weight).sum(axis=1)
    return data

def var(history, freq: str, conf_level: float) -> float:
    """
    #### Description:
    Calculate the Value at Risk (VaR) for a given financial instrument based on its historical performance.
    VaR measures the maximum potential loss at a specified confidence level over a given time horizon.

    #### Parameters:
    - history (pandas Series or DataFrame): Historical price or return data of the financial instrument.
    - freq (str): The frequency of the data, e.g., 'D' for daily, 'M' for monthly.
    - conf_level (float): The confidence level for VaR calculation, typically between 0 and 1.

    #### Returns:
    - float: The Value at Risk (VaR), representing the maximum potential loss at the specified confidence level.
    """
    # Resample the historical data based on the specified frequency
    history = history.resample(freq).ffill()

    # Calculate the percentage change and drop any NaN values
    history = history.pct_change().dropna()

    # Sort the percentage change data
    history = history.sort_values()

    # Calculate VaR at the specified confidence level
    var = history.iloc[round(len(history) * conf_level)]

    # Return the absolute value of VaR
    return abs(var)

def momentum(history: pd.Series, period: int, differential: str ='last', method: str ='normal') -> pd.Series:
    """
    #### Description:
    Calculates the momentum of a time series data over a specified period.

    Momentum measures the rate of change in the price of a security over a specific period.
    It is commonly used by traders and analysts to identify trends and potential buying or selling signals.

    #### Parameters:
    - history (pd.Series or pd.DataFrame): Time series data representing the price or value of a security.
    - period (int): The number of data points over which momentum is calculated.
    - differential (str, default='last'): Determines how the reference value is calculated within the period.
        - 'last': Uses the last value within the rolling window as the reference value.
        - 'mean': Uses the mean (average) value within the rolling window as the reference value.
    - method (str, default='normal'): Determines the method of calculating momentum.
        - 'normal': Calculates the difference between the current value and the reference value.
        - 'roc': Calculates the Rate of Change (ROC) as a percentage.
        - 'roclog': Calculates the logarithmic Rate of Change (ROC) as a percentage.

    #### Returns:
    - pd.Series: Series containing the calculated momentum values.
    """
    # Calculate the reference values based on the selected method
    if differential == 'last':
        ctx = history.rolling(window=period).apply(lambda x: x[0], raw=True).dropna()
    elif differential == 'mean':
        ctx = history.rolling(window=period).mean().dropna()

    # Reindex the original history to align with the reference values
    ct = history.reindex(ctx.index)

    # Calculate momentum based on the selected method
    if method == 'normal':
        mo = ct - ctx
    elif method == 'roc':
        mo = 100 * ct / ctx
    elif method == 'roclog':
        mo = 100 * np.log(np.array(ct / ctx))
        mo = pd.Series(mo, index=history.index[-len(ct):])
    return mo






