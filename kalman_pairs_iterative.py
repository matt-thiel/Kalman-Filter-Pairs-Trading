import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Tuple, Dict, List
from dataclasses import dataclass
from collections import deque
import yfinance as yf
import matplotlib.pyplot as plt

@dataclass
class StrategyState:
    """Class to store strategy state variables"""
    position: float = 0
    last_signal: float = 0
    warm_up_data: pd.DataFrame = None
    is_warmed_up: bool = False
    # Kalman filter state variables
    mu: float = 0
    gamma: float = 0
    P: np.ndarray = None
    # EMA state variables
    spread_ema: float = 0
    spread_var_ema: float = 0
    
    def __post_init__(self):
        self.P = 1e-5 * np.eye(2)
        self.warm_up_data = pd.DataFrame()

class KalmanIterative:
    def __init__(self, ticker1: str, ticker2: str, lookback_period: int = 120, threshold: float = 0.7, 
                 warm_up_period: int = 250, decay: float = 0.94):
        """
        Initialize KalmanIterative with strategy parameters.
        
        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            lookback_period: Period for calculations (default: 120)
            threshold: Z-score threshold for trading signals (default: 0.7)
            warm_up_period: Number of days needed before trading starts (default: 250)
            decay: EMA decay factor (default: 0.94)
        """
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.warm_up_period = warm_up_period
        self.decay = decay
        self.state = StrategyState()
        
        # Kalman filter parameters
        self.Tt = np.eye(2)
        self.Qt = 1e-5 * np.eye(2)
        self.Ht = 1e-3
        
    def _update_kalman(self, price1: float, price2: float) -> None:
        """Update Kalman filter estimates for mu and gamma"""
        # Prediction step
        x_pred = np.array([self.state.mu, self.state.gamma])
        P_pred = self.state.P + self.Qt
        
        # Update step
        F = np.array([1, price2])
        y = price1
        y_pred = np.dot(F, x_pred)
        
        S = np.dot(np.dot(F, P_pred), F.T) + self.Ht
        K = np.dot(P_pred, F.T) / S
        
        # Update state estimates
        innovation = y - y_pred
        x_new = x_pred + K * innovation
        self.state.mu = x_new[0]
        self.state.gamma = x_new[1]
        
        # Update covariance
        self.state.P = P_pred - np.outer(K, np.dot(F, P_pred))
    
    def _update_spread_stats(self, spread: float) -> None:
        """Update spread statistics using EMA"""
        if not self.state.is_warmed_up:
            # Initialize with historical statistics from warm-up period only
            price1 = np.log(self.state.warm_up_data[self.ticker1])
            price2 = np.log(self.state.warm_up_data[self.ticker2])
            w1 = 1
            w2 = -(1 + self.state.gamma)
            historical_spreads = (w1 * price1 + (-self.state.gamma) * price2) / (1 + self.state.gamma) - self.state.mu/(1 + self.state.gamma)
            self.state.spread_ema = historical_spreads.mean()
            self.state.spread_var_ema = historical_spreads.var()
        else:
            # Update mean and variance using EMA
            delta = spread - self.state.spread_ema
            self.state.spread_ema += (1 - self.decay) * delta
            self.state.spread_var_ema = (
                self.decay * self.state.spread_var_ema + 
                (1 - self.decay) * delta * delta
            )
    
    def _generate_signal(self, z_score: float) -> float:
        """Generate trading signal based on Z-score threshold."""
        if not self.state.is_warmed_up:
            return 0
            
        threshold_long = -self.threshold
        threshold_short = self.threshold
        
        if self.state.last_signal == 0:  # no position
            if z_score <= threshold_long:
                return 1
            elif z_score >= threshold_short:
                return -1
            return 0
        elif self.state.last_signal == 1:  # long position
            return 0 if z_score >= 0 else self.state.last_signal
        else:  # short position
            return 0 if z_score <= 0 else self.state.last_signal
    
    def _initialize_kalman_params(self) -> None:
        """Initialize mu and gamma using OLS on warm-up data only"""
        if len(self.state.warm_up_data) >= self.warm_up_period:
            # Use only warm-up period data for initialization
            train_data = self.state.warm_up_data.iloc[-self.warm_up_period:]
            price1 = np.log(train_data[self.ticker1])
            price2 = np.log(train_data[self.ticker2])
            
            # Perform OLS regression
            formula = f"{self.ticker1}~{self.ticker2}"
            model = smf.ols(formula=formula, data=np.log(train_data)).fit()
            
            # Set initial values
            self.state.mu = model.params[0]
            self.state.gamma = model.params[1]
    
    def step(self, prices_df: pd.DataFrame, current_idx: pd.Timestamp) -> Dict[str, float]:
        """
        Process one step of the strategy.
        
        Args:
            prices_df: DataFrame with all price data up to current_idx
            current_idx: Current timestamp being processed
        """
        # Get data up to current index (no lookahead)
        current_data = prices_df.loc[:current_idx]
        
        # Get current prices
        current_prices = {
            self.ticker1: current_data.loc[current_idx, self.ticker1],
            self.ticker2: current_data.loc[current_idx, self.ticker2]
        }
        
        # Update warm-up data with only what we need
        self.state.warm_up_data = current_data.iloc[-self.warm_up_period:] if len(current_data) > self.warm_up_period else current_data
        
        if len(self.state.warm_up_data) < self.warm_up_period:
            return {self.ticker1: 0, self.ticker2: 0}
        
        # Initialize parameters if not done yet
        if not self.state.is_warmed_up:
            self._initialize_kalman_params()
            self.state.is_warmed_up = True
        
        # Get log prices for current step
        price1 = np.log(current_prices[self.ticker1])
        price2 = np.log(current_prices[self.ticker2])
        
        # Update Kalman filter estimates
        self._update_kalman(price1, price2)
        
        # Calculate spread using updated formula
        # Matching palomar_strat.py implementation
        w1 = 1
        w2 = -(1 + self.state.gamma)
        spread = (w1 * price1 + (-self.state.gamma) * price2) / (1 + self.state.gamma) - self.state.mu/(1 + self.state.gamma)
        
        # Update spread statistics
        self._update_spread_stats(spread)
        
        # Calculate z-score
        if self.state.spread_var_ema > 0:
            z_score = (spread - self.state.spread_ema) / np.sqrt(self.state.spread_var_ema)
        else:
            z_score = 0
        
        # Generate signal
        signal = self._generate_signal(z_score)
        self.state.last_signal = signal
        
        # Calculate positions using weighted portfolio approach
        # Matching palomar_strat.py implementation
        if signal != 0:
            w1 = 1
            w2 = -(1 + self.state.gamma)
            norm = abs(w1) + abs(w2)
            pos1 = signal * w1 / norm
            pos2 = signal * w2 / norm
        else:
            pos1 = 0
            pos2 = 0
            
        return {self.ticker1: pos1, self.ticker2: pos2}

def run_strategy(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Run the strategy iteratively through the price series."""
    tickers = prices_df.columns
    strategy = KalmanIterative(tickers[0], tickers[1])
    positions = []
    
    # Ensure data is sorted by date
    #prices_df = prices_df.sort_index()
    
    for idx in prices_df.index:
        # Pass only data up to current index
        current_data = prices_df.loc[:idx]
        pos = strategy.step(current_data, idx)
        positions.append(pos)
    
    positions_df = pd.DataFrame(positions, index=prices_df.index)
    
    # Calculate returns without lookahead
    returns = pd.DataFrame(index=prices_df.index)
    for col in prices_df.columns:
        price_rets = prices_df[col].pct_change()
        returns[f'{col}_return'] = positions_df[col].shift(1) * price_rets
    
    returns['total_return'] = returns.sum(axis=1)
    
    return pd.concat([positions_df, returns], axis=1)

def main():
    # Example usage
    symbols = ['EWH', 'EWZ']
    start_date = '2004-01-01'
    end_date = '2015-12-31'
    
    # Download data
    prices = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    # Run strategy
    results = run_strategy(prices)
    
    # Calculate performance metrics
    cum_returns = (1 + results['total_return']).cumprod()
    sharpe = results['total_return'].mean() / results['total_return'].std() * np.sqrt(252)
    
    print("\nStrategy Performance:")
    print(f"Total Return: {cum_returns.iloc[-1]-1:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    cum_returns.plot(title='Cumulative Strategy Returns')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()