import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Dict, List
from dataclasses import dataclass
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.tsa.stattools import coint

@dataclass
class StrategyState:
    """Class to store strategy state variables"""
    position: float = 0
    last_signal: float = 0
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

class DynamicPairsTrader:
    def __init__(self, universe: List[str], lookback_period: int = 120, threshold: float = 0.7, 
                 warm_up_period: int = 250, decay: float = 0.94, rebalance_period: int = 126,
                 max_pairs: int = 5, coint_threshold: float = 0.05):
        """
        Initialize DynamicPairsTrader with strategy parameters.
        
        Args:
            universe: List of ticker symbols to consider
            lookback_period: Period for calculations (default: 120)
            threshold: Z-score threshold for trading signals (default: 0.7)
            warm_up_period: Number of days needed before trading starts (default: 250)
            decay: EMA decay factor (default: 0.94)
            rebalance_period: Number of days between pair selections (default: 126)
            max_pairs: Maximum number of pairs to trade (default: 5)
            coint_threshold: P-value threshold for cointegration test (default: 0.05)
        """
        self.universe = universe
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.warm_up_period = warm_up_period
        self.decay = decay
        self.rebalance_period = rebalance_period
        self.max_pairs = max_pairs
        self.coint_threshold = coint_threshold
        
        self.active_pairs = []
        self.pair_states = {}
        
        # Kalman filter parameters
        self.Tt = np.eye(2)
        self.Qt = 1e-5 * np.eye(2)
        self.Ht = 1e-3

    def find_cointegrated_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find cointegrated pairs from the universe."""
        n = len(self.universe)
        pvalue_matrix = np.ones((n, n))
        pairs = []
        
        # Calculate cointegration p-values
        for i, j in combinations(range(n), 2):
            ticker1 = self.universe[i]
            ticker2 = self.universe[j]
            
            # Skip if price data is missing
            if ticker1 not in prices.columns or ticker2 not in prices.columns:
                continue
                
            # Get price series
            p1 = prices[ticker1]
            p2 = prices[ticker2]
            
            # Test for cointegration
            _, pvalue, _ = coint(p1, p2)
            pvalue_matrix[i, j] = pvalue
            pvalue_matrix[j, i] = pvalue
            
            if pvalue < self.coint_threshold:
                pairs.append((ticker1, ticker2))
        
        # Sort pairs by p-value and take top max_pairs
        pairs.sort(key=lambda x: pvalue_matrix[self.universe.index(x[0]), 
                                             self.universe.index(x[1])])
        return pairs[:self.max_pairs]

    def _update_kalman(self, price1: float, price2: float, state: StrategyState) -> None:
        """Update Kalman filter estimates for mu and gamma"""
        # Prediction step
        x_pred = np.array([state.mu, state.gamma])
        P_pred = state.P + self.Qt
        
        # Update step
        F = np.array([1, price2])
        y = price1
        y_pred = np.dot(F, x_pred)
        
        S = np.dot(np.dot(F, P_pred), F.T) + self.Ht
        K = np.dot(P_pred, F.T) / S
        
        # Update state estimates
        innovation = y - y_pred
        x_new = x_pred + K * innovation
        state.mu = x_new[0]
        state.gamma = x_new[1]
        
        # Update covariance
        state.P = P_pred - np.outer(K, np.dot(F, P_pred))
    
    def _update_spread_stats(self, spread: float, state: StrategyState) -> None:
        """Update spread statistics using EMA"""
        if not state.is_warmed_up:
            state.spread_ema = spread
            state.spread_var_ema = 0.0001
        else:
            # Update mean
            delta = spread - state.spread_ema
            state.spread_ema += (1 - self.decay) * delta
            
            # Update variance
            state.spread_var_ema = (
                self.decay * state.spread_var_ema + 
                (1 - self.decay) * delta * delta
            )
    
    def _generate_signal(self, z_score: float, state: StrategyState) -> float:
        """Generate trading signal based on Z-score threshold."""
        if not state.is_warmed_up:
            return 0
            
        threshold_long = -self.threshold
        threshold_short = self.threshold
        
        if state.last_signal == 0:  # no position
            if z_score <= threshold_long:
                return 1
            elif z_score >= threshold_short:
                return -1
            return 0
        elif state.last_signal == 1:  # long position
            return 0 if z_score >= 0 else state.last_signal
        else:  # short position
            return 0 if z_score <= 0 else state.last_signal

    def step(self, prices: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Process one step of the strategy.
        
        Args:
            prices: DataFrame with historical prices up to current date
            current_date: Current timestamp
        
        Returns:
            Dictionary with positions for each asset
        """
        # Check if we need to rebalance pairs
        if len(self.active_pairs) == 0 or len(prices) % self.rebalance_period == 0:
            # Find new pairs
            training_data = prices.loc[:current_date]
            new_pairs = self.find_cointegrated_pairs(training_data)
            
            # Close positions for old pairs
            positions = {ticker: 0 for ticker in self.universe}
            
            # Initialize states for new pairs
            self.active_pairs = new_pairs
            self.pair_states = {
                pair: StrategyState() for pair in new_pairs
            }
            
            return positions
        
        # Calculate positions for active pairs
        positions = {ticker: 0 for ticker in self.universe}
        
        for (ticker1, ticker2) in self.active_pairs:
            state = self.pair_states[(ticker1, ticker2)]
            
            price1 = np.log(prices.loc[current_date, ticker1])
            price2 = np.log(prices.loc[current_date, ticker2])
            
            # Update Kalman filter estimates
            self._update_kalman(price1, price2, state)
            
            # Calculate spread
            w1 = 1
            w2 = -state.gamma
            spread = w1 * price1 + w2 * price2 - state.mu/(1 + state.gamma)
            
            # Update spread statistics
            self._update_spread_stats(spread, state)
            
            # Calculate z-score
            if state.spread_var_ema > 0:
                z_score = (spread - state.spread_ema) / np.sqrt(state.spread_var_ema)
            else:
                z_score = 0
            
            # Generate signal
            signal = self._generate_signal(z_score, state)
            state.last_signal = signal
            
            if not state.is_warmed_up and len(prices) >= self.warm_up_period:
                state.is_warmed_up = True
            
            # Calculate positions
            if signal != 0:
                w1 = 1
                w2 = -(1 + state.gamma)
                norm = abs(w1) + abs(w2)
                positions[ticker1] += signal * w1 / norm
                positions[ticker2] += signal * w2 / norm
        
        # Normalize positions across all pairs
        if len(self.active_pairs) > 0:
            scale = 1.0 / len(self.active_pairs)
            positions = {k: v * scale for k, v in positions.items()}
            
        return positions

class DistancePairsTrader:
    def __init__(self, universe: List[str], lookback_period: int = 120, threshold: float = 2.0,
                 warm_up_period: int = 250, rebalance_period: int = 126, max_pairs: int = 1):
        """
        Initialize DistancePairsTrader with strategy parameters.
        
        Args:
            universe: List of ticker symbols to consider
            lookback_period: Period for calculations (default: 120)
            threshold: Standard deviation threshold for trading signals (default: 2.0)
            warm_up_period: Number of days needed before trading starts (default: 250)
            rebalance_period: Number of days between pair selections (default: 126)
            max_pairs: Maximum number of pairs to trade (default: 5)
        """
        self.universe = universe
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.warm_up_period = warm_up_period
        self.rebalance_period = rebalance_period
        self.max_pairs = max_pairs
        
        self.active_pairs = []
        self.pair_states = {}

    def calculate_distance(self, price1: np.ndarray, price2: np.ndarray) -> float:
        """Calculate the distance between two normalized price series."""
        # Normalize prices to start at 1
        norm_price1 = price1 / price1[0]
        norm_price2 = price2 / price2[0]
        
        # Calculate sum of squared differences
        distance = np.sum((norm_price1 - norm_price2) ** 2)
        return distance

    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str]]:
        """Find pairs with the smallest price distance."""
        n = len(self.universe)
        distance_matrix = np.zeros((n, n))
        pairs = []
        
        # Calculate distance matrix
        for i, j in combinations(range(n), 2):
            ticker1 = self.universe[i]
            ticker2 = self.universe[j]
            
            # Skip if price data is missing
            if ticker1 not in prices.columns or ticker2 not in prices.columns:
                continue
                
            # Get price series
            p1 = prices[ticker1].values[-self.lookback_period:]
            p2 = prices[ticker2].values[-self.lookback_period:]
            
            # Calculate distance
            dist = self.calculate_distance(p1, p2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
            pairs.append((ticker1, ticker2, dist))
        
        # Sort pairs by distance and take top max_pairs
        pairs.sort(key=lambda x: x[2])
        return [(p[0], p[1]) for p in pairs[:self.max_pairs]]

    def calculate_spread(self, price1: np.ndarray, price2: np.ndarray) -> Tuple[float, float, float]:
        """Calculate the spread and its statistics using normalized prices."""
        # Normalize prices
        norm_price1 = price1 / price1[0]
        norm_price2 = price2 / price2[0]
        
        # Calculate spread as difference between normalized prices
        spread = norm_price1 - norm_price2
        
        # Calculate spread statistics
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        current_spread = spread[-1]
        
        return current_spread, spread_mean, spread_std

    def step(self, prices: pd.DataFrame, current_date: pd.Timestamp) -> Dict[str, float]:
        """
        Process one step of the strategy.
        
        Args:
            prices: DataFrame with historical prices up to current date
            current_date: Current timestamp
        
        Returns:
            Dictionary with positions for each asset
        """
        # Check if we need to rebalance pairs
        if len(self.active_pairs) == 0 or len(prices) % self.rebalance_period == 0:
            # Find new pairs
            training_data = prices.loc[:current_date]
            new_pairs = self.find_pairs(training_data)
            
            # Close positions for old pairs
            positions = {ticker: 0 for ticker in self.universe}
            
            # Set new active pairs
            self.active_pairs = new_pairs
            return positions
        
        # Calculate positions for active pairs
        positions = {ticker: 0 for ticker in self.universe}
        
        if len(prices) < self.warm_up_period:
            return positions
            
        for (ticker1, ticker2) in self.active_pairs:
            # Get historical prices for lookback period
            hist_price1 = prices[ticker1].values[-self.lookback_period:]
            hist_price2 = prices[ticker2].values[-self.lookback_period:]
            
            # Calculate spread statistics
            current_spread, spread_mean, spread_std = self.calculate_spread(hist_price1, hist_price2)
            
            # Calculate z-score
            if spread_std > 0:
                z_score = (current_spread - spread_mean) / spread_std
            else:
                z_score = 0
            
            # Generate positions based on z-score
            if z_score <= -self.threshold:
                # Long spread
                positions[ticker1] += 0.5
                positions[ticker2] -= 0.5
            elif z_score >= self.threshold:
                # Short spread
                positions[ticker1] -= 0.5
                positions[ticker2] += 0.5
        
        # Normalize positions across all pairs
        if len(self.active_pairs) > 0:
            scale = 1.0 / len(self.active_pairs)
            positions = {k: v * scale for k, v in positions.items()}
            
        return positions

def run_strategy(prices_df: pd.DataFrame, max_pairs: int = 3, warm_up_period: int = 250, **kwargs) -> pd.DataFrame:
    """Run the dynamic strategy iteratively through the price series."""
    universe = list(prices_df.columns)
    strategy = DynamicPairsTrader(universe, max_pairs=max_pairs, warm_up_period=warm_up_period, **kwargs)
    positions = []
    dates = []
    
    for idx in prices_df.index:
        current_prices = prices_df.loc[:idx]
        
        # During warm-up period, store zero positions
        if len(current_prices) < warm_up_period:
            pos = {ticker: 0 for ticker in universe}
        else:
            pos = strategy.step(current_prices, idx)
            
        positions.append(pos)
        dates.append(idx)
    
    # Create positions DataFrame
    positions_df = pd.DataFrame(positions, index=dates)
    
    # Calculate returns
    returns = pd.DataFrame(index=prices_df.index)
    for col in prices_df.columns:
        price_rets = prices_df[col].pct_change()
        returns[f'{col}_return'] = positions_df[col].shift(1) * price_rets
        
    returns['total_return'] = returns.sum(axis=1)
    
    return pd.concat([positions_df, returns], axis=1)

def run_distance_strategy(prices_df: pd.DataFrame, max_pairs: int = 3, **kwargs) -> pd.DataFrame:
    """Run the distance strategy iteratively through the price series."""
    universe = list(prices_df.columns)
    strategy = DistancePairsTrader(universe, max_pairs=max_pairs, **kwargs)
    positions = []
    dates = []
    
    # Add 6-month warm-up period (approximately 126 trading days)
    warm_up_days = 126
    
    for idx in prices_df.index:
        current_prices = prices_df.loc[:idx]
        
        # During warm-up period, store zero positions
        if len(current_prices) < warm_up_days:
            pos = {ticker: 0 for ticker in universe}
        else:
            pos = strategy.step(current_prices, idx)
            
        positions.append(pos)
        dates.append(idx)
    
    # Create positions DataFrame
    positions_df = pd.DataFrame(positions, index=dates)
    
    # Calculate returns
    returns = pd.DataFrame(index=prices_df.index)
    for col in prices_df.columns:
        price_rets = prices_df[col].pct_change()
        returns[f'{col}_return'] = positions_df[col].shift(1) * price_rets
        
    returns['total_return'] = returns.sum(axis=1)
    
    return pd.concat([positions_df, returns], axis=1)

def run_backtest(strategy_type: str = 'dynamic', plot_results: bool = True, **kwargs) -> None:
    """
    Run backtest for either dynamic or distance strategy.
    
    Args:
        strategy_type: 'dynamic' for DynamicPairsTrader or 'distance' for DistancePairsTrader
        plot_results: Whether to plot the results
        **kwargs: Additional parameters to pass to the strategy
    """
    # Parameters
    universe = ['EWH', 'EWZ', 'EWT', 'EWY', 'EWA', 'EWM', 'EWS']
    start_date = '2004-01-01'
    end_date = '2015-12-31'
    
    # Download data
    prices = yf.download(universe, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    # Run appropriate strategy
    results = run_strategy(prices, **kwargs) if strategy_type == 'dynamic' else run_distance_strategy(prices, **kwargs)
    
    # Calculate strategy performance metrics
    cum_returns = (1 + results['total_return']).cumprod()
    sharpe = results['total_return'].mean() / results['total_return'].std() * np.sqrt(252)
    
    # Calculate benchmark returns (equal-weighted portfolio)
    benchmark_weights = pd.DataFrame(1.0/len(universe), index=prices.index, columns=universe)
    benchmark_returns = (prices.pct_change() * benchmark_weights).sum(axis=1)
    cum_benchmark = (1 + benchmark_returns).cumprod()
    benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)
    
    # Print performance metrics
    print(f"\nStrategy Performance ({strategy_type}):")
    print(f"Total Return: {cum_returns.iloc[-1]-1:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("\nEqual Weight Performance:")
    print(f"Total Return: {cum_benchmark.iloc[-1]-1:.2%}")
    print(f"Sharpe Ratio: {benchmark_sharpe:.2f}")
    
    if plot_results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot cumulative returns
        cum_returns.plot(ax=ax1, label='Strategy')
        cum_benchmark.plot(ax=ax1, label='Equal-Weight Portfolio', style='--')
        ax1.set_title('Cumulative Strategy Returns')
        ax1.grid(True)
        ax1.legend()
        
        # Plot number of active positions
        active_positions = (results[universe] != 0).sum(axis=1)
        active_positions.plot(ax=ax2)
        ax2.set_title('Number of Active Positions')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # Strategy parameters
    dynamic_params = {
        'max_pairs': 5,
        'lookback_period': 120,
        'threshold': 0.7,
        'warm_up_period': 250,
        'decay': 0.94,
        'rebalance_period': 126,
        'coint_threshold': 0.05
    }
    
    distance_params = {
        'max_pairs': 3,
        'lookback_period': 120,
        'threshold': 2.0,
        'warm_up_period': 250,
        'rebalance_period': 126
    }
    
    print("\nRunning Dynamic Strategy...")
    run_backtest('dynamic', **dynamic_params)
    
    print("\nRunning Distance Strategy...")
    run_backtest('distance', **distance_params)

if __name__ == "__main__":
    main() 