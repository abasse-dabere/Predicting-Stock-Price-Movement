import yfinance as yf
import pandas as pd

from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import ChaikinMoneyFlowIndicator

class StockDataProcessor:
    def __init__(self, start_date, end_date):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

    def download_data(self, stock):
        start_date = pd.to_datetime(self.start_date) - pd.DateOffset(days=365)
        end_date = pd.to_datetime(self.end_date) + pd.DateOffset(days=1) # exclusive
        data = yf.download(tickers=stock, interval="1d", start=start_date, end=end_date)
        return data
    
    def compute_trend_indicators(self, data):
        # normalized SMA
        for window in [10, 15, 20, 50, 100, 200]:
            data[f'SMA_{window}'] = SMAIndicator(close=data['Close'], window=window).sma_indicator() / data['Close']
        
        # normalized EMA
        for window in [10, 12, 14, 26, 30, 50, 100]:
            data[f'EMA_{window}'] = EMAIndicator(close=data['Close'], window=window).ema_indicator() / data['Close']

        # ADX
        for window in [14, 20, 25, 30]:
            adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=window)
            data[f'ADX_{window}'] = adx.adx()
            data[f'ADX_{window}_neg'] = adx.adx_neg()
            data[f'ADX_{window}_pos'] = adx.adx_pos()

        return data
    
    def compute_volatility_indicators(self, data):
        # normalized ATR
        for window in [14, 20, 28]:
            atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=window)
            data[f'ATR_{window}'] = atr.average_true_range() / data['Close']

        return data
    
    def compute_momentum_indicators(self, data):
        # RSI
        for window in [7, 14, 21]:
            data[f'RSI_{window}'] = RSIIndicator(close=data['Close'], window=window).rsi()

        # Stochastic Oscillator
        for window in [14, 21, 28]:
            stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=window, smooth_window=3)
            data[f'Stoch_{window}'] = stoch.stoch()
            data[f'Stoch_{window}_signal'] = stoch.stoch_signal()

        return data
    
    def compute_volume_indicators(self, data):
        # Chaikin Money Flow
        for window in [14, 20, 28]:
            cmf = ChaikinMoneyFlowIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=window)
            data[f'CMF_{window}'] = cmf.chaikin_money_flow()

        # Volume Rate of Change (VROC)
        for window in [7, 14, 21, 28]:
            data[f'VROC_{window}'] = data['Volume'].pct_change(periods=window)

        return data

    def compute_indicators(self, data):
        data = self.compute_trend_indicators(data)
        data = self.compute_volatility_indicators(data)
        data = self.compute_momentum_indicators(data)
        data = self.compute_volume_indicators(data)
        return data
    
    def add_target_and_clean_data(self, data):
        # add target
        data['target'] = data['Close'].pct_change().shift(-1) * 100
        data = data.drop(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        data = data.loc[data.index >= self.start_date]
        data = data.reset_index()
        return data

    def get_stock_data(self, stock):
        data = self.download_data(stock)
        data = self.compute_indicators(data)
        data = self.add_target_and_clean_data(data)
        return data