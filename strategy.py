from abc import ABCMeta, abstractmethod
from typing import Optional

import pandas as pd
import pandas_ta as ta


class BaseStrategy(metaclass=ABCMeta):

    # Minimum number of candles required for the strategy to generate a valid signal
    start_up_candle_count: int = 1
    # Timeframe for the strategy
    time_frame: str = "5m"
    # Asset to be traded by the strategy
    asset: Optional[str] = None
    # Stop loss for the strategy: Defaults to 1%
    stop_loss: float = 0.01
    # Take profit for the strategy: Defaults to 4%
    take_profit: float = 0.04
    # Amount of total trade-able balance to stake per trade
    stake_amount: int = 100
    # Allows the bot to execute short trades
    can_short: bool = False

    @abstractmethod
    def populate_indicators(self, data_frame: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def populate_entry_signal(self, data_frame: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def populate_exit_signal(self, data_frame: pd.DataFrame):
        raise NotImplementedError


class BaseAIStrategy(BaseStrategy):
    @abstractmethod
    def feature_engineering(self, data_frame: pd.DataFrame):
        raise NotImplementedError


class KNNStrategy(BaseAIStrategy):
    start_up_candle_count = 28
    time_frame = "3m"
    asset = "ETH/USDT"
    stop_loss = 0.0002
    take_profit = 0.0008
    stake_amount = 100
    # Extra Attributes for the strategy
    long_window = 28
    short_window = 14

    def feature_engineering(self, data_frame: pd.DataFrame):
        return data_frame

    def populate_indicators(self, data_frame: pd.DataFrame):
        data_frame["v_max_rolling_long"] = (
            data_frame["volume"].rolling(window=self.long_window).max()
        )
        data_frame["v_min_rolling_long"] = (
            data_frame["volume"].rolling(window=self.long_window).min()
        )
        data_frame["v_max_rolling_short"] = (
            data_frame["volume"].rolling(window=self.short_window).max()
        )
        data_frame["v_min_rolling_short"] = (
            data_frame["volume"].rolling(window=self.short_window).min()
        )
        data_frame["vs"] = (
            99
            * (data_frame["volume"] - data_frame["v_min_rolling_long"])
            / (data_frame["v_max_rolling_long"] - data_frame["v_min_rolling_long"])
        )
        data_frame["vf"] = (
            99
            * (data_frame["volume"] - data_frame["v_min_rolling_short"])
            / (data_frame["v_max_rolling_short"] - data_frame["v_min_rolling_short"])
        )
        data_frame["rs"] = ta.rsi(data_frame["close"], self.long_window)
        data_frame["rf"] = ta.rsi(data_frame["close"], self.short_window)
        data_frame["cs"] = ta.cci(
            close=data_frame["close"],
            length=self.long_window,
            high=data_frame["high"],
            low=data_frame["low"],
        )
        data_frame["cf"] = ta.cci(
            close=data_frame["close"],
            length=self.short_window,
            high=data_frame["high"],
            low=data_frame["low"],
        )
        data_frame["os"] = ta.roc(data_frame["close"], self.long_window)
        data_frame["of"] = ta.roc(data_frame["close"], self.short_window)
        return data_frame

    def populate_entry_signal(self, data_frame: pd.DataFrame):
        return data_frame

    def populate_exit_signal(self, data_frame: pd.DataFrame):
        return data_frame
