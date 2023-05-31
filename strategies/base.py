from abc import ABCMeta, abstractmethod
from typing import Optional

import joblib
import pandas as pd


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
    # For AI Strategies
    ai_enabled: bool = False

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
    ai_enabled = True
    model_file: Optional[str] = None

    @abstractmethod
    def populate_features(self, data_frame: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def populate_predictions(self, data_frame: pd.DataFrame):
        raise NotImplementedError

    @property
    def model(self):
        return joblib.load(self.model_file)
