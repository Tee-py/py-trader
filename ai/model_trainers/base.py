from strategies import BaseAIStrategy
from abc import ABCMeta, abstractmethod
import pandas as pd


class BaseStrategyModelTrainer(metaclass=ABCMeta):
    def __init__(self, strategy: BaseAIStrategy):
        self.strategy = strategy

    def load_data_frame(self) -> pd.DataFrame:
        asset = self.strategy.asset.replace("/", "").lower()
        interval = self.strategy.time_frame
        return pd.read_csv(f"data/{asset}-{interval}.csv")

    def populate_features(self, dataframe) -> pd.DataFrame:
        return self.strategy.populate_features(dataframe)

    @abstractmethod
    def train(self):
        raise NotImplementedError
