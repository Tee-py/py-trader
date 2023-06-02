from enum import Enum


class CommandType(Enum):
    DATA = "fetch-data"
    TRAIN = "train"
    BACKTEST = "backtest"


class StrategyType(Enum):
    KNN = "knn"
