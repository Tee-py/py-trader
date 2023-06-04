from constants import CommandType, StrategyType
from strategies import KNNStrategy, KNNEMARibbonStrategy
from ai.model_trainers import KNNStrategyModelTrainer, KNNEMARibbonModelTrainer


def get_command_type(arg: str) -> CommandType:
    if arg == "fetch-data":
        return CommandType.DATA
    elif arg == "train":
        return CommandType.TRAIN
    elif arg == "backtest":
        return CommandType.BACKTEST
    else:
        raise ValueError("Invalid Argument")


def get_strategy_type(arg: str) -> StrategyType:
    if arg == "knn":
        return StrategyType.KNN
    if arg == "knn-ema":
        return StrategyType.KNN_EMA
    else:
        raise ValueError("Invalid Strategy")


def get_strategy_class(strategy: StrategyType):
    if strategy == StrategyType.KNN:
        return KNNStrategy
    elif strategy == StrategyType.KNN_EMA:
        return KNNEMARibbonStrategy
    else:
        raise ValueError("Invalid Strategy Type")


def get_model_trainer_class(strategy: StrategyType):
    if strategy == StrategyType.KNN:
        return KNNStrategyModelTrainer
    if strategy == StrategyType.KNN_EMA:
        return KNNEMARibbonModelTrainer
    else:
        raise ValueError("Invalid Strategy")
