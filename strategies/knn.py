import numpy as np
import pandas as pd
import pandas_ta as ta

from .base import BaseAIStrategy


class KNNStrategy(BaseAIStrategy):
    start_up_candle_count = 28
    model_file = "knn_model.pkl"
    time_frame = "3m"
    asset = "ETH/USDT"
    stop_loss = 0.0002
    take_profit = 0.0008
    stake_amount = 100
    # Extra Attributes for the strategy
    long_window = 28
    short_window = 14

    def populate_features(self, data_frame: pd.DataFrame):
        data_frame["feature_1"] = data_frame[["vf", "rf", "cf", "of"]].mean(axis=1)
        data_frame["feature_2"] = data_frame[["vs", "rs", "cs", "os"]].mean(axis=1)

        final_df = data_frame[["feature_1", "feature_2", "open", "close"]][
            self.start_up_candle_count :
        ].reset_index(drop=True)
        final_df["label"] = np.where(
            final_df["close"].shift(1) > final_df["close"], -1, 1
        )
        return final_df

    def populate_predictions(self, data_frame: pd.DataFrame):
        predictions = self.model.predict(data_frame[["feature_1", "feature_2"]])
        data_frame["predicted"] = predictions
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
        data_frame.loc[data_frame["predicted"] == 1, "signal"] = "enter_long"
        return data_frame

    def populate_exit_signal(self, data_frame: pd.DataFrame):
        data_frame.loc[data_frame["predicted"] == -1, "signal"] = "exit_long"
        return data_frame
