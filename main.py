import datetime
import sys

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
from backtest import BackTester
from strategies import KNNStrategy

LONG_WINDOW = 28
SHORT_WINDOW = 14


def dump_ohlcv(symbol: str, tf: str):
    since = int((datetime.datetime.now() - datetime.timedelta(hours=50)).timestamp() * 1000)
    binance = ccxt.binance()
    result = binance.fetch_ohlcv(symbol, tf, since=since, limit=1000)
    pd.DataFrame(
        result,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    ).to_csv(f"data/eth-usdt-{tf}.csv")


def feature_engineering(df: pd.DataFrame):
    df["v_max_rolling_long"] = df["volume"].rolling(window=LONG_WINDOW).max()
    df["v_min_rolling_long"] = df["volume"].rolling(window=LONG_WINDOW).min()
    df["v_max_rolling_short"] = df["volume"].rolling(window=SHORT_WINDOW).max()
    df["v_min_rolling_short"] = df["volume"].rolling(window=SHORT_WINDOW).min()
    df["vs"] = 99 * (df["volume"] - df["v_min_rolling_long"]) / (df["v_max_rolling_long"] - df["v_min_rolling_long"])
    df["vf"] = 99 * (df["volume"] - df["v_min_rolling_short"]) / (df["v_max_rolling_short"] - df["v_min_rolling_short"])
    df["rs"] = ta.rsi(df["close"], LONG_WINDOW)
    df["rf"] = ta.rsi(df["close"], SHORT_WINDOW)
    df["cs"] = ta.cci(close=df["close"], length=LONG_WINDOW, high=df["high"], low=df["low"])
    df["cf"] = ta.cci(close=df["close"], length=SHORT_WINDOW, high=df["high"], low=df["low"])
    df["os"] = ta.roc(df["close"], LONG_WINDOW)
    df["of"] = ta.roc(df["close"], SHORT_WINDOW)
    df["feature_1"] = df[["vf", "rf", "cf", "of"]].mean(axis=1)
    df["feature_2"] = df[["vs", "rs", "cs", "os"]].mean(axis=1)

    final_df = df[["feature_1", "feature_2", "open", "close"]][28:].reset_index(drop=True)
    final_df["label"] = np.where(final_df["close"].shift(1) > final_df["close"], -1, 1)
    return final_df


def train():
    df = pd.read_csv("data/eth-usdt-3m.csv")
    training_df = feature_engineering(df)[-400:-200]
    # Split the data into training and test datasets
    x = training_df[['feature_1', 'feature_2']]
    y = training_df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train and Test the KNN classifier/Model
    knn = KNeighborsClassifier(n_neighbors=60)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)

    # # Train and Test the SVM classifier/Model
    # svm = SVC()
    # svm.fit(x_train, y_train)
    # y_pred_svm = svm.predict(x_test)

    # Print classification report for KNN
    print("KNN: \n", classification_report(y_test, y_pred_knn))

    # Print classification report for SVM
    # print("SVM: \n", classification_report(y_test, y_pred_svm))

    # Save Models
    joblib.dump(knn, 'knn_model.pkl')
    # joblib.dump(svm, 'svm_model.pkl')


if __name__ == "__main__":
    arg = sys.argv[-1]
    if arg == "train":
        train()
    elif arg == "dump":
        dump_ohlcv("ETH/USDT", "3m")
    elif arg == "backtest":
        strategy = KNNStrategy()
        backtester = BackTester(strategy, 1000)
        backtester.run()
    else:
        print("Invalid Argument!!!\nAccepts only `train` and `dump`")
