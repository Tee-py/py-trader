import datetime
import sys
from data import BinanceDataHandler

# import ccxt
# import pandas as pd
# import numpy as np
# import pandas_ta as ta
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
# import joblib
from backtest import BackTester
from strategies import KNNStrategy

LONG_WINDOW = 28
SHORT_WINDOW = 14



# def feature_engineering(df: pd.DataFrame):
#     df["v_max_rolling_long"] = df["volume"].rolling(window=LONG_WINDOW).max()
#     df["v_min_rolling_long"] = df["volume"].rolling(window=LONG_WINDOW).min()
#     df["v_max_rolling_short"] = df["volume"].rolling(window=SHORT_WINDOW).max()
#     df["v_min_rolling_short"] = df["volume"].rolling(window=SHORT_WINDOW).min()
#     df["vs"] = 99 * (df["volume"] - df["v_min_rolling_long"]) / (df["v_max_rolling_long"] - df["v_min_rolling_long"])
#     df["vf"] = 99 * (df["volume"] - df["v_min_rolling_short"]) / (df["v_max_rolling_short"] - df["v_min_rolling_short"])
#     df["rs"] = ta.rsi(df["close"], LONG_WINDOW)
#     df["rf"] = ta.rsi(df["close"], SHORT_WINDOW)
#     df["cs"] = ta.cci(close=df["close"], length=LONG_WINDOW, high=df["high"], low=df["low"])
#     df["cf"] = ta.cci(close=df["close"], length=SHORT_WINDOW, high=df["high"], low=df["low"])
#     df["os"] = ta.roc(df["close"], LONG_WINDOW)
#     df["of"] = ta.roc(df["close"], SHORT_WINDOW)
#     df["feature_1"] = df[["vf", "rf", "cf", "of"]].mean(axis=1)
#     df["feature_2"] = df[["vs", "rs", "cs", "os"]].mean(axis=1)
#
#     final_df = df[["feature_1", "feature_2", "open", "close"]][28:].reset_index(drop=True)
#     final_df["label"] = np.where(final_df["close"].shift(1) > final_df["close"], -1, 1)
#     return final_df
#
#
# def train():
#     df = pd.read_csv("data/eth-usdt-3m.csv")
#     training_df = feature_engineering(df)[-400:-200]
#     # Split the data into training and test datasets
#     x = training_df[['feature_1', 'feature_2']]
#     y = training_df['label']
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
#     # Train and Test the KNN classifier/Model
#     knn = KNeighborsClassifier(n_neighbors=60)
#     knn.fit(x_train, y_train)
#     y_pred_knn = knn.predict(x_test)
#
#     # # Train and Test the SVM classifier/Model
#     # svm = SVC()
#     # svm.fit(x_train, y_train)
#     # y_pred_svm = svm.predict(x_test)
#
#     # Print classification report for KNN
#     print("KNN: \n", classification_report(y_test, y_pred_knn))
#
#     # Print classification report for SVM
#     # print("SVM: \n", classification_report(y_test, y_pred_svm))
#
#     # Save Models
#     joblib.dump(knn, 'knn_model.pkl')
#     # joblib.dump(svm, 'svm_model.pkl')


if __name__ == "__main__":
    arg = sys.argv
    if arg[1] == "fetch-data":
        '''
        Description: To Fetch and Dump Market Data.
        Command: `python main.py fetch-data <symbol> <interval> <start_date> <end_date>`
        Example: `python main.py fetch-data ETHUSD 3m 2020/1/1-00:00 2023/6/1-00:00`
        '''
        handler = BinanceDataHandler()
        symbol, interval, from_dt, to_dt = arg[2], arg[3], arg[4], arg[5]
        handler.dump_market_data(symbol, interval, from_dt, to_dt)
    elif arg[1] == "backtest":
        strategy = KNNStrategy()
        backtester = BackTester(strategy, 1000)
        backtester.run()
    else:
        print("Invalid Argument!!!\nAccepts only `train` and `dump`")
