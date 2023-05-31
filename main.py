import datetime
import sys

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import joblib
from backtest import BackTester

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


def backtest():
    df = pd.read_csv("data/eth-usdt-3m.csv")
    backtest_df = feature_engineering(df)[-200:]
    knn_model = joblib.load('knn_model.pkl')
    predictions = knn_model.predict(backtest_df[["feature_1", "feature_2"]])
    backtest_df["predicted"] = predictions
    backtest_df.loc[backtest_df["predicted"] == 1, "signal"] = "enter_long"
    backtest_df.loc[backtest_df["predicted"] == -1, "signal"] = "exit_long"

    balance = 100
    stake_amount = 10
    stop_loss = 0.0002  # 2% of stake amount
    take_profit = 0.0008  # 12% of stake amount
    # long_signal = 1
    # short_signal = -1
    # trades = []  # can only go long for now
    # losses = []
    # wins = []
    # last_price = None
    # total_trades = 0
    # sell_trades = []
    # Loop through every row and execute trade
    # for index, row in backtest_df.iterrows():
    #     # Entry Signal
    #     if row["predicted"] == long_signal:
    #         if balance >= stake_amount:
    #             current_price = row["close"]
    #             amount_to_buy = stake_amount/current_price
    #             trades.append({"price": current_price, "amount": amount_to_buy, "exited": False})
    #             balance -= stake_amount
    #             total_trades += 1
    #     # Exit Signal
    #     if row["predicted"] == short_signal:
    #         for trade in trades:
    #             if trade["exited"]:
    #                 # print("Already exited trade")
    #                 continue
    #             position_value = trade["amount"] * row["close"]
    #             if position_value < stake_amount:
    #                 losses.append(stake_amount - position_value)
    #                 # print(f"Loss Encountered: {stake_amount - position_value}")
    #             else:
    #                 wins.append(position_value - stake_amount)
    #                 # print(f"Gain Encountered: {position_value - stake_amount}")
    #             balance += position_value
    #         trades = []  # Liquidate all buy positions and reset the trade array
    #
    #     # Check for Stop Loss and Tp Hits
    #     for trade in trades:
    #         if trade["exited"]:
    #             continue
    #         position_value = trade["amount"] * row["close"]
    #         if position_value < stake_amount:
    #             # check if stop loss has been hit
    #             loss = stake_amount - position_value
    #             pct_loss = loss/stake_amount
    #             if pct_loss >= stop_loss:
    #                 losses.append(loss)
    #                 balance += position_value
    #                 trade["exited"] = True
    #         if position_value > stake_amount:
    #             # check if tp has been hit
    #             gain = position_value - stake_amount
    #             pct_gain = gain/stake_amount
    #             if pct_gain >= take_profit:
    #                 print(f"Exiting Trade Due to TP Hit: {pct_gain}")
    #                 wins.append(gain)
    #                 balance += position_value
    #                 trade["exited"] = True
    #     last_price = row["close"]
    #
    # # Liquidate Existing Trades and Update The Balance
    # for trade in trades:
    #     if trade["exited"]:
    #         continue
    #     position_value = trade["amount"] * last_price
    #     if position_value < stake_amount:
    #         losses.append(stake_amount - position_value)
    #     elif position_value > stake_amount:
    #         wins.append(position_value - stake_amount)
    #     balance += position_value
    # win_percentage = len(wins)/total_trades * 100
    # loss_percentage = len(losses)/total_trades * 100
    # win_loss_ratio = len(wins)/len(losses)
    # print(f"Total Trades: {total_trades}\tWin Count: {len(wins)}\tLoss Count: {len(losses)}"
    #       f"\nTotal Win Amount: {sum(wins)}\tTotal Loss Amount: {sum(losses)}")
    # print(f"Final Balance: {balance}")
    # print(f"Win %: {win_percentage}\nLoss %: {loss_percentage}")
    # print(f"W/L ratio: {win_loss_ratio}")

    backtester = BackTester(backtest_df, balance, stake_amount, stop_loss, take_profit)
    metrics = backtester.run()
    print(metrics)


if __name__ == "__main__":
    arg = sys.argv[-1]
    if arg == "train":
        train()
    elif arg == "dump":
        dump_ohlcv("ETH/USDT", "3m")
    elif arg == "backtest":
        backtest()
    else:
        print("Invalid Argument!!!\nAccepts only `train` and `dump`")
