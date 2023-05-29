import pandas as pd
from typing import Optional, Dict
from tabulate import tabulate


class BackTester:
    def __init__(
        self, df: pd.DataFrame,
        starting_balance: int, stake_amount: int,
        stop_loss: Optional[float], take_profit: Optional[float]
    ):
        self.dataframe = df
        self.balance = starting_balance
        self.stake_amount = stake_amount
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = []
        self.last_action = None
        self.wins = []
        self.losses = []
        self.sl_hits = 0
        self.tp_hits = 0
        self.force_exits = 0
        self.total_trades = 0
        self.last_row = None

    def _enter_long(self, row: pd.Series):
        """
        Enters Long Trade
        :param row: Pandas row where the Entry Signal was generated. Must contain `close` column
        :return: None
        """
        if self.balance >= self.stake_amount:
            current_price = row["close"]
            amount_to_buy = self.stake_amount / current_price
            self.trades.append({"price": current_price, "amount": amount_to_buy, "exited": False})
            self.balance -= self.stake_amount
            self.total_trades += 1
            self.last_action = "enter_long"

    def _exit_trade(self, current_row: pd.Series, trade: Dict, is_sl_tp: bool = False, force_exit: bool = False):
        """
        Exits Single Trade
        :param current_row: Pandas row where the exit signal was generated. Must contain `close` column.
        :param trade: The trade to be exited. Contains `amount` bought and the buying `price`.
        :param is_sl_tp: Tells the backtester to only exit trades where stop_loss or take_profit has been hit.
        :param force_exit: Force exits the trade when set to True.
        :return: None
        """
        if trade["exited"]:
            return
        position_value = trade["amount"] * current_row["close"]
        # Loosing position
        if position_value < self.stake_amount:
            if not is_sl_tp:
                if force_exit:
                    self.force_exits += 1
                    self.losses.append(self.stake_amount - position_value)
                else:
                    return
            else:
                loss = self.stake_amount - position_value
                pct_loss = loss / self.stake_amount
                if pct_loss >= self.stop_loss:
                    self.sl_hits += 1
                    self.losses.append(self.stake_amount - position_value)
                else:
                    return
        # Gaining Position
        elif position_value > self.stake_amount:
            if not is_sl_tp:
                if force_exit:
                    self.force_exits += 1
                    self.wins.append(position_value - self.stake_amount)
                else:
                    return
            else:
                gain = position_value - self.stake_amount
                pct_gain = gain / self.stake_amount
                if pct_gain >= self.take_profit:
                    self.tp_hits += 1
                    self.wins.append(position_value - self.stake_amount)
                else:
                    return
        # No change
        else:
            # Exit the function if the trade is not to be exited forcefully
            if not force_exit:
                return
            else:
                self.force_exits += 1
        self.balance += position_value
        trade["exited"] = True
        self.last_action = "exit_trade"

    def _exit_trades(self, current_row: pd.Series, is_sl_tp: bool = False, force_exit: bool = False):
        """
        Exits All trades
        :param current_row: Pandas row where the exit signal was generated. Must contain `close` column.
        :param is_sl_tp: Tells the backtester to only exit trades where stop_loss or take_profit has been hit.
        :param force_exit: Force exits trades when set to True.
        :return:
        """
        for trade in self.trades:
            self._exit_trade(current_row, trade, is_sl_tp, force_exit)

    def _calculate_metrics(self) -> Dict:
        no_of_wins = len(self.wins)
        no_of_loss = len(self.losses)
        win_total = sum(self.wins)
        loss_total = sum(self.losses)
        win_percentage = no_of_wins / self.total_trades * 100
        loss_percentage = no_of_loss / self.total_trades * 100
        win_loss_ratio = no_of_wins / no_of_loss
        metrics = {
            "wins": [no_of_wins],
            "losses": [no_of_loss],
            "win_amount": [win_total],
            "loss_amount": [loss_total],
            "sl_hits": [self.sl_hits],
            "tp_hits": [self.tp_hits],
            "force_exits": [self.force_exits],
            "win_%": [win_percentage],
            "loss_%": [loss_percentage],
            "w/l_ratio": [win_loss_ratio],
            "total_trades": [self.total_trades],
            "final_balance": [self.balance]
        }
        return metrics

    def run(self):
        # Loop through every row and execute trade
        for _, row in self.dataframe.iterrows():
            # Entry Signal
            if row["signal"] == "enter_long":
                if self.last_action != "enter_long":
                    self._enter_long(row)
            # Exit Signal
            if row["signal"] == "exit_long":
                self._exit_trades(row, force_exit=True)
            # Check for Stop Loss and Tp Hits
            self._exit_trades(row, True)
            self.last_row = row

        # Liquidate Existing Trades and Update The Balance
        self._exit_trades(self.last_row, force_exit=True)
        metrics = self._calculate_metrics()
        print("\t\t\t\t\t\t------------------ BACKTEST RESULTS ----------------")
        print(tabulate(metrics, headers="keys", tablefmt="fancy_grid"))
