import pandas as pd
from typing import Optional


class BackTester:
    def __init__(self, df: pd.DataFrame,
                 starting_balance: int, stake_amount: int,
                 stop_loss: Optional[int], take_profit: Optional[int]):
        self.dataframe = df
        self.balance = starting_balance
        self.stake_amount = stake_amount
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trades = []
        self.wins = []
        self.losses = []
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

    def _exit_trade(self, current_row: pd.Series, trade: dict, is_sl_tp: bool = False):
        """
        Exits Single Trade
        :param current_row: Pandas row where the exit signal was generated. Must contain `close` column.
        :param trade: The trade to be exited. Contains `amount` bought and the buying `price`.
        :param is_sl_tp: Tells the backtester to only exit trades where stop_loss or take_profit has been hit.
        :return: None
        """
        if trade["exited"]:
            return
        position_value = trade["amount"] * current_row["close"]
        if position_value < self.stake_amount:
            if not is_sl_tp:
                self.losses.append(self.stake_amount - position_value)
            else:
                loss = self.stake_amount - position_value
                pct_loss = loss / self.stake_amount
                if pct_loss >= self.stop_loss:
                    self.losses.append(self.stake_amount - position_value)
                else:
                    return
        if position_value > self.stake_amount:
            if not is_sl_tp:
                self.wins.append(position_value - self.stake_amount)
            else:
                gain = self.stake_amount - position_value
                pct_gain = gain / self.stake_amount
                if pct_gain >= self.take_profit:
                    self.wins.append(position_value - self.stake_amount)
                else:
                    return
        self.balance += position_value
        trade["exited"] = True

    def _exit_trades(self, current_row: pd.Series, is_sl_tp: bool = False):
        """
        Exits All trades
        :param current_row: Pandas row where the exit signal was generated. Must contain `close` column.
        :param is_sl_tp: Tells the backtester to only exit trades where stop_loss or take_profit has been hit.
        :return:
        """
        for trade in self.trades:
            self._exit_trade(current_row, trade, is_sl_tp)

    def run(self):
        # Loop through every row and execute trade
        for _, row in self.dataframe.iterrows():
            # Entry Signal
            if row["signal"] == "enter_long":
                self._enter_long(row)
            # Exit Signal
            if row["signal"] == "exit_long":
                self._exit_trades(row)

            # Check for Stop Loss and Tp Hits
            self._exit_trades(row, True)
            self.last_row = row["close"]

        # Liquidate Existing Trades and Update The Balance
        self._exit_trades(self.last_row)
        win_percentage = len(self.wins) / total_trades * 100
        loss_percentage = len(losses) / total_trades * 100
        win_loss_ratio = len(wins) / len(losses)
        metrics = {
            "wins": len(wins),
            "losses": len(losses),
            "win_amount": sum(wins),
            "loss_amount": sum(losses),
            "win_%": win_percentage,
            "loss_%": loss_percentage,
            "w/l_ratio": win_loss_ratio,
            "total_trades": total_trades,
            "final_balance": balance
        }
        return metrics
