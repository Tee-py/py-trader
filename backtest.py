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
        self.last_price = None

    def _enter_long(self, row: pd.Series):
        if self.balance >= self.stake_amount:
            current_price = row["close"]
            amount_to_buy = self.stake_amount / current_price
            self.trades.append({"price": current_price, "amount": amount_to_buy, "exited": False})
            self.balance -= self.stake_amount
            self.total_trades += 1

    def _exit_long(self, row: pd.Series):
        for trade in self.trades:
            if trade["exited"]:
                continue
            position_value = trade["amount"] * row["close"]
            if position_value < self.stake_amount:
                self.losses.append(self.stake_amount - position_value)
            if position_value > self.stake_amount:
                self.wins.append(position_value - self.stake_amount)
            self.balance += position_value
            trade["exited"] = True

    def run(self):
        balance = self.starting_balance
        stake_amount = self.stake_amount
        stop_loss = self.stop_loss
        take_profit = self.take_profit
        trades = []
        losses = []
        wins = []
        last_price = None
        total_trades = 0
        # Loop through every row and execute trade
        for _, row in self.dataframe.iterrows():
            # Entry Signal
            if row["signal"] == "enter_long":
                self._enter_long(row)
            # Exit Signal
            if row["signal"] == "exit_long":
                self._exit_long(row)

            # Check for Stop Loss and Tp Hits
            for trade in trades:
                if trade["exited"]:
                    continue
                position_value = trade["amount"] * row["close"]
                if position_value < stake_amount:
                    # check if stop loss has been hit
                    loss = stake_amount - position_value
                    pct_loss = loss / stake_amount * 100
                    if pct_loss >= stop_loss:
                        losses.append(loss)
                        balance += position_value
                        trade["exited"] = True
                if position_value > stake_amount:
                    # check if tp has been hit
                    gain = position_value - stake_amount
                    pct_gain = gain / stake_amount * 100
                    if pct_gain >= take_profit:
                        wins.append(gain)
                        balance += position_value
                        trade["exited"] = True
            self.last_price = row["close"]

        # Liquidate Existing Trades and Update The Balance
        for trade in trades:
            if trade["exited"]:
                continue
            position_value = trade["amount"] * last_price
            if position_value < stake_amount:
                losses.append(stake_amount - position_value)
            elif position_value > stake_amount:
                wins.append(position_value - stake_amount)
            balance += position_value
        win_percentage = len(wins) / total_trades * 100
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
