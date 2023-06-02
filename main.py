import sys
from data import BinanceDataHandler

from backtest import BackTester
from strategies import KNNStrategy


if __name__ == "__main__":
    args = sys.argv
    if args[1] == "fetch-data":
        '''
        Description: To Fetch and Dump Market Data.
        Command: `python main.py fetch-data <symbol> <interval> <start_date> <end_date>`
        Example: `python main.py fetch-data ETHUSDT 3m 2020/1/1-00:00 2023/6/1-00:00`
        '''
        handler = BinanceDataHandler()
        symbol, interval, from_dt, to_dt = args[2], args[3], args[4], args[5]
        print(f"Fetching data for {symbol} from {from_dt} to {to_dt} with timeframe {interval}")
        handler.dump_market_data(symbol, interval, from_dt, to_dt)
    elif args[1] == "backtest":
        from_dt, to_dt = args[2], args[3]
        strategy = KNNStrategy()
        data_handler = BinanceDataHandler()
        backtester = BackTester(strategy, data_handler, 1000, from_dt, to_dt)
        backtester.run()
    else:
        print("Invalid Argument!!!\nAccepts only `train` and `dump`")
