import sys
import time
from data import BinanceDataHandler
from utils import CommandType, get_command_type, get_strategy_type, get_model_trainer_class, get_strategy_class

from backtest import BackTester


if __name__ == "__main__":
    args = sys.argv
    command_type = get_command_type(args[1])
    start_time = time.time()
    if command_type == CommandType.DATA:
        '''
        Description: To Fetch and Dump Market Data.
        Command: `python main.py fetch-data <symbol> <interval> <start_date> <end_date>`
        Example: `python main.py fetch-data ETHUSDT 3m 2020/1/1-00:00 2023/6/1-00:00`
        '''
        handler = BinanceDataHandler()
        symbol, interval, from_dt, to_dt = args[2], args[3], args[4], args[5]
        print(f"Fetching data for {symbol} from {from_dt} to {to_dt} with timeframe {interval}")
        handler.dump_market_data(symbol, interval, from_dt, to_dt)
    elif command_type == CommandType.TRAIN:
        '''
        Description: Train AI Models and save in pickle file.
        Command: `python main.py train <strategy>`
        Example: `python main.py train knn`
        '''
        strategy_type = get_strategy_type(args[2])
        strategy_class = get_strategy_class(strategy_type)
        trainer_class = get_model_trainer_class(strategy_type)
        trainer = trainer_class(strategy_class())
        trainer.train()
    elif command_type == CommandType.BACKTEST:
        '''
        Description: Backtest Trading Strategy.
        Command: `python main.py backtest <strategy> <start_date> <end_date>`
        Example: `python main.py backtest knn 2023/1/1-00:00 2023/6/1-00:00`
        '''
        from_dt, to_dt = args[3], args[4]
        strategy_type = get_strategy_type(args[2])
        strategy_class = get_strategy_class(strategy_type)
        strategy = strategy_class()
        data_handler = BinanceDataHandler()
        backtester = BackTester(strategy, data_handler, 1000, from_dt, to_dt)
        backtester.run()
    print(f"Executed in {time.time() - start_time} seconds")
