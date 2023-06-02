import datetime
import json
import time
from typing import Optional

import aiohttp
import asyncio
import pandas as pd


class BinanceDataHandler:
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"

    async def fetch_klines_data(
        self,
        symbol: str,
        interval: str = "1d",
        limit=1000,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        async with aiohttp.ClientSession() as session:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            if start_time:
                params["startTime"] = start_time * 1000
            if end_time:
                params["endTime"] = end_time * 1000
            async with session.get(
                f"{self.base_url}/klines", params=params
            ) as response:
                if response.status == 200:
                    content = await response.content.read()
                    klines = json.loads(content)
                    klines_data = [
                        {
                            "timestamp": int(kline[0] / 1000),
                            "open": float(kline[1]),
                            "high": float(kline[2]),
                            "low": float(kline[3]),
                            "close": float(kline[4]),
                            "volume": float(kline[5]),
                        }
                        for kline in klines
                    ]
                    return klines_data
                else:
                    content = await response.content.read()
                    print(content)

    async def dump_ohlcv(
        self, symbol: str, interval: str, from_dt: str, to_dt: Optional[str] = None
    ):
        start_timestamp = int(
            datetime.datetime.strptime(from_dt, "%Y/%m/%d-%H:%M").timestamp()
        )
        end_timestamp = (
            int(datetime.datetime.strptime(to_dt, "%Y/%m/%d-%H:%M").timestamp())
            if to_dt
            else int(time.time())
        )
        latest_timestamp, klines_data = start_timestamp, []

        while latest_timestamp < end_timestamp:
            print(f"Fetching From: {latest_timestamp} to {end_timestamp}")
            response = await self.fetch_klines_data(
                symbol, interval, 1000, latest_timestamp
            )
            t_klines_data = [
                kline for kline in response if kline["timestamp"] <= end_timestamp
            ]
            klines_data.extend(t_klines_data)
            latest_timestamp = klines_data[-1]["timestamp"] + (
                klines_data[-1]["timestamp"] - klines_data[-2]["timestamp"]
            )
        df = pd.DataFrame(klines_data)
        df.set_index("timestamp")
        df.to_csv(f"data/{symbol.lower()}-{interval}.csv")

    def dump_market_data(self, symbol: str, interval: str, from_dt: str, to_dt: Optional[str] = None):
        asyncio.run(self.dump_ohlcv(symbol, interval, from_dt, to_dt))

    def load_market_data(self, symbol: str, interval: str, start: int, end: int):  # noqa
        df = pd.read_csv(f"data/{symbol.lower()}-{interval}.csv")
        return df.query('timestamp >= @start and timestamp <= @end')


if __name__ == "__main__":
    handler = BinanceDataHandler()
    handler.dump_market_data("ETHUSDT", "30m", "2022/1/1-00:00", "2023/6/2-00:00")
