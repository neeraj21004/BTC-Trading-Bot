import logging
import time
from pathlib import Path
from typing import Optional, List

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

BINANCE_MAX_PER_REQUEST = 1000


def _create_binance_exchange():
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True}
    })
    exchange.load_markets()
    return exchange


def _fetch_ohlcv_with_retries(
    exchange,
    symbol,
    timeframe,
    since,
    limit,
    max_retries,
    retry_delay
) -> List[list]:
    for attempt in range(1, max_retries + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeError) as error:
            if attempt == max_retries:
                raise
            logger.warning(
                f"Fetch attempt {attempt} failed for {symbol} {timeframe}: {error}. Retrying..."
            )
            time.sleep(retry_delay * attempt)
    return []


def fetch_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '15m',
    since: Optional[int] = None,
    limit: int = 1000,
    max_bars: Optional[int] = None,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    save_path: str = 'data/btc_data.csv',
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance and return a pandas DataFrame.

    FIXED BEHAVIOR:
    - Binance returns max 1000 candles per request.
    - If max_bars is provided, this function will fetch up to max_bars candles.
    - If since is None and max_bars > 1000, it paginates backwards to get older candles.

    Args:
        symbol: Market symbol, e.g. 'BTC/USDT'.
        timeframe: Candlestick timeframe, e.g. '15m'.
        since: Timestamp (ms) to start from. If None, fetches most recent candles and paginates backwards.
        limit: Bars per request (capped to 1000).
        max_bars: Total candles to fetch across requests.
        max_retries: Retries per request.
        retry_delay: Base delay between retries.
        save_path: CSV export path.

    Returns:
        pandas.DataFrame indexed by datetime with columns:
        ['open','high','low','close','volume']
    """
    exchange = _create_binance_exchange()

    # Binance max per fetch
    per_request_limit = min(int(limit), BINANCE_MAX_PER_REQUEST)

    # If user didn't specify max_bars, we just fetch 1 batch (up to per_request_limit)
    target_total = int(max_bars) if max_bars is not None else per_request_limit
    if target_total <= 0:
        raise ValueError(f"max_bars must be > 0, got {target_total}")

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    bars: List[list] = []

    # =========================================================
    # MODE A: since provided -> forward pagination
    # =========================================================
    if since is not None:
        current_since = since
        remaining = target_total

        while remaining > 0:
            fetch_limit = min(per_request_limit, remaining)

            batch = _fetch_ohlcv_with_retries(
                exchange,
                symbol,
                timeframe,
                current_since,
                fetch_limit,
                max_retries,
                retry_delay,
            )

            if not batch:
                break

            # remove duplicate boundary candle
            if bars and batch[0][0] == bars[-1][0]:
                batch = batch[1:]

            if not batch:
                break

            bars.extend(batch)
            remaining -= len(batch)

            # stop if exchange returns less
            if len(batch) < fetch_limit:
                break

            current_since = batch[-1][0] + timeframe_ms
            if current_since >= exchange.milliseconds():
                break

    # =========================================================
    # MODE B: since None -> backward pagination (older history)
    # =========================================================
    else:
        # 1) get most recent batch
        latest = _fetch_ohlcv_with_retries(
            exchange,
            symbol,
            timeframe,
            None,
            per_request_limit,
            max_retries,
            retry_delay,
        )

        if not latest:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        bars = latest[:]

        # If we already have enough, trim and finish
        if len(bars) >= target_total:
            bars = bars[-target_total:]
        else:
            # paginate backwards until we have target_total candles
            while len(bars) < target_total:
                oldest_ts = bars[0][0]
                # move back by one full batch window
                older_since = oldest_ts - (per_request_limit * timeframe_ms)

                older = _fetch_ohlcv_with_retries(
                    exchange,
                    symbol,
                    timeframe,
                    older_since,
                    per_request_limit,
                    max_retries,
                    retry_delay,
                )

                if not older:
                    break

                # keep only candles strictly older than current oldest to avoid overlap duplicates
                older = [b for b in older if b[0] < oldest_ts]

                if not older:
                    break

                bars = older + bars

                # stop if exchange returns less than max (no more history)
                if len(older) < per_request_limit:
                    break

            # trim to exact size (most recent target_total candles)
            bars = bars[-target_total:]

    # =========================================================
    # BUILD DATAFRAME
    # =========================================================
    df = pd.DataFrame(
        bars,
        columns=['time', 'open', 'high', 'low', 'close', 'volume'],
    )

    if df.empty:
        return df

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.drop_duplicates(subset=['time']).sort_values('time')
    df.set_index('time', inplace=True)

    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].astype(float)

    # save
    data_path = Path(save_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path)
    logger.info(f"Data saved to {data_path.resolve()}")
    logger.info(f"Fetched candles: {len(df)}")
    logger.info(f"Date range: {df.index[0]} -> {df.index[-1]}")

    return df