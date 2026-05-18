import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

# =========================
# STRATEGY PARAMETERS
# =========================
# Base timeframe = 15m

# HTF bias timeframe (resample)
HTF_TIMEFRAME = "1h"
HTF_EMA_FAST = 50
HTF_EMA_SLOW = 200

EMA_FAST = 50
EMA_SLOW = 200

EMA_DISTANCE_THRESHOLD = 0.0008
EMA_SLOPE_PERIOD = 5
EMA_SLOPE_THRESHOLD = 0.00005

BREAKOUT_PERIOD = 20
VOLUME_PERIOD = 20
ATR_PERIOD = 14

# Stop logic
ATR_MULTIPLIER = 1.8

# Retest logic
RETEST_WINDOW = 14
RETEST_TOL_ATR = 0.35
RISK_REWARD_RATIO = 3.0  # kept for compatibility

# Filters
VOLUME_THRESHOLD = 1.10
CANDLE_STRENGTH_RATIO = 0.20
ATR_BREAKOUT_MARGIN = 0.05

# Regime filters
ADX_PERIOD = 14
ADX_MIN = 16
ATR_PCT_SMA_PERIOD = 80


def _add_htf_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 1H bias using EMA50/EMA200 on 1H candles and map it back to 15m.
    """
    ohlc = df[['open', 'high', 'low', 'close', 'volume']].copy()

    htf = ohlc.resample(HTF_TIMEFRAME).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    htf['htf_ema50'] = EMAIndicator(close=htf['close'], window=HTF_EMA_FAST).ema_indicator()
    htf['htf_ema200'] = EMAIndicator(close=htf['close'], window=HTF_EMA_SLOW).ema_indicator()

    htf['htf_bull'] = (htf['htf_ema50'] > htf['htf_ema200']) & (htf['close'] > htf['htf_ema200'])
    htf['htf_bear'] = (htf['htf_ema50'] < htf['htf_ema200']) & (htf['close'] < htf['htf_ema200'])

    # Map back to base timeframe
    df['htf_bull'] = htf['htf_bull'].reindex(df.index, method='ffill').fillna(False)
    df['htf_bear'] = htf['htf_bear'].reindex(df.index, method='ffill').fillna(False)

    return df


def apply_strategy(df, validate=True):
    required_cols = ['open', 'high', 'low', 'close', 'volume']

    if validate:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(df) < 400:
            raise ValueError(f"Not enough data. Got {len(df)} rows.")
        if df[required_cols].isna().any().any():
            raise ValueError("OHLCV contains NaN values")

    df = df.copy()

    # =========================
    # HTF BIAS (1H)
    # =========================
    df = _add_htf_bias(df)

    # =========================
    # INDICATORS (15m)
    # =========================
    df['ema50'] = EMAIndicator(close=df['close'], window=EMA_FAST).ema_indicator()
    df['ema200'] = EMAIndicator(close=df['close'], window=EMA_SLOW).ema_indicator()

    df['ema50_slope'] = (
        (df['ema50'] - df['ema50'].shift(EMA_SLOPE_PERIOD))
        / df['ema50'].shift(EMA_SLOPE_PERIOD)
    )

    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_PERIOD)
    df['atr'] = atr.average_true_range()
    df['atr'] = df['atr'].clip(lower=df['close'] * 0.0001)

    # volatility regime
    df['atr_pct'] = df['atr'] / df['close']
    df['atr_pct_sma'] = df['atr_pct'].rolling(ATR_PCT_SMA_PERIOD).mean()
    df['vol_ok'] = df['atr_pct'] > df['atr_pct_sma']

    # ADX trend strength
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=ADX_PERIOD)
    df['adx'] = adx.adx()
    df['adx_ok'] = df['adx'] > ADX_MIN

    df['ema_distance'] = abs(df['ema50'] - df['ema200']) / df['close']

    # =========================
    # TREND FILTER (15m + HTF)
    # =========================
    df['strong_bullish'] = (
        df['htf_bull'] &
        (df['ema50'] > df['ema200']) &
        (df['ema_distance'] > EMA_DISTANCE_THRESHOLD) &
        (df['ema50_slope'] > EMA_SLOPE_THRESHOLD) &
        df['vol_ok'] &
        df['adx_ok']
    )

    df['strong_bearish'] = (
        df['htf_bear'] &
        (df['ema50'] < df['ema200']) &
        (df['ema_distance'] > EMA_DISTANCE_THRESHOLD) &
        (df['ema50_slope'] < -EMA_SLOPE_THRESHOLD) &
        df['vol_ok'] &
        df['adx_ok']
    )

    # =========================
    # LEVELS
    # =========================
    df['high_lvl'] = df['high'].rolling(BREAKOUT_PERIOD).max().shift(1)
    df['low_lvl'] = df['low'].rolling(BREAKOUT_PERIOD).min().shift(1)

    # =========================
    # VOLUME + MOMENTUM FILTERS
    # =========================
    df['volume_avg'] = df['volume'].rolling(VOLUME_PERIOD).mean()
    df['high_volume'] = df['volume'] > (df['volume_avg'] * VOLUME_THRESHOLD)

    df['candle_body'] = (df['close'] - df['open']).abs()
    df['strong_candle'] = df['candle_body'] > (df['atr'] * CANDLE_STRENGTH_RATIO)

    df['bull_candle'] = df['close'] > df['open']
    df['bear_candle'] = df['close'] < df['open']

    # Extra confirmation: momentum continuation
    df['bull_confirm'] = df['bull_candle'] & (df['close'] > df['close'].shift(1))
    df['bear_confirm'] = df['bear_candle'] & (df['close'] < df['close'].shift(1))

    # =========================
    # BREAKOUT EVENT (NO ENTRY YET)
    # =========================
    df['breakout_long'] = (
        df['strong_bullish'] &
        df['high_volume'] &
        df['strong_candle'] &
        df['bull_candle'] &
        (df['close'] > (df['high_lvl'] + df['atr'] * ATR_BREAKOUT_MARGIN))
    )

    df['breakout_short'] = (
        df['strong_bearish'] &
        df['high_volume'] &
        df['strong_candle'] &
        df['bear_candle'] &
        (df['close'] < (df['low_lvl'] - df['atr'] * ATR_BREAKOUT_MARGIN))
    )

    # =========================
    # RETEST ENTRY (breakout -> retest -> reclaim + confirmation)
    # =========================
    df['recent_breakout_long'] = (
        df['breakout_long'].shift(1)
        .rolling(RETEST_WINDOW)
        .max()
        .fillna(0)
        .astype(bool)
    )

    df['recent_breakout_short'] = (
        df['breakout_short'].shift(1)
        .rolling(RETEST_WINDOW)
        .max()
        .fillna(0)
        .astype(bool)
    )

    tol = df['atr'] * RETEST_TOL_ATR

    # Long: wick retests level, closes above, and confirm candle momentum
    df['buy'] = (
        df['recent_breakout_long'] &
        (df['low'] <= (df['high_lvl'] + tol)) &
        (df['close'] > df['high_lvl']) &
        df['bull_confirm']
    ).fillna(False).astype(bool)

    # Short: wick retests level, closes below, and confirm momentum
    df['sell'] = (
        df['recent_breakout_short'] &
        (df['high'] >= (df['low_lvl'] - tol)) &
        (df['close'] < df['low_lvl']) &
        df['bear_confirm']
    ).fillna(False).astype(bool)

    # =========================
    # STOPS
    # Put stops behind retest structure + ATR buffer
    # =========================
    df['buy_sl'] = pd.concat([
        (df['high_lvl'] - df['atr'] * 1.0),
        (df['close'] - df['atr'] * ATR_MULTIPLIER)
    ], axis=1).min(axis=1)

    df['sell_sl'] = pd.concat([
        (df['low_lvl'] + df['atr'] * 1.0),
        (df['close'] + df['atr'] * ATR_MULTIPLIER)
    ], axis=1).max(axis=1)

    df['buy_risk'] = (df['close'] - df['buy_sl']).clip(lower=0.0001)
    df['sell_risk'] = (df['sell_sl'] - df['close']).clip(lower=0.0001)

    # kept for compatibility
    df['buy_tp'] = df['close'] + (df['buy_risk'] * RISK_REWARD_RATIO)
    df['sell_tp'] = df['close'] - (df['sell_risk'] * RISK_REWARD_RATIO)

    # =========================
    # DEBUG
    # =========================
    print("\n===== STRATEGY DEBUG =====")
    print("HTF Bull:", int(df['htf_bull'].sum()))
    print("HTF Bear:", int(df['htf_bear'].sum()))
    print("Bullish:", int(df['strong_bullish'].sum()))
    print("Bearish:", int(df['strong_bearish'].sum()))
    print("Breakout Long events:", int(df['breakout_long'].sum()))
    print("Breakout Short events:", int(df['breakout_short'].sum()))
    print("Retest Long entries:", int(df['buy'].sum()))
    print("Retest Short entries:", int(df['sell'].sum()))

    return df