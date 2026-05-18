import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange


# =========================================================
# PARAMETERS (TUNABLE)
# =========================================================
HTF_TIMEFRAME = "1h"
HTF_EMA_FAST = 50
HTF_EMA_SLOW = 200

ATR_PERIOD = 14
ADX_PERIOD = 14

# Regime filters
ADX_TREND_MIN = 18          # higher = fewer trades, cleaner trends
VOL_FILTER_LOOKBACK = 80    # volatility regime baseline
VOL_MIN_MULT = 0.90         # allow trades only if atr_pct >= 0.9 * baseline

# Momentum windows (intraday)
MOM_FAST = 8
MOM_SLOW = 32
MOM_THRESHOLD = 0.0008      # 0.08% momentum threshold

# Risk & Targets
STOP_ATR_MULT = 2.0
MIN_RR = 2.0                # minimum RR (hard)
TRAIL_START_R = 1.5          # start trailing after +1.5R
TRAIL_ATR_MULT = 1.2         # trailing distance in ATR


def _add_htf_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 1H bias using EMA trend and map back to base timeframe.
    """
    ohlc = df[['open', 'high', 'low', 'close', 'volume']].copy()

    htf = ohlc.resample(HTF_TIMEFRAME).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    htf['ema_fast'] = EMAIndicator(close=htf['close'], window=HTF_EMA_FAST).ema_indicator()
    htf['ema_slow'] = EMAIndicator(close=htf['close'], window=HTF_EMA_SLOW).ema_indicator()

    htf['htf_bull'] = (htf['ema_fast'] > htf['ema_slow']) & (htf['close'] > htf['ema_slow'])
    htf['htf_bear'] = (htf['ema_fast'] < htf['ema_slow']) & (htf['close'] < htf['ema_slow'])

    df['htf_bull'] = htf['htf_bull'].reindex(df.index, method='ffill').fillna(False)
    df['htf_bear'] = htf['htf_bear'].reindex(df.index, method='ffill').fillna(False)

    return df


def apply_strategy(df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    required_cols = ['open', 'high', 'low', 'close', 'volume']

    if validate:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(df) < 600:
            raise ValueError(f"Not enough data. Got {len(df)} rows. Need >= 600")
        if df[required_cols].isna().any().any():
            raise ValueError("OHLCV contains NaN values")

    df = df.copy()

    # =========================
    # HTF BIAS
    # =========================
    df = _add_htf_bias(df)

    # =========================
    # ATR + VOL REGIME
    # =========================
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_PERIOD)
    df['atr'] = atr.average_true_range().clip(lower=df['close'] * 0.0001)

    df['atr_pct'] = df['atr'] / df['close']
    df['atr_pct_base'] = df['atr_pct'].rolling(VOL_FILTER_LOOKBACK).mean()
    df['vol_ok'] = df['atr_pct'] >= (df['atr_pct_base'] * VOL_MIN_MULT)

    # =========================
    # ADX TREND FILTER
    # =========================
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=ADX_PERIOD)
    df['adx'] = adx.adx()
    df['trend_ok'] = df['adx'] >= ADX_TREND_MIN

    # =========================
    # MOMENTUM (TSMOM STYLE)
    # =========================
    df['ret_fast'] = df['close'].pct_change(MOM_FAST)
    df['ret_slow'] = df['close'].pct_change(MOM_SLOW)

    df['mom_up'] = (df['ret_fast'] > MOM_THRESHOLD) & (df['ret_slow'] > 0)
    df['mom_dn'] = (df['ret_fast'] < -MOM_THRESHOLD) & (df['ret_slow'] < 0)

    # =========================
    # SIGNALS (BUY/SELL)
    # =========================
    df['buy'] = (
        df['htf_bull'] &
        df['trend_ok'] &
        df['vol_ok'] &
        df['mom_up']
    ).fillna(False).astype(bool)

    df['sell'] = (
        df['htf_bear'] &
        df['trend_ok'] &
        df['vol_ok'] &
        df['mom_dn']
    ).fillna(False).astype(bool)

    # =========================
    # STOPS + TARGETS (MIN RR)
    # =========================
    df['buy_sl'] = df['close'] - (df['atr'] * STOP_ATR_MULT)
    df['sell_sl'] = df['close'] + (df['atr'] * STOP_ATR_MULT)

    df['buy_risk'] = (df['close'] - df['buy_sl']).clip(lower=0.0001)
    df['sell_risk'] = (df['sell_sl'] - df['close']).clip(lower=0.0001)

    df['buy_tp'] = df['close'] + (df['buy_risk'] * MIN_RR)
    df['sell_tp'] = df['close'] - (df['sell_risk'] * MIN_RR)

    # These are used by backtester for trailing
    df['trail_start_r'] = TRAIL_START_R
    df['trail_atr_mult'] = TRAIL_ATR_MULT

    # =========================
    # DEBUG
    # =========================
    print("\n===== STRATEGY DEBUG (TSMOM) =====")
    print("HTF Bull:", int(df['htf_bull'].sum()))
    print("HTF Bear:", int(df['htf_bear'].sum()))
    print("Trend OK:", int(df['trend_ok'].sum()))
    print("Vol OK:", int(df['vol_ok'].sum()))
    print("Mom Up:", int(df['mom_up'].sum()))
    print("Mom Down:", int(df['mom_dn'].sum()))
    print("Buy Signals:", int(df['buy'].sum()))
    print("Sell Signals:", int(df['sell'].sum()))

    return df