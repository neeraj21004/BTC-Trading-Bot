import pandas as pd
import numpy as np
from typing import Dict, Tuple

# =========================================================
# CONFIG
# =========================================================
SLIPPAGE_BPS = 2
TRANSACTION_COST_BPS = 1
MAX_HOLDING_PERIODS = 80

# Trade management tuned for BTC intraday trend bursts
BREAKEVEN_R = 0.8          # earlier BE
TRAIL_START_R = 1.2        # earlier trailing start so winners don't die in TIMEOUT
TRAIL_ATR_MULT = 2.8       # wide enough to avoid noise stop-out

USE_TP = False  # keep False to allow big moves


def _fee_notional(notional: float) -> float:
    return notional * (TRANSACTION_COST_BPS / 10000)


def run_backtest(
    df: pd.DataFrame,
    initial_balance: float = 10000,
    risk_per_trade: float = 0.01,
    slippage_bps: int = SLIPPAGE_BPS,
    max_periods: int = MAX_HOLDING_PERIODS,
) -> Tuple[pd.DataFrame, Dict]:

    required_cols = [
        'buy', 'sell',
        'open', 'high', 'low', 'close',
        'buy_sl', 'buy_tp', 'sell_sl', 'sell_tp',
        'atr'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    balance = float(initial_balance)
    balance_history = [balance]
    trade_logs = []

    wins = 0
    losses = 0

    i = 0
    while i < len(df) - 2:
        row = df.iloc[i]
        entry_idx = i + 1

        # =========================
        # LONG
        # =========================
        if row['buy']:
            entry_open = float(df.iloc[entry_idx]['open'])
            entry = entry_open * (1 + slippage_bps / 10000)

            initial_sl = float(row['buy_sl'])
            tp = float(row['buy_tp'])

            risk_per_unit = entry - initial_sl
            if risk_per_unit <= 0 or np.isnan(risk_per_unit):
                i += 1
                continue

            capital_risk = balance * float(risk_per_trade)  # ✅ FIX: use parameter
            qty = capital_risk / risk_per_unit

            balance -= _fee_notional(qty * entry)

            sl = initial_sl
            highest = entry
            trade_closed = False

            future = df.iloc[entry_idx: entry_idx + max_periods]
            for bars_held, (t, candle) in enumerate(future.iterrows(), start=1):
                high = float(candle['high'])
                low = float(candle['low'])

                highest = max(highest, high)

                # Breakeven earlier
                if highest >= entry + (risk_per_unit * BREAKEVEN_R):
                    sl = max(sl, entry)

                # Trail earlier
                if highest >= entry + (risk_per_unit * TRAIL_START_R):
                    trail = highest - (float(row['atr']) * TRAIL_ATR_MULT)
                    sl = max(sl, trail)

                # Stop
                if low <= sl:
                    exit_price = sl * (1 - slippage_bps / 10000)

                    pnl = (exit_price - entry) * qty
                    balance += pnl
                    balance -= _fee_notional(qty * exit_price)

                    reason = "SL" if sl == initial_sl else ("BE" if sl == entry else "TRAIL_SL")

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    trade_logs.append({
                        'type': 'BUY',
                        'entry_time': df.index[entry_idx],
                        'exit_time': t,
                        'entry': round(entry, 2),
                        'exit': round(exit_price, 2),
                        'reason': reason,
                        'bars_held': bars_held,
                        'pnl': round(pnl, 2),
                        'balance': round(balance, 2),
                    })

                    trade_closed = True
                    i = df.index.get_loc(t)
                    break

                # Optional TP
                if USE_TP and high >= tp:
                    exit_price = tp * (1 - slippage_bps / 10000)

                    pnl = (exit_price - entry) * qty
                    balance += pnl
                    balance -= _fee_notional(qty * exit_price)

                    wins += 1

                    trade_logs.append({
                        'type': 'BUY',
                        'entry_time': df.index[entry_idx],
                        'exit_time': t,
                        'entry': round(entry, 2),
                        'exit': round(exit_price, 2),
                        'reason': "TP",
                        'bars_held': bars_held,
                        'pnl': round(pnl, 2),
                        'balance': round(balance, 2),
                    })

                    trade_closed = True
                    i = df.index.get_loc(t)
                    break

            # Timeout exit
            if not trade_closed and not future.empty:
                last_t = future.index[-1]
                last_close = float(future.iloc[-1]['close'])
                exit_price = last_close * (1 - slippage_bps / 10000)

                pnl = (exit_price - entry) * qty
                balance += pnl
                balance -= _fee_notional(qty * exit_price)

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                trade_logs.append({
                    'type': 'BUY',
                    'entry_time': df.index[entry_idx],
                    'exit_time': last_t,
                    'entry': round(entry, 2),
                    'exit': round(exit_price, 2),
                    'reason': "TIMEOUT",
                    'bars_held': len(future),
                    'pnl': round(pnl, 2),
                    'balance': round(balance, 2),
                })

                i = df.index.get_loc(last_t)

            balance_history.append(balance)
            i += 1
            continue

        # =========================
        # SHORT
        # =========================
        if row['sell']:
            entry_open = float(df.iloc[entry_idx]['open'])
            entry = entry_open * (1 - slippage_bps / 10000)

            initial_sl = float(row['sell_sl'])
            tp = float(row['sell_tp'])

            risk_per_unit = initial_sl - entry
            if risk_per_unit <= 0 or np.isnan(risk_per_unit):
                i += 1
                continue

            capital_risk = balance * float(risk_per_trade)  # ✅ FIX: use parameter
            qty = capital_risk / risk_per_unit

            balance -= _fee_notional(qty * entry)

            sl = initial_sl
            lowest = entry
            trade_closed = False

            future = df.iloc[entry_idx: entry_idx + max_periods]
            for bars_held, (t, candle) in enumerate(future.iterrows(), start=1):
                high = float(candle['high'])
                low = float(candle['low'])

                lowest = min(lowest, low)

                if lowest <= entry - (risk_per_unit * BREAKEVEN_R):
                    sl = min(sl, entry)

                if lowest <= entry - (risk_per_unit * TRAIL_START_R):
                    trail = lowest + (float(row['atr']) * TRAIL_ATR_MULT)
                    sl = min(sl, trail)

                if high >= sl:
                    exit_price = sl * (1 + slippage_bps / 10000)

                    pnl = (entry - exit_price) * qty
                    balance += pnl
                    balance -= _fee_notional(qty * exit_price)

                    reason = "SL" if sl == initial_sl else ("BE" if sl == entry else "TRAIL_SL")

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    trade_logs.append({
                        'type': 'SELL',
                        'entry_time': df.index[entry_idx],
                        'exit_time': t,
                        'entry': round(entry, 2),
                        'exit': round(exit_price, 2),
                        'reason': reason,
                        'bars_held': bars_held,
                        'pnl': round(pnl, 2),
                        'balance': round(balance, 2),
                    })

                    trade_closed = True
                    i = df.index.get_loc(t)
                    break

                if USE_TP and low <= tp:
                    exit_price = tp * (1 + slippage_bps / 10000)

                    pnl = (entry - exit_price) * qty
                    balance += pnl
                    balance -= _fee_notional(qty * exit_price)

                    wins += 1

                    trade_logs.append({
                        'type': 'SELL',
                        'entry_time': df.index[entry_idx],
                        'exit_time': t,
                        'entry': round(entry, 2),
                        'exit': round(exit_price, 2),
                        'reason': "TP",
                        'bars_held': bars_held,
                        'pnl': round(pnl, 2),
                        'balance': round(balance, 2),
                    })

                    trade_closed = True
                    i = df.index.get_loc(t)
                    break

            if not trade_closed and not future.empty:
                last_t = future.index[-1]
                last_close = float(future.iloc[-1]['close'])
                exit_price = last_close * (1 + slippage_bps / 10000)

                pnl = (entry - exit_price) * qty
                balance += pnl
                balance -= _fee_notional(qty * exit_price)

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

                trade_logs.append({
                    'type': 'SELL',
                    'entry_time': df.index[entry_idx],
                    'exit_time': last_t,
                    'entry': round(entry, 2),
                    'exit': round(exit_price, 2),
                    'reason': "TIMEOUT",
                    'bars_held': len(future),
                    'pnl': round(pnl, 2),
                    'balance': round(balance, 2),
                })

                i = df.index.get_loc(last_t)

            balance_history.append(balance)
            i += 1
            continue

        i += 1

    trades_df = pd.DataFrame(trade_logs)

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0

    total_return = balance - initial_balance
    total_return_pct = ((balance / initial_balance) - 1) * 100

    balance_array = np.array(balance_history, dtype=float)
    running_max = np.maximum.accumulate(balance_array)
    drawdown = ((balance_array - running_max) / running_max) * 100
    max_drawdown = float(np.min(drawdown)) if len(drawdown) else 0.0

    # ✅ FIX: avoid NaN when no winners
    avg_win = float(trades_df[trades_df['pnl'] > 0]['pnl'].mean()) if not trades_df.empty and (trades_df['pnl'] > 0).any() else 0.0
    avg_loss = float(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if not trades_df.empty and (trades_df['pnl'] < 0).any() else 0.0

    gross_profit = float(trades_df[trades_df['pnl'] > 0]['pnl'].sum()) if not trades_df.empty else 0.0
    gross_loss = float(abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())) if not trades_df.empty else 0.0

    profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else 0.0
    expectancy = float(trades_df['pnl'].mean()) if not trades_df.empty else 0.0

    metrics = {
        'balance_initial': float(round(initial_balance, 2)),
        'balance_final': float(round(balance, 2)),
        'total_return': float(round(total_return, 2)),
        'total_return_pct': float(round(total_return_pct, 2)),
        'total_trades': int(total_trades),
        'wins': int(wins),
        'losses': int(losses),
        'win_rate': float(round(win_rate, 2)),
        'avg_win': float(round(avg_win, 2)),
        'avg_loss': float(round(avg_loss, 2)),
        'profit_factor': float(round(profit_factor, 2)),
        'expectancy': float(round(expectancy, 2)),
        'max_drawdown_pct': float(round(max_drawdown, 2)),
        'avg_bars_held': float(round(trades_df['bars_held'].mean(), 1)) if not trades_df.empty else 0.0,
    }

    return trades_df, metrics