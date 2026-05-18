#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC Quant Research System - Main Entry Point (Production Hardened)

This is a bulletproof, production-ready quantitative trading research system.

Architecture:
  Data -> Strategy -> Backtest -> Metrics

Design Principles:
  - Fail fast with clear error messages
  - Validate at every step
  - Handle all edge cases gracefully
  - Support future extensions (optimizer, live trading, AI)
  - No external dependencies beyond core libraries
"""

import importlib
import os
import sys
import traceback
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import pandas as pd
import numpy as np

from strategies import tsmom_regime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)



# =============================================================================
# CONFIGURATION - ALL TUNABLE PARAMETERS IN ONE PLACE
# =============================================================================
CONFIG = {
    'data': {
        'symbol': 'BTC/USDT',
        'timeframe': '15m',
        'limit': 10000,
    },
    'backtest': {
        'initial_balance': 10000.0,
        'risk_per_trade': 0.01,
        'max_periods': 30,
    },
    'strategy': {
        'module': 'breakout',
        'name': 'BTC breakout regime',
    },
    'output': {
        'results_dir': 'results',
        'save_strategy': True,
        'save_trades': True,
        'verbose': True,
    },
    'strategy': {
    'module': 'tsmom_regime',
    'name': 'BTC intraday TSMOM + regime',
},



   
}

# Expected column names from strategy
STRATEGY_REQUIRED_COLS = {
    'buy', 'sell', 'close', 'high', 'low', 'open', 'volume',
    'buy_sl', 'buy_tp', 'sell_sl', 'sell_tp',
    'buy_risk', 'sell_risk'
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def validate_config() -> bool:
    """
    Validate CONFIG dictionary for consistency.
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Validate data config
        if CONFIG['data']['limit'] < 100:
            logger.error("CONFIG['data']['limit'] must be >= 100")
            return False
        if CONFIG['data']['limit'] > 5000:
            logger.warning("CONFIG['data']['limit'] > 5000 may cause API rate limits")
        
        # Validate backtest config
        balance = CONFIG['backtest']['initial_balance']
        if balance <= 0:
            logger.error(f"initial_balance must be > 0, got {balance}")
            return False
        
        risk = CONFIG['backtest']['risk_per_trade']
        if risk <= 0 or risk >= 0.5:
            logger.error(f"risk_per_trade must be in (0, 0.5), got {risk}")
            return False
        
        max_periods = CONFIG['backtest']['max_periods']
        if max_periods <= 0:
            logger.error(f"max_periods must be > 0, got {max_periods}")
            return False
        
        # Validate output config
        if not isinstance(CONFIG['output']['results_dir'], str):
            logger.error("results_dir must be a string")
            return False
        
        return True
    
    except KeyError as e:
        logger.error(f"CONFIG missing required key: {e}")
        return False
    except Exception as e:
        logger.error(f"CONFIG validation error: {e}")
        return False


def validate_dataframe(df: pd.DataFrame, name: str) -> bool:
    """
    Validate that DataFrame has expected structure.
    
    Args:
        df: DataFrame to validate
        name: Name for error messages
    
    Returns:
        bool: True if valid
    """
    if df is None:
        logger.error(f"{name} is None")
        return False
    
    if not isinstance(df, pd.DataFrame):
        logger.error(f"{name} is not a DataFrame, got {type(df)}")
        return False
    
    if df.empty:
        logger.error(f"{name} is empty")
        return False
    
    if len(df) < 10:
        logger.warning(f"{name} has very few rows: {len(df)}")
    
    if not df.index.is_monotonic_increasing:
        logger.error(f"{name} index is not monotonic increasing")
        return False
    
    return True


def validate_strategy_output(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has all required strategy columns.
    
    Args:
        df: DataFrame from strategy
    
    Returns:
        bool: True if all required columns present
    """
    if not validate_dataframe(df, "strategy DataFrame"):
        return False
    
    missing_cols = STRATEGY_REQUIRED_COLS - set(df.columns)
    if missing_cols:
        logger.error(f"Strategy output missing columns: {missing_cols}")
        return False
    
    # Validate data types and ranges
    try:
        # Check buy/sell are boolean
        if not pd.api.types.is_bool_dtype(df['buy']):
            logger.warning(f"buy column dtype is {df['buy'].dtype}, expected bool")
        
        if not pd.api.types.is_bool_dtype(df['sell']):
            logger.warning(f"sell column dtype is {df['sell'].dtype}, expected bool")
        
        # Check no mutual buy/sell signals
        mutual = (df['buy'] & df['sell']).sum()
        if mutual > 0:
            logger.warning(f"{mutual} rows have both buy and sell signals (should be exclusive)")
        
        # Check price columns are numeric and positive
        price_cols = ['close', 'high', 'low', 'open', 'buy_sl', 'buy_tp', 'sell_sl', 'sell_tp']
        for col in price_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column '{col}' is not numeric")
                return False
            
            if (df[col] <= 0).any():
                logger.error(f"Column '{col}' has non-positive values")
                return False
        
        # Check risk columns are numeric and non-negative
        risk_cols = ['buy_risk', 'sell_risk']
        for col in risk_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column '{col}' is not numeric")
                return False
            
            if (df[col] < 0).any():
                logger.error(f"Column '{col}' has negative values")
                return False
        
        # Check high >= low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            logger.error(f"{invalid_hl} rows have high < low")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Strategy output validation failed: {e}")
        return False


def validate_backtest_output(trades_df: pd.DataFrame, metrics: Dict) -> bool:
    """
    Validate backtest output structure.
    
    Args:
        trades_df: Trade log DataFrame
        metrics: Metrics dictionary
    
    Returns:
        bool: True if valid
    """
    if trades_df is None or metrics is None:
        logger.error("Backtest returned None")
        return False
    
    if not isinstance(trades_df, pd.DataFrame):
        logger.error(f"trades_df is not DataFrame: {type(trades_df)}")
        return False
    
    if not isinstance(metrics, dict):
        logger.error(f"metrics is not dict: {type(metrics)}")
        return False
    
    # Check required metric keys
    required_metrics = {
        'total_trades', 'wins', 'losses', 'win_rate',
        'balance_initial', 'balance_final', 'total_return',
        'total_return_pct', 'avg_win', 'avg_loss', 'expectancy',
        'profit_factor', 'max_drawdown_pct', 'avg_bars_held'
    }
    missing_metrics = required_metrics - set(metrics.keys())
    if missing_metrics:
        logger.error(f"Metrics missing keys: {missing_metrics}")
        return False
    
    # Validate metric values
    if metrics['total_trades'] < 0:
        logger.error(f"total_trades is negative: {metrics['total_trades']}")
        return False
    
    if metrics['wins'] + metrics['losses'] != metrics['total_trades']:
        logger.error(f"wins + losses != total_trades")
        return False
    
    if metrics['balance_final'] < 0:
        logger.error(f"balance_final is negative: {metrics['balance_final']}")
        return False
    
    return True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def ensure_results_dir() -> bool:
    """
    Ensure results directory exists.
    
    Returns:
        bool: True if successful
    """
    try:
        results_dir = Path(CONFIG['output']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results directory ready: {results_dir.absolute()}")
        return True
    except Exception as e:
        logger.error(f"Failed to create results directory: {e}")
        return False


def import_modules() -> bool:
    """
    Safely import required modules.
    
    Returns:
        bool: True if all imports successful
    """
    try:
        global fetch_data, apply_strategy, run_backtest

        from data_loader import fetch_data
        strategy_module = CONFIG['strategy']['module']
        strategy = importlib.import_module(f"strategies.{strategy_module}")
        apply_strategy = strategy.apply_strategy
        from backtester import run_backtest

        logger.info("All modules imported successfully")
        return True

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(
            "Make sure data_loader.py, strategies/{CONFIG['strategy']['module']}.py, backtester.py exist"
        )
        return False
    except Exception as e:
        logger.error(f"Unexpected import error: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main() -> Optional[Dict[str, Any]]:
    """
    Main execution pipeline with comprehensive error handling.
    
    Flow:
      1. Validate configuration
      2. Import modules
      3. Create output directories
      4. Fetch data
      5. Apply strategy
      6. Run backtest
      7. Save results
      8. Display summary
    
    Returns:
        dict: Results containing trades, metrics, signals if successful
        None: If any step fails
    """
    
    print("\n" + "=" * 70)
    print("BTC QUANT RESEARCH SYSTEM - PRODUCTION HARDENED")
    print("=" * 70 + "\n")
    
    try:
        # =====================================================
        # STEP 1: VALIDATE CONFIGURATION
        # =====================================================
        logger.info("Step 1/8: Validating configuration...")
        if not validate_config():
            logger.error("Configuration validation failed")
            return None
        logger.info("Configuration valid")
        
        # =====================================================
        # STEP 2: IMPORT MODULES
        # =====================================================
        logger.info("Step 2/8: Importing modules...")
        if not import_modules():
            logger.error("Module import failed")
            return None
        logger.info("Modules imported")
        
        # =====================================================
        # STEP 3: SETUP OUTPUT DIRECTORY
        # =====================================================
        logger.info("Step 3/8: Setting up output directory...")
        if not ensure_results_dir():
            logger.error("Output directory setup failed")
            return None
        logger.info("Output directory ready")
        
        # =====================================================
        # STEP 4: FETCH DATA
        # =====================================================
        logger.info("Step 4/8: Fetching market data...")
        try:
            df = fetch_data(
                symbol=CONFIG['data']['symbol'],
                timeframe=CONFIG['data']['timeframe'],
                limit=5000,
                max_bars=CONFIG['data']['limit']
            )
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            logger.error("Check network, API key, and Binance status")
            return None
        
        if not validate_dataframe(df, "Fetched data"):
            logger.error("Data validation failed")
            return None
        
        logger.info(f"Data loaded: {len(df)} candles")
        logger.info(f"  Symbol: {CONFIG['data']['symbol']}")
        logger.info(f"  Timeframe: {CONFIG['data']['timeframe']}")
        logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
        
        # =====================================================
        # STEP 5: APPLY STRATEGY
        # =====================================================
        logger.info("Step 5/8: Applying strategy...")
        try:
            df = apply_strategy(df, validate=True)
        except Exception as e:
            logger.error(f"Strategy application failed: {e}")
            traceback.print_exc()
            return None
        
        if not validate_strategy_output(df):
            logger.error("Strategy output validation failed")
            return None
        
        total_buy = int(df['buy'].sum())
        total_sell = int(df['sell'].sum())
        total_signals = total_buy + total_sell
        
        logger.info(f"Strategy applied successfully")
        logger.info(f"  Strategy module: {CONFIG['strategy']['module']}")
        logger.info(f"  Buy signals: {total_buy}")
        logger.info(f"  Sell signals: {total_sell}")
        logger.info(f"  Total signals: {total_signals}")
        
        if total_signals == 0:
            logger.warning("No signals generated - backtest will have no trades")
        
        # =====================================================
        # STEP 6: SAVE STRATEGY OUTPUT
        # =====================================================
        if CONFIG['output']['save_strategy']:
            logger.info("Step 6/8: Saving strategy output...")
            try:
                strategy_file = f"{CONFIG['strategy']['module']}_strategy_output.csv"
                strategy_path = Path(CONFIG['output']['results_dir']) / strategy_file
                df.to_csv(strategy_path)
                logger.info(f"Strategy output saved: {strategy_path}")
            except Exception as e:
                logger.error(f"Failed to save strategy output: {e}")
                return None
        else:
            logger.info("Step 6/8: Skipping strategy save (disabled in config)")
        
        # =====================================================
        # STEP 7: RUN BACKTEST
        # =====================================================
        logger.info("Step 7/8: Running backtest...")
        try:
            trades_df, metrics = run_backtest(
                df=df,
                initial_balance=CONFIG['backtest']['initial_balance'],
                risk_per_trade=CONFIG['backtest']['risk_per_trade'],
                max_periods=CONFIG['backtest']['max_periods']
            )
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            traceback.print_exc()
            return None
        
        if not validate_backtest_output(trades_df, metrics):
            logger.error("Backtest output validation failed")
            return None
        
        logger.info(f"Backtest completed successfully")
        logger.info(f"  Total trades: {metrics['total_trades']}")
        logger.info(f"  Win rate: {metrics['win_rate']:.2f}%")
        logger.info(f"  Return: {metrics['total_return_pct']:.2f}%")
        
        # =====================================================
        # STEP 8: SAVE RESULTS
        # =====================================================
        if CONFIG['output']['save_trades'] and len(trades_df) > 0:
            logger.info("Step 8/8: Saving trade logs...")
            try:
                trades_file = f"{CONFIG['strategy']['module']}_trade_logs.csv"
                trades_path = Path(CONFIG['output']['results_dir']) / trades_file
                trades_df.to_csv(trades_path, index=False)
                logger.info(f"Trade logs saved: {trades_path}")
            except Exception as e:
                logger.error(f"Failed to save trade logs: {e}")
                return None
        else:
            logger.info("Step 8/8: Skipping trade save (disabled or no trades)")
        
        # =====================================================
        # DISPLAY RESULTS
        # =====================================================
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 70 + "\n")
        
        print(f"TRADES")
        print(f"  Total trades:        {metrics['total_trades']}")
        print(f"  Winning trades:      {metrics['wins']}")
        print(f"  Losing trades:       {metrics['losses']}")
        print(f"  Win rate:            {metrics['win_rate']:.2f}%\n")
        
        print(f"PROFITABILITY")
        print(f"  Initial balance:     ${metrics['balance_initial']:,.2f}")
        print(f"  Final balance:       ${metrics['balance_final']:,.2f}")
        print(f"  Net profit/loss:     ${metrics['total_return']:,.2f}")
        print(f"  Return %:            {metrics['total_return_pct']:.2f}%\n")
        
        print(f"RISK METRICS")
        print(f"  Avg win:             ${metrics['avg_win']:,.2f}")
        print(f"  Avg loss:            ${metrics['avg_loss']:,.2f}")
        print(f"  Profit factor:       {metrics['profit_factor']:.2f}")
        print(f"  Expectancy:          ${metrics['expectancy']:,.2f}")
        print(f"  Max drawdown:        {metrics['max_drawdown_pct']:.2f}%\n")
        
        print(f"TRADE TIMING")
        print(f"  Avg bars held:       {metrics['avg_bars_held']:.1f}\n")
        
        if len(trades_df) > 0 and len(trades_df) <= 20:
            print(f"TRADE DETAILS (all trades)")
            cols = ['entry_time', 'entry', 'exit', 'reason', 'pnl', 'balance']
            display_cols = [c for c in cols if c in trades_df.columns]
            print(trades_df[display_cols].to_string(index=False))
        elif len(trades_df) > 20:
            print(f"TRADE DETAILS (last 10 of {len(trades_df)} trades)")
            cols = ['entry_time', 'entry', 'exit', 'reason', 'pnl', 'balance']
            display_cols = [c for c in cols if c in trades_df.columns]
            print(trades_df[display_cols].tail(10).to_string(index=False))
        else:
            print("TRADE DETAILS: No trades executed")
        
        print("\n" + "=" * 70)
        logger.info("Pipeline completed successfully")
        print("=" * 70 + "\n")
        
        # =====================================================
        # RETURN RESULTS
        # =====================================================
        return {
            'trades_df': trades_df,
            'metrics': metrics,
            'signals': {
                'buy': total_buy,
                'sell': total_sell,
                'total': total_signals,
            },
            'status': 'success'
        }

    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# PROGRAM ENTRY
# =============================================================================
if __name__ == "__main__":
    try:
        result = main()
        exit_code = 0 if result is not None else 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        exit_code = 1
    
    sys.exit(exit_code)
