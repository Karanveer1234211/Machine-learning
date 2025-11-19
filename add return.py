#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Augment daily parquet files with forward-looking metrics:

- ret_next1d_pct  : % return to close[t+1]
- ret_next3d_pct  : % return to close[t+3]
- ret_next1w_pct  : % return to close[t+5]  (trading-week = 5 trading days)
- peak_ret_next3d_pct : % to max(high[t+1..t+3]) vs close[t]
- peak_ret_next5d_pct : % to max(high[t+1..t+5]) vs close[t]
- max_dd_next3d_pct   : % to min(low [t+1..t+3])  vs close[t] (usually negative)
- max_dd_next5d_pct   : % to min(low [t+1..t+5])  vs close[t] (usually negative)

It also normalizes `timestamp` to tz-naive on write to avoid future tz parse issues.
"""

import os
# Relax tz validation so Polars won't panic on files that encode '+05:30'
os.environ.setdefault("POLARS_IGNORE_TIMEZONE_PARSE_ERROR", "1")

import sys
import glob
import time
import argparse
from typing import List

import polars as pl


# --------- Your folder ----------
DEFAULT_ROOT = r"C:\Users\karanvsi\Desktop\Pycharm\Cache\cache_daily_new"
PATTERN = "*_daily.parquet"


# ---------- Helpers ----------
def safe_read_parquet(path: str) -> pl.DataFrame:
    """
    Read a parquet with Polars. If tz parsing still trips, fall back to
    PyArrow/Pandas and strip timezone from 'timestamp', then return a Polars DF.
    """
    try:
        return pl.read_parquet(path)
    except Exception as e:
        # Fallback: PyArrow/Pandas -> normalize 'timestamp' -> Polars
        import pandas as pd
        try:
            pdf = pd.read_parquet(path, engine="pyarrow")
        except Exception as e2:
            raise RuntimeError(f"read_parquet failed for {path}: {e2}") from e

        if "timestamp" in pdf.columns:
            ts = pd.to_datetime(pdf["timestamp"], errors="coerce", utc=True)
            # Convert to Asia/Kolkata then drop tz -> naive
            try:
                ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            except Exception:
                # If tz_convert fails (already naive), just ensure tz removed
                try:
                    ts = ts.dt.tz_localize(None)
                except Exception:
                    pass
            pdf["timestamp"] = ts

        return pl.from_pandas(pdf)


def ensure_sorted_by_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" in df.columns:
        # Try to ensure proper Datetime
        try:
            if df["timestamp"].dtype != pl.Datetime:
                df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, strict=False))
        except Exception:
            pass
        try:
            df = df.sort("timestamp")
        except Exception:
            pass
    return df


def pct_change_future(fut: pl.Expr, cur: pl.Expr) -> pl.Expr:
    """100 * (future - current) / current, with divide-by-zero guard."""
    return (
        pl.when((cur.is_not_null()) & (cur != 0))
        .then((fut - cur) / cur * 100.0)
        .otherwise(None)
    )


def add_forward_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds:
      ret_next1d_pct, ret_next3d_pct, ret_next1w_pct
      peak_ret_next3d_pct, max_dd_next3d_pct
      peak_ret_next5d_pct, max_dd_next5d_pct
    to each row based on future windows (EXCLUDING today):
      - Next 3 trading days: t+1..t+3
      - Next 5 trading days: t+1..t+5
    Peak uses future highs; drawdown uses future lows.
    """
    needed = {"close", "high", "low"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}")

    df = ensure_sorted_by_timestamp(df)

    # Forward closes for simple forward returns
    close_next1 = pl.col("close").shift(-1)
    close_next3 = pl.col("close").shift(-3)
    close_next5 = pl.col("close").shift(-5)

    # Build lists of future shifts (EXCLUDE today by starting at shift(-1))
    future_highs_3 = [pl.col("high").shift(-k) for k in (1, 2, 3)]
    future_lows_3  = [pl.col("low").shift(-k)  for k in (1, 2, 3)]
    future_highs_5 = [pl.col("high").shift(-k) for k in (1, 2, 3, 4, 5)]
    future_lows_5  = [pl.col("low").shift(-k)  for k in (1, 2, 3, 4, 5)]

    # Use Polars' built-ins for horizontal reductions across Expr
    max_high_next3 = pl.max_horizontal(*future_highs_3)
    min_low_next3  = pl.min_horizontal(*future_lows_3)
    max_high_next5 = pl.max_horizontal(*future_highs_5)
    min_low_next5  = pl.min_horizontal(*future_lows_5)

    df = df.with_columns([
        # Forward close-to-close returns (percent)
        pct_change_future(close_next1, pl.col("close")).alias("ret_next1d_pct"),
        pct_change_future(close_next3, pl.col("close")).alias("ret_next3d_pct"),
        pct_change_future(close_next5, pl.col("close")).alias("ret_next1w_pct"),

        # Peak gain in next window (based on highs)
        pct_change_future(max_high_next3, pl.col("close")).alias("peak_ret_next3d_pct"),
        pct_change_future(max_high_next5, pl.col("close")).alias("peak_ret_next5d_pct"),

        # Max drawdown in next window (based on lows; typically negative)
        pct_change_future(min_low_next3, pl.col("close")).alias("max_dd_next3d_pct"),
        pct_change_future(min_low_next5, pl.col("close")).alias("max_dd_next5d_pct"),
    ])

    # Persist timestamp as tz-naive so future reads never fail on '+05:30'
    if "timestamp" in df.columns and df["timestamp"].dtype == pl.Datetime:
        try:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
        except Exception:
            pass

    return df


def replace_atomic_with_retry(tmp_path: str, final_path: str, max_retries: int = 6):
    """Windows-friendly atomic replace with small backoff to avoid PermissionError."""
    for k in range(max_retries):
        try:
            os.replace(tmp_path, final_path)
            return
        except PermissionError:
            time.sleep(0.15 + 0.1 * k)
    os.replace(tmp_path, final_path)


def process_one_file(path_parquet: str, make_backup: bool = False, dry_run: bool = False) -> str:
    df = safe_read_parquet(path_parquet)
    df_aug = add_forward_metrics(df)

    if dry_run:
        return "DRYRUN"

    if make_backup:
        try:
            import shutil
            shutil.copy2(path_parquet, path_parquet + ".bak")
        except Exception:
            pass

    tmp_path = path_parquet + ".tmp"
    df_aug.write_parquet(tmp_path, compression="snappy")
    replace_atomic_with_retry(tmp_path, path_parquet)
    return "OK"


# -------------- CLI ---------------
def main():
    ap = argparse.ArgumentParser(
        description="Augment daily parquet files with forward returns & peak/drawdown."
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, help="Folder with *_daily.parquet")
    ap.add_argument("--pattern", default=PATTERN, help="Glob pattern")
    ap.add_argument("--backup", action="store_true", help="Write .bak before overwrite")
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"Folder not found: {root}")
        sys.exit(1)

    files = glob.glob(os.path.join(root, args.pattern))
    if not files:
        print("No matching parquet files.")
        sys.exit(0)

    print(f"Found {len(files)} files in {root}")
    print("Adding columns:")
    print("  ret_next1d_pct, ret_next3d_pct, ret_next1w_pct")
    print("  peak_ret_next3d_pct, peak_ret_next5d_pct")
    print("  max_dd_next3d_pct,  max_dd_next5d_pct")
    if args.dry_run:
        print("DRY RUN mode")
    if args.backup:
        print("Backup mode (.bak)")

    ok, err = 0, 0
    for i, fp in enumerate(sorted(files), 1):
        base = os.path.basename(fp)
        try:
            status = process_one_file(fp, make_backup=args.backup, dry_run=args.dry_run)
            ok += 1
            print(f"[{i}/{len(files)}] {status} {base}")
        except Exception as e:
            err += 1
            print(f"[{i}/{len(files)}] ERR  {base} -> {e!s}")

    print(f"Done. OK={ok}, ERR={err}")


if __name__ == "__main__":
    main()
