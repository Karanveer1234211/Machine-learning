
# predictor_v5_5_resume_fixed.py
# v5.5-resume-fixed — GUI file/folder picker, resume panel (no rebuild unless --rebuild),
# 1D classification + isotonic calibration (version-aware), residual-based 5D std,
# faster chunking, checkpoints, SHAP, WhatWorked, combined summary.
# Author: M365 Copilot for Singh, Karanveer

import os, glob, json, time, sys, re, datetime as dt, itertools, math
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd

# ===================== DEFAULTS / PATHS =====================
DATA_DIR_DEFAULT = r"C:\\Users\\karanvsi\\Desktop\\Pycharm\\Cache\\cache_daily_new"

PANEL_OUT = None
PROCESSED_LIST = None
WATCHLIST_OUT = None
SHAP_CARDS_CSV = None
SHAP_CARDS_JSON = None
SHAP_GLOBAL_SUMMARY = None
GLOBAL_WORKS_1D_XLSX = None
GLOBAL_WORKS_3D_XLSX = None
GLOBAL_WORKS_5D_XLSX = None
SYMBOL_PLAYBOOKS_XLSX = None
ACTIONABLE_5D_OUT = None
LOG_DIR = None
LOAD_ERRORS_LOG = None
QUARANTINE_LIST = None
STATUS_PATH = None  # status.json heartbeat

# ===================== MODEL / CV =====================
FOLDS = 5
EMBARGO_DAYS = 0

# Estimators (can override via CLI)
N_EST_ALL = 300
N_EST_1D = N_EST_ALL
N_EST_3D = N_EST_ALL
N_EST_5D = N_EST_ALL

# Early stopping
EARLY_STOPPING_ROUNDS = 50

# LGB Params (shared defaults; can extend)
MAX_DEPTH_ALL = 8

# Filters for watchlist
MIN_CLOSE = 50.0
MIN_AVG20_VOL = 200_000
TOP_SHAP_PER_SYMBOL = 10
COMPUTE_SHAP_FOR = ["1d"]  # add "3d","5d" if needed
SHAP_MAX_SYMBOLS = None  # CLI

# Chunking / Excel safety
CHUNK_SIZE = 150
EXCEL_MAX = 1_000_000

# OOS summary thresholds
MIN_SUPPORT_GLOBAL = 2000
MIN_SUPPORT_COMBO = 2500

# Size-aware support thresholds for higher-order combos
MAX_COMBO_SIZE = 3  # default tighter cap (can increase via CLI)
MIN_SUPPORT_COMBO_4 = 4000
MIN_SUPPORT_COMBO_5 = 8000
TOP_ROWS_PER_SYMBOL = 5

# Approvals/gates for Actionable
WILSON_LOW_MIN = 0.50
FDR_ALPHA = 0.05

# Toggle actionable 5D overlay
WRITE_ACTIONABLE_5D = True

# HAC / inference controls
HAC_LAG = "auto"  # "auto"|int
THIN_INFERENCE = True  # thin overlapping rows for 3D/5D inference

# Notes/conviction thresholds for combined summary
STRONG_HITRATE_1D = 55.0  # %
STRONG_HITRATE_5D = 55.0  # %
MIN_PLAYBOOK_SUPPORT = 50  # rows per symbol-condition

# Probability std options for 5D
PROB_STD_METHOD = "residual"  # "residual"|"symbol_hist"|"cross"|"none"
PROB_STD_WINDOW = 252
PROB_STD_MIN_ROWS = 60

# 1D classification margin (% absolute return treated neutral and excluded from training)
CLS_MARGIN = 0.05

# ===================== ETA helper =====================
class ProgressETA:
    def __init__(self, total:int, label:str=""):
        self.total = max(1, int(total))
        self.label = label
        self.start = time.perf_counter()
        self.done = 0
        self._last = ""
    def tick(self, note:str=""):
        self.done += 1
        elapsed = max(1e-6, time.perf_counter() - self.start)
        rate = self.done / elapsed
        remain = max(0, self.total - self.done)
        eta_s = int(remain / rate) if rate > 0 else 0
        m, s = divmod(eta_s, 60); h, m = divmod(m, 60)
        eta = f"{h:02d}:{m:02d}:{s:02d}" if h>0 else f"{m:02d}:{s:02d}"
        pct = 100 * self.done / self.total
        msg = f"[{self.label}] {self.done}/{self.total} ({pct:5.1f}%) ETA {eta}"
        if note: msg += f" {note}"
        if msg != self._last:
            self._last = msg
            print(msg)

# ===================== Heartbeat =====================
def write_status(phase: str, note: str = ""):
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"),
           "phase": phase, "note": note}
    try:
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
    except Exception:
        pass

# ===================== GUI picker =====================
def pick_files_or_dir_gui() -> List[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        print("GUI not available (tkinter missing or headless). Falling back to CLI args.")
        return []
    root = tk.Tk()
    root.withdraw()
    root.update()
    files = filedialog.askopenfilenames(
        title="Select daily files (parquet/csv). Cancel to choose a folder.",
        filetypes=[("Parquet/CSV", "*.parquet *.csv"), ("All files", "*.*")]
    )
    if files:
        paths = [Path(f) for f in files]
        root.destroy()
        return paths
    folder = filedialog.askdirectory(title="Select the cache folder (daily files)")
    root.destroy()
    if folder:
        return [Path(folder)]
    return []

# ===================== CLI =====================
def parse_cli():
    import argparse
    ap = argparse.ArgumentParser(
        description="Learn 1D (classification+calibrated prob)/3D/5D returns with SHAP & OOS summaries from daily cache."
    )
    ap.add_argument("--data-dir", default=DATA_DIR_DEFAULT,
                    help="Folder containing <SYMBOL>_daily.parquet/csv.")
    ap.add_argument("--files", nargs="*", default=None,
                    help="Specific daily files to process (parquet/csv). If set, overrides --data-dir.")
    ap.add_argument("--gui", action="store_true",
                    help="Open GUI picker to choose files or a folder.")
    ap.add_argument("--panel-path", type=str, default=None,
                    help="Optional: path to an existing panel_cache.csv to load (skip collection).")
    ap.add_argument("--out-dir", default=".",
                    help="Folder to save outputs (panel, models, watchlist, SHAP, logs).")
    ap.add_argument("--symbols-like", default=None,
                    help="Regex to filter symbols (e.g., ^TCS|INFY$).")
    ap.add_argument("--limit-files", type=int, default=None,
                    help="Cap number of files to process (quick tests).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete panel_cache.csv & processed_symbols.txt before run.")
    # CV & inference rigor
    ap.add_argument("--cv-splits", type=int, default=FOLDS,
                    help="Number of walk-forward folds (default=5).")
    ap.add_argument("--embargo-days", type=int, default=EMBARGO_DAYS,
                    help="Embargo gap in days between train/test split (default=0).")
    ap.add_argument("--hac-lag", default=HAC_LAG,
                    help='Lag for HAC (Newey–West) SE of mean: "auto" or integer.')
    ap.add_argument("--thin-inference", type=str, default=str(THIN_INFERENCE),
                    help='Use non-overlapping rows for 3D/5D WhatWorked ("true"/"false").')
    # Accuracy windows & symbol reports
    ap.add_argument("--last-month-days", type=int, default=30,
                    help="Calendar days for recent-accuracy report (default=30).")
    ap.add_argument("--write-symbol-accuracy", type=str, default="true",
                    help='Write symbol-wise accuracy report across full OOS ("true"/"false").')
    # WhatWorked knobs
    ap.add_argument("--max-combo-size", type=int, default=MAX_COMBO_SIZE,
                    help="Maximum combo size for WhatWorked (default=3).")
    ap.add_argument("--min-support-global", type=int, default=MIN_SUPPORT_GLOBAL,
                    help="Minimal support for singles (default=2000).")
    ap.add_argument("--min-support-combo", type=int, default=MIN_SUPPORT_COMBO,
                    help="Minimal support for size 2–3 combos (default=2500).")
    ap.add_argument("--min-support-combo-4", type=int, default=MIN_SUPPORT_COMBO_4)
    ap.add_argument("--min-support-combo-5", type=int, default=MIN_SUPPORT_COMBO_5)
    # Optional Excel-friendly exports
    ap.add_argument("--excel-compact", action="store_true",
                    help="Write panel_compact.xlsx (Latest + SymbolsSummary).")
    ap.add_argument("--excel-top-rows", type=int, default=None,
                    help=f"Write panel_topN.xlsx with first N rows (≤{EXCEL_MAX}).")
    # Accept broader daily naming
    ap.add_argument("--accept-any-daily", type=str, default="false",
                    help='Accept *.parquet/*.csv besides *_daily.* ("true"/"false").')
    # Perf/quality knobs
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                    help="Symbols per write chunk for panel (default=150).")
    ap.add_argument("--n-estimators-1d", type=int, default=N_EST_1D,
                    help="LightGBM trees for 1D model (default=300).")
    ap.add_argument("--n-estimators-3d", type=int, default=N_EST_3D,
                    help="LightGBM trees for 3D model (default=300).")
    ap.add_argument("--n-estimators-5d", type=int, default=N_EST_5D,
                    help="LightGBM trees for 5D model (default=300).")
    ap.add_argument("--early-stopping-rounds", type=int, default=EARLY_STOPPING_ROUNDS,
                    help="LightGBM early stopping rounds (0=disable, default=50).")
    ap.add_argument("--shap-max-symbols", type=int, default=None,
                    help="Cap count of symbols for SHAP after liquidity filter (optional).")
    ap.add_argument("--prob-std-method", type=str, default=PROB_STD_METHOD,
                    choices=["residual", "symbol_hist", "cross", "none"],
                    help="Std method for prob_up_5d: residual (OOS error), symbol_hist (realized 5D rolling), cross (pred 5D x-sec), none (heuristic).")
    ap.add_argument("--prob-std-window", type=int, default=PROB_STD_WINDOW,
                    help="Rolling window (days) for symbol_hist (default=252).")
    ap.add_argument("--prob-std-min-rows", type=int, default=PROB_STD_MIN_ROWS,
                    help="Minimum rows required for symbol_hist std (default=60).")
    # 1D classification mode
    ap.add_argument("--train-1d-mode", type=str, default="cls",
                    choices=["cls", "reg", "both"],
                    help="Train 1D as classification (calibrated prob), regression, or both (default=cls).")
    ap.add_argument("--cls-margin", type=float, default=CLS_MARGIN,
                    help="Neutral margin (%) around 0 for 1D classification; rows with |ret|≤margin are excluded.")
    return ap.parse_args()

def setup_paths(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    global PANEL_OUT, PROCESSED_LIST, WATCHLIST_OUT
    global SHAP_CARDS_CSV, SHAP_CARDS_JSON, SHAP_GLOBAL_SUMMARY
    global GLOBAL_WORKS_1D_XLSX, GLOBAL_WORKS_3D_XLSX, GLOBAL_WORKS_5D_XLSX
    global SYMBOL_PLAYBOOKS_XLSX, ACTIONABLE_5D_OUT
    global LOG_DIR, LOAD_ERRORS_LOG, QUARANTINE_LIST, STATUS_PATH
    PANEL_OUT = str(out / "panel_cache.csv")
    PROCESSED_LIST = str(out / "processed_symbols.txt")
    WATCHLIST_OUT = str(out / "watchlist_model_next_1_3_5d.csv")
    SHAP_CARDS_CSV = str(out / "shap_cards_latest.csv")
    SHAP_CARDS_JSON = str(out / "shap_lookout_cards_latest.json")
    SHAP_GLOBAL_SUMMARY = str(out / "shap_global_summary.csv")
    GLOBAL_WORKS_1D_XLSX = str(out / "global_what_works_1d.xlsx")
    GLOBAL_WORKS_3D_XLSX = str(out / "global_what_works_3d.xlsx")
    GLOBAL_WORKS_5D_XLSX = str(out / "global_what_works_5d.xlsx")
    SYMBOL_PLAYBOOKS_XLSX = str(out / "symbol_playbooks.xlsx")
    ACTIONABLE_5D_OUT = str(out / "actionable_watchlist_5d.csv")
    LOG_DIR = out / "logs"
    LOG_DIR.mkdir(exist_ok=True)
    LOAD_ERRORS_LOG = LOG_DIR / "load_errors.csv"
    QUARANTINE_LIST = out / "quarantine_files.txt"
    STATUS_PATH = str(out / "status.json")

# ===================== IO helpers (load/save) =====================
def _strict_file_list(data_dir: str,
                      symbols_like: Optional[str],
                      limit_files: Optional[int],
                      accept_any_daily: bool = False) -> List[Path]:
    paths: List[str] = []
    paths += glob.glob(os.path.join(data_dir, "*_daily.parquet"))
    paths += glob.glob(os.path.join(data_dir, "*_daily.csv"))
    if str(accept_any_daily).lower() in ("true","1","yes","y","t"):
        paths += glob.glob(os.path.join(data_dir, "*.parquet"))
        paths += glob.glob(os.path.join(data_dir, "*.csv"))
    paths = sorted(set(paths))
    if symbols_like:
        pat = re.compile(symbols_like)
        filtered = []
        for p in paths:
            sym = Path(p).name.replace("_daily.parquet","").replace("_daily.csv","")
            if pat.search(sym):
                filtered.append(p)
        paths = filtered or paths
    if limit_files and limit_files > 0:
        paths = paths[:limit_files]
    return [Path(p) for p in paths]

def _log_load_error(sym: str, filename: str, error: str):
    rec = {"symbol": sym, "file": filename, "error": error,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        df = pd.DataFrame([rec])
        mode = "a" if Path(LOAD_ERRORS_LOG).exists() else "w"
        df.to_csv(LOAD_ERRORS_LOG, mode=mode,
                  header=not Path(LOAD_ERRORS_LOG).exists(), index=False)
    except Exception:
        pass
    with open(QUARANTINE_LIST, "a", encoding="utf-8") as f:
        f.write(f"{filename}\n")

def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    seen: Dict[str,int] = {}
    new_cols: List[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            k = seen[c]; seen[c] = k + 1
            new_cols.append(f"{c}__dup{k}")
    df = df.copy()
    df.columns = new_cols
    return df

def load_one(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Load failed: {e}")
    if "timestamp" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        else:
            raise RuntimeError("'timestamp' column missing")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    df = _ensure_unique_columns(df)
    return df

# ===================== CPR + features =====================
def _unify_categorical(df: pd.DataFrame, base_name: str) -> pd.Series:
    cols = [c for c in df.columns if c == base_name or c.startswith(base_name + "__dup")]
    if not cols:
        return pd.Series(index=df.index, dtype="object")
    s = pd.Series(index=df.index, dtype="object")
    for c in cols:
        sc = df[c].astype("string")
        s = sc.where(sc.notna(), s)
    return s

DAILY_BASE_FEATURES = [
    "D_golden_regime","D_ema_stack_20_50_100","D_rsi7","D_rsi14","D_rsi7_gt_rsi14",
    "D_adx14","D_pdi14","D_mdi14","D_ema20_angle_deg",
    "D_daily_trend","D_weekly_trend","D_monthly_trend",
    "D_cpr_pivot","D_cpr_bc","D_cpr_tc","D_cpr_width_pct",
    "D_tmr_cpr_bc","D_tmr_cpr_tc",
    "D_obv","D_obv_slope","D_price_and_obv_rising","D_vpoc","D_weekly_vpoc",
    "D_nr","D_nr_length","D_nr_day",
    "D_sma5","D_sma20","D_sma50","D_sma200",
    "D_atr14","D_atr30","D_atr_ratio_14_30","D_range_to_atr14",
    "D_hh","D_hl","D_lh","D_ll",
    "D_prev_high","D_prev_low","D_prev_close",
    "D_pivot","D_support1","D_resistance1","D_support2","D_resistance2",
    "D_oli","D_inside_day","D_prev_inside_day",
]
LAG_FEATURES = [
    "D_rsi14_lag1","D_rsi14_lag2",
    "D_adx14_lag1","D_adx14_lag2",
    "D_ema20_angle_deg_lag1","D_ema20_angle_deg_lag2",
    "D_obv_slope_lag1","D_obv_slope_lag2",
]
CPR_YDAY = [f"CPR_Yday_{x}" for x in ("Above","Below","Inside","Overlap")]
CPR_TMR = [f"CPR_Tmr_{x}" for x in ("Above","Below","Inside","Overlap")]
STRUCT_ONEHOT = ["Struct_uptrend","Struct_downtrend","Struct_range"]
DAYTYPE_ONEHOT = ["DayType_bullish","DayType_bearish","DayType_inside"]

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    for h in (1,3,5):
        df[f"ret_{h}d_close_pct"] = (df["close"].shift(-h) / df["close"] - 1) * 100
    # open-to-close variants
    df["ret_next_open_to_close_pct"] = (df["close"].shift(-1) / df["open"].shift(-1) - 1) * 100
    for h in (1,3,5):
        df[f"ret_next_open_to_close_{h}d_pct"] = (df["close"].shift(-h) / df["open"].shift(-1) - 1) * 100
        hi = df["high"].shift(-1).rolling(h, min_periods=1).max()
        lo = df["low"].shift(-1).rolling(h, min_periods=1).min()
        df[f"mfe_{h}d_pct"] = (hi / df["close"] - 1) * 100
        df[f"mae_{h}d_pct"] = (lo / df["close"] - 1) * 100
    # classification target helper: raw sign (we'll apply margin later)
    df["ret_1d_sign"] = np.sign(df["ret_1d_close_pct"])
    return df

def add_lags(df: pd.DataFrame, cols: List[str], lags: Tuple[int,int]=(1,2)):
    for c in cols:
        if c in df.columns:
            for L in lags:
                df[f"{c}_lag{L}"] = df[c].shift(L)
    return df

def featureize(df: pd.DataFrame):
    df = add_lags(df, ["D_rsi14","D_adx14","D_ema20_angle_deg","D_obv_slope"], lags=(1,2))
    # Unify CPR categorical variants to one-hot
    yday_unified = _unify_categorical(df, "D_cpr_vs_yday")
    tmr_unified = _unify_categorical(df, "D_tmr_cpr_vs_today")
    if yday_unified.notna().any():
        df = df.drop(columns=[c for c in df.columns if c == "D_cpr_vs_yday" or c.startswith("D_cpr_vs_yday__dup")], errors="ignore")
        df["D_cpr_vs_yday_unified"] = yday_unified
        df = pd.get_dummies(df, columns=["D_cpr_vs_yday_unified"], prefix="CPR_Yday")
    if tmr_unified.notna().any():
        df = df.drop(columns=[c for c in df.columns if c == "D_tmr_cpr_vs_today" or c.startswith("D_tmr_cpr_vs_today__dup")], errors="ignore")
        df["D_tmr_cpr_vs_today_unified"] = tmr_unified
        df = pd.get_dummies(df, columns=["D_tmr_cpr_vs_today_unified"], prefix="CPR_Tmr")
    for col in CPR_YDAY:
        if col not in df.columns: df[col] = 0
    for col in CPR_TMR:
        if col not in df.columns: df[col] = 0
    # Optional structure/day onehots
    if "D_structure_trend" in df.columns:
        df["D_structure_trend"] = df["D_structure_trend"].astype("string")
        df = pd.get_dummies(df, columns=["D_structure_trend"], prefix="Struct")
    for col in STRUCT_ONEHOT:
        if col not in df.columns: df[col] = 0
    if "D_day_type" in df.columns:
        df["D_day_type"] = df["D_day_type"].astype("string")
        df = pd.get_dummies(df, columns=["D_day_type"], prefix="DayType")
    for col in DAYTYPE_ONEHOT:
        if col not in df.columns: df[col] = 0
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    feats = DAILY_BASE_FEATURES + LAG_FEATURES + CPR_YDAY + CPR_TMR + STRUCT_ONEHOT + DAYTYPE_ONEHOT
    return df, feats

def score_bias(df: pd.DataFrame) -> pd.DataFrame:
    col = lambda c: df[c] if c in df.columns else pd.Series([np.nan]*len(df))
    golden = col("D_golden_regime").fillna(False).astype(bool)
    ema_stack = col("D_ema_stack_20_50_100").fillna(False).astype(bool)
    rsi14 = pd.to_numeric(col("D_rsi14"), errors="coerce")
    rsi_cross = col("D_rsi7_gt_rsi14").fillna(False).astype(bool)
    adx14 = pd.to_numeric(col("D_adx14"), errors="coerce")
    pdi14 = pd.to_numeric(col("D_pdi14"), errors="coerce")
    mdi14 = pd.to_numeric(col("D_mdi14"), errors="coerce")
    obv_rise = col("D_price_and_obv_rising").fillna(False).astype(bool)
    atr14 = pd.to_numeric(col("D_atr14"), errors="coerce")
    cpr_w = pd.to_numeric(col("D_cpr_width_pct"), errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")
    avg20 = vol.rolling(20, min_periods=1).mean()
    cpr_above = (df.get("CPR_Yday_Above", 0) == 1)
    cpr_below = (df.get("CPR_Yday_Below", 0) == 1)
    long_score = (
        (golden*2).astype(float) + (ema_stack*2).astype(float) + (rsi_cross*1).astype(float) +
        ((rsi14.between(50,70, inclusive="both").fillna(False).astype(float)*1)) +
        ((((adx14>20)&(pdi14>mdi14)).fillna(False).astype(float)*1.5)) +
        (obv_rise*1).astype(float) + (cpr_above.astype(float)*0.5) +
        (((pd.to_numeric(col("D_daily_trend"), errors="coerce")==1) &
          (pd.to_numeric(col("D_weekly_trend"), errors="coerce")==1)).fillna(False).astype(float)*0.5)
    )
    short_score = (
        ((~golden)*1).astype(float) + ((~ema_stack)*1).astype(float) +
        ((rsi14<45).fillna(False).astype(float)*1) +
        (((adx14>20)&(mdi14>pdi14)).fillna(False).astype(float)*1.5) +
        (cpr_below.astype(float)*0.5) +
        (((pd.to_numeric(col("D_daily_trend"), errors="coerce")==-1) &
          (pd.to_numeric(col("D_weekly_trend"), errors="coerce")==-1)).fillna(False).astype(float)*0.5)
    )
    atr_pct = (atr14/df["close"]).replace([np.inf,-np.inf],np.nan)*100
    risk_pen = (atr_pct>4).fillna(False).astype(float)*0.5 + (cpr_w>1.0).fillna(False).astype(float)*0.3
    liq_pen = (avg20<MIN_AVG20_VOL).fillna(False).astype(float)*0.5
    df["long_score"] = long_score - risk_pen - liq_pen
    df["short_score"] = short_score - risk_pen - liq_pen
    return df

# ===================== Autosave (panel) =====================
def load_processed_symbols() -> set:
    s = set()
    p = Path(PROCESSED_LIST)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line: s.add(line)
    return s

def mark_processed(sym: str):
    with open(PROCESSED_LIST, "a", encoding="utf-8") as f:
        f.write(sym+"\n")

def _unique_preserve(seq: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

MASTER_KEEP_STATIC = _unique_preserve(
    ["timestamp","symbol","open","high","low","close","volume",
     "ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
     "long_score","short_score","D_atr14","D_cpr_width_pct"]
    + DAILY_BASE_FEATURES + LAG_FEATURES + CPR_YDAY + CPR_TMR + STRUCT_ONEHOT + DAYTYPE_ONEHOT
)

def append_panel_rows(chunks: List[pd.DataFrame], header_written: bool):
    if not chunks: return header_written
    aligned: List[pd.DataFrame] = []
    for df in chunks:
        df = _ensure_unique_columns(df)
        for col in MASTER_KEEP_STATIC:
            if col not in df.columns:
                if (col.startswith("CPR_Yday_") or col.startswith("CPR_Tmr_") or
                    col.startswith("Struct_") or col.startswith("DayType_")):
                    df[col] = 0
                else:
                    df[col] = np.nan
        df = df.reindex(columns=MASTER_KEEP_STATIC)
        aligned.append(df)
    df_all = pd.concat(aligned, ignore_index=True, sort=False)
    mode = "a" if header_written else "w"
    df_all.to_csv(PANEL_OUT, mode=mode, header=not header_written, index=False)
    return True

def _derive_symbol_name(p: Path) -> str:
    name = p.name
    name = name.replace("_daily.parquet","").replace("_daily.csv","")
    name = re.sub(r"\.(parquet|csv)$","",name, flags=re.IGNORECASE)
    return name

def collect_panel_from_paths(paths: List[Path]):
    # Expand any directories to files here to keep function simple
    expanded: List[Path] = []
    for p in paths:
        if Path(p).is_dir():
            expanded += _strict_file_list(str(p), None, None, accept_any_daily=False)
        else:
            expanded.append(Path(p))
    paths = expanded

    total = len(paths)
    if total == 0:
        pd.DataFrame(columns=MASTER_KEEP_STATIC).to_csv(PANEL_OUT, index=False)
        raise SystemExit(
            "No matching files found.\n"
            "Select files via --files or --gui, or provide --data-dir with *_daily.* files."
        )
    processed = load_processed_symbols()
    header_written = Path(PANEL_OUT).exists()
    eta = ProgressETA(total=total, label="Load+Engineer")
    chunk: List[pd.DataFrame] = []
    total_rows_written = 0
    for p in paths:
        path_obj = Path(p)
        sym = _derive_symbol_name(path_obj)
        if sym in processed:
            eta.tick(f"SKIP {sym}")
            continue
        try:
            df = load_one(path_obj)
            # upstream sanity checks (ranges)
            def _range_check(col, lo, hi):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    bad = (~s.between(lo,hi)) & s.notna()
                    if bad.any():
                        _log_load_error(sym, str(path_obj), f"Range anomaly {col} out of [{lo},{hi}] on {bad.sum()} rows")
            _range_check("D_rsi14", 0, 100)
            df = add_targets(df)
            df, feats = featureize(df)
            df = score_bias(df)
            df["symbol"] = sym
            df_train = df.dropna(subset=["ret_1d_close_pct"])
            if df_train.empty:
                nrows = len(df)
                msg = (f"NO TRAIN {sym} rows={nrows} "
                       f"req_cols_ok={all(c in df.columns for c in ['timestamp','open','high','low','close','volume'])}")
                print("[Load+Engineer]", msg)
                eta.tick(msg)
                mark_processed(sym)
                continue
            rows = df_train[["timestamp","symbol","open","high","low","close","volume"] + feats +
                            ["ret_1d_close_pct","ret_3d_close_pct","ret_5d_close_pct",
                             "long_score","short_score","D_atr14","D_cpr_width_pct"]].copy()
            chunk.append(rows)
            total_rows_written += len(rows)
            if len(chunk) >= CHUNK_SIZE:
                header_written = append_panel_rows(chunk, header_written)
                chunk.clear()
            mark_processed(sym)
            eta.tick(f"OK {sym} (+{len(rows)} rows)")
        except KeyboardInterrupt:
            print("\nInterrupted! Autosaving current chunk...")
            if chunk:
                header_written = append_panel_rows(chunk, header_written)
                chunk.clear()
            raise
        except Exception as e:
            _log_load_error(sym, str(path_obj), str(e))
            eta.tick(f"ERR {sym}: {e}")
            # do NOT mark processed on failure; allow retry next run
    if chunk:
        header_written = append_panel_rows(chunk, header_written)
        chunk.clear()
    print(f"[Panel] Total rows appended: {total_rows_written}")
    if not Path(PANEL_OUT).exists():
        pd.DataFrame(columns=MASTER_KEEP_STATIC).to_csv(PANEL_OUT, index=False)
        raise SystemExit(
            "No rows were written to panel_cache.csv.\n"
            "Possible causes:\n"
            " • Files produced 0 trainable rows after target dropna\n"
            "Fixes:\n"
            " • Try a small run: --limit-files 5 --rebuild\n"
            " • Inspect a file to confirm columns: [timestamp, open, high, low, close, volume]\n"
            " • Ensure enough rows so ret_1d_close_pct isn’t all NaN"
        )
    panel = pd.read_csv(PANEL_OUT, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
    panel = panel.dropna(subset=["timestamp"]).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    feats = [
        c for c in MASTER_KEEP_STATIC
        if c.startswith("D_") or c.startswith("CPR_Yday_") or c.startswith("CPR_Tmr_")
        or c.startswith("Struct_") or c.startswith("DayType_")
    ]
    return panel, feats

# ===================== Type sanitization =====================
def sanitize_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
        elif X[c].dtype == object:
            s = X[c].astype(str).str.lower()
            uniq = set(s.unique())
            if uniq <= {"true","false","nan"}:
                X[c] = s.map({"true":1,"false":0}).astype("Int64").fillna(0).astype(int)
            else:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        if c.startswith(("CPR_Yday_","CPR_Tmr_","Struct_","DayType_")):
            X[c] = X[c].fillna(0).astype(int)
    return X

# ===================== Vectorized Normal CDF + prob_up =====================
def _phi_approx(z: np.ndarray) -> np.ndarray:
    """Vectorized normal CDF approximation Φ(z) using Abramowitz–Stegun 7.1.26."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos_inf = np.isposinf(z); neg_inf = np.isneginf(z)
    out[pos_inf] = 1.0
    out[neg_inf] = 0.0
    finite = np.isfinite(z)
    if np.any(finite):
        x = z[finite]
        t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
        a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        poly = ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
        nd = (1.0 / np.sqrt(2.0*np.pi)) * np.exp(-0.5 * x*x)
        approx = 1.0 - nd * poly
        out[finite] = np.where(x >= 0, approx, 1.0 - approx)
    return out

def prob_up_from_gaussian(mean, std):
    """
    Probability P(R>0) under Gaussian(mean, std^2) using Φ(mean/std).
    Fallback when std <= 0 / NaN reflects sign of mean.
    """
    mean = np.asarray(mean, dtype=float)
    std  = np.asarray(std, dtype=float)
    z = np.divide(mean, std,
                  out=np.full_like(mean, np.nan, dtype=float),
                  where=(std > 0))
    p = _phi_approx(z)
    fallback = np.where(mean > 0, 0.75, np.where(mean < 0, 0.25, 0.50))
    p = np.where(np.isfinite(p), p, fallback)
    return np.clip(p, 0.0, 1.0)

# ===================== Time splits (walk-forward by date) =====================
def time_cv_by_timestamp(panel: pd.DataFrame,
                         n_splits: int = 5,
                         embargo_days: int = 0,
                         target_mask: Optional[pd.Series] = None):
    """
    Generator of (train_idx, test_idx) splits based on unique normalized dates.
    Keeps temporal order; optional embargo before test window.
    """
    idx = panel.index if target_mask is None else panel.index[target_mask]
    ts_all = pd.to_datetime(panel.loc[idx, "timestamp"]).dt.normalize()
    uniq_dates = pd.Series(ts_all).sort_values().unique()
    if len(uniq_dates) < n_splits + 1:
        n_splits = max(1, min(len(uniq_dates) - 1, n_splits))
    cut = np.linspace(0, len(uniq_dates), n_splits + 1, dtype=int)
    for i in range(n_splits):
        start_date = uniq_dates[cut[i]]
        end_date = uniq_dates[cut[i+1]-1] if i < n_splits - 1 else uniq_dates[-1]
        te_mask = (panel["timestamp"].dt.normalize() >= start_date) & (panel["timestamp"].dt.normalize() <= end_date)
        tr_mask = (panel["timestamp"].dt.normalize() < start_date)
        if embargo_days and embargo_days > 0:
            embargo_edge = start_date - pd.Timedelta(days=int(embargo_days))
            tr_mask = (panel["timestamp"].dt.normalize() <= embargo_edge)
        tr_idx = panel.index[tr_mask & panel.index.isin(idx)]
        te_idx = panel.index[te_mask & panel.index.isin(idx)]
        if len(te_idx) > 0 and len(tr_idx) > 0:
            yield tr_idx, te_idx

# ===================== LightGBM checks =====================
def _check_lightgbm():
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor, LGBMClassifier
        return lgb, LGBMRegressor, LGBMClassifier
    except Exception as e:
        raise SystemExit("LightGBM is not installed. Please run: pip install lightgbm") from e

# --- sklearn calibration helper (handles old/new API) ---
def _make_calibrator(est, method="isotonic"):
    """
    Returns a CalibratedClassifierCV compatible with both old (base_estimator)
    and new (estimator) scikit-learn versions.
    """
    from sklearn.calibration import CalibratedClassifierCV
    try:
        # Newer scikit-learn (>= 1.0): uses 'estimator'
        return CalibratedClassifierCV(estimator=est, method=method, cv="prefit")
    except TypeError:
        # Older scikit-learn: falls back to 'base_estimator'
        return CalibratedClassifierCV(base_estimator=est, method=method, cv="prefit")

# ---------- 1D classification with calibration ----------
def train_1d_cls_calibrated(panel: pd.DataFrame, feats: List[str], margin_pct: float,
                            n_estimators: int, n_splits: int, embargo_days: int,
                            early_stopping_rounds: int):
    lgb, _, LGBMClassifier = _check_lightgbm()
    # Label: 1 if ret > margin; 0 if ret < -margin; NaN otherwise (excluded)
    r = pd.to_numeric(panel["ret_1d_close_pct"], errors="coerce")
    y = pd.Series(np.nan, index=panel.index)
    y[(r > margin_pct)] = 1
    y[(r < -margin_pct)] = 0
    mask = y.notna()
    if not mask.any():
        raise ValueError("No labeled rows for 1D classification after applying margin.")
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(panel, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(panel.loc[mask, feats].copy())
    y_full = y.loc[mask].astype(int).values
    oos_prob = np.full(len(X_full), np.nan)
    eta = ProgressETA(total=len(folds), label="Train 1D-CLS")
    models = []

    def make_lgbm(random_state: int):
        depth = MAX_DEPTH_ALL if isinstance(MAX_DEPTH_ALL, int) and MAX_DEPTH_ALL > 0 else -1
        return LGBMClassifier(
            n_estimators=int(n_estimators),
            learning_rate=0.05,
            num_leaves=63,
            max_depth=depth,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=80,
            n_jobs=-1,
            random_state=int(random_state),
            verbosity=-1,
        )

    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0:
            continue
        clf = make_lgbm(42 + fold_no)
        X_tr = X_full.iloc[tr_pos]; y_tr = y_full[tr_pos]
        X_te = X_full.iloc[te_pos]; y_te = y_full[te_pos]
        callbacks = [lgb.callback.log_evaluation(period=0)]
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.insert(0, lgb.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], eval_metric="binary_logloss", callbacks=callbacks)

        # Calibrate on the same validation fold
        calib = _make_calibrator(clf, method="isotonic")
        calib.fit(X_te, y_te)
        prob = calib.predict_proba(X_te)[:, 1]
        oos_prob[te_pos] = prob
        eta.tick(f"fold {fold_no} n={len(te_pos)}")
        models.append(calib)

    # Final calibrated model: reserve last 20% (by date) for calibration
    final_base = make_lgbm(1234)
    df_valid = panel.loc[mask, ["timestamp"]].copy()
    order = np.argsort(df_valid["timestamp"].values)
    split = int(round(len(order) * 0.8))
    tr_order = order[:split]; te_order = order[split:]
    X_tr_all = X_full.iloc[tr_order]; y_tr_all = y_full[tr_order]
    X_te_all = X_full.iloc[te_order]; y_te_all = y_full[te_order]
    callbacks = [lgb.callback.log_evaluation(period=0)]
    if early_stopping_rounds and early_stopping_rounds > 0:
        callbacks.insert(0, lgb.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
    final_base.fit(X_tr_all, y_tr_all, eval_set=[(X_te_all, y_te_all)], eval_metric="binary_logloss", callbacks=callbacks)

    final_calib = _make_calibrator(final_base, method="isotonic")
    final_calib.fit(X_te_all, y_te_all)

    return final_calib, oos_prob, valid_idx

# ---------- Regression (for 3D/5D; optional for 1D) ----------
def train_rf(panel: pd.DataFrame, feats: List[str], target_col: str, label: str,
             n_estimators: int = N_EST_ALL, refit_final: bool = False, n_splits: int = FOLDS,
             embargo_days: int = EMBARGO_DAYS, early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    lgb, LGBMRegressor, _ = _check_lightgbm()
    mask = panel[target_col].notna()
    if not mask.any():
        raise ValueError(f"No non-NaN rows for target {target_col}")
    valid_idx = panel.index[mask].to_numpy()
    folds = list(time_cv_by_timestamp(panel, n_splits=n_splits, embargo_days=embargo_days, target_mask=mask))
    X_full = sanitize_feature_matrix(panel.loc[mask, feats].copy())
    y_full = panel.loc[mask, target_col].values
    oos = np.full(len(X_full), np.nan)
    eta = ProgressETA(total=len(folds), label=f"Train {label}")
    models = []

    def make_lgbm(random_state: int):
        depth = MAX_DEPTH_ALL if isinstance(MAX_DEPTH_ALL, int) and MAX_DEPTH_ALL > 0 else -1
        return LGBMRegressor(
            n_estimators=int(n_estimators),
            learning_rate=0.05,
            num_leaves=63,
            max_depth=depth,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            min_data_in_leaf=80,
            n_jobs=-1,
            random_state=int(random_state),
            verbosity=-1,
        )

    for fold_no, (tr_idx, te_idx) in enumerate(folds, start=1):
        tr_pos = np.where(np.isin(valid_idx, tr_idx))[0]
        te_pos = np.where(np.isin(valid_idx, te_idx))[0]
        if len(te_pos) == 0 or len(tr_pos) == 0:
            continue
        gbm = make_lgbm(42 + fold_no)
        X_tr = X_full.iloc[tr_pos]; y_tr = y_full[tr_pos]
        X_te = X_full.iloc[te_pos]; y_te = y_full[te_pos]
        callbacks = [lgb.callback.log_evaluation(period=0)]
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.insert(0, lgb.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
        gbm.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            eval_metric="l2",
            callbacks=callbacks
        )
        pred = gbm.predict(X_te, num_iteration=getattr(gbm, "best_iteration_", None))
        oos[te_pos] = pred
        mae = float(np.mean(np.abs(y_te - pred)))
        eta.tick(f"fold {fold_no} MAE={mae:.3f}% n={len(te_pos)}")
        models.append(gbm)

    final = models[-1] if models else make_lgbm(1234)
    if not models:
        final.fit(X_full, y_full, callbacks=[lgb.callback.log_evaluation(period=0)])
    if refit_final:
        print(f"[Train {label}] refit on ALL valid rows ...")
        final = make_lgbm(1234)
        callbacks = [lgb.callback.log_evaluation(period=0)]
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.insert(0, lgb.callback.early_stopping(stopping_rounds=int(early_stopping_rounds)))
        final.fit(X_full, y_full, eval_set=[(X_full, y_full)], eval_metric="l2", callbacks=callbacks)
    return final, oos, valid_idx

# ===================== Watchlist helpers =====================
def _cross_sectional_std(values: np.ndarray, floor: float = 1e-6) -> float:
    s = float(np.nanstd(values))
    return s if np.isfinite(s) and s > floor else 1.0

def _symbol_hist_roll_std(panel: pd.DataFrame, window: int, min_rows: int) -> pd.Series:
    df = panel.sort_values(["symbol","timestamp"])
    grp = df.groupby("symbol", sort=False)
    roll = grp["ret_5d_close_pct"].apply(lambda x: pd.to_numeric(x, errors="coerce").rolling(window, min_periods=min_rows).std())
    df2 = df.copy()
    df2["roll_std_5d"] = roll.values
    last = df2.groupby("symbol", as_index=True)["roll_std_5d"].tail(1)
    return last

def _residual_std_5d(panel: pd.DataFrame, oos5_df: pd.DataFrame) -> Dict[str, float]:
    if oos5_df is None or oos5_df.empty:
        return {}
    idx = oos5_df["panel_idx"].values
    pred = pd.to_numeric(oos5_df["oos_pred"], errors="coerce").values
    in_panel = np.isin(idx, panel.index.values)
    idx = idx[in_panel]; pred = pred[in_panel]
    reals = pd.to_numeric(panel.loc[idx, "ret_5d_close_pct"], errors="coerce").values
    syms = panel.loc[idx, "symbol"].values
    resid = reals - pred
    df = pd.DataFrame({"symbol": syms, "resid": resid})
    stds = df.groupby("symbol")["resid"].std().to_dict()
    return {k: float(v) for k, v in stds.items() if np.isfinite(v) and v > 1e-6}

def nightly_watchlist(panel: pd.DataFrame, feats: List[str],
                      m1_cls, m1_reg, m3_reg, m5_reg,
                      oos5_path: Optional[Path],
                      prob_std_method: str = PROB_STD_METHOD,
                      prob_std_window: int = PROB_STD_WINDOW,
                      prob_std_min_rows: int = PROB_STD_MIN_ROWS):
    panel = panel.copy().sort_values(["symbol", "timestamp"])
    panel["avg20_vol"] = panel.groupby("symbol")["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    last = panel.groupby("symbol", as_index=False).tail(1).copy()
    X = sanitize_feature_matrix(last[feats].copy())

    # Predictions
    if m1_cls is not None:
        last["prob_up_1d"] = m1_cls.predict_proba(X)[:, 1]
    else:
        last["prob_up_1d"] = np.nan

    if m1_reg is not None:
        last["pred_ret_1d_pct"] = m1_reg.predict(X)
    else:
        last["pred_ret_1d_pct"] = np.nan

    m3_use = m3_reg
    m5_use = m5_reg
    last["pred_ret_3d_pct"] = m3_use.predict(X) if m3_use is not None else np.nan
    best_it_5 = getattr(m5_use, "best_iteration_", None) if m5_use is not None else None
    last["pred_ret_5d_pct"] = (m5_use.predict(X, num_iteration=best_it_5) if m5_use is not None else np.nan)

    # ---- Standard deviation for prob_up_5d ----
    std5 = np.full(len(last), np.nan, dtype=float)
    if prob_std_method == "residual" and oos5_path is not None and Path(oos5_path).exists():
        oos5_df = pd.read_csv(oos5_path)
        residual_map = _residual_std_5d(panel, oos5_df)
        std5 = last["symbol"].map(residual_map).astype(float).values
        # Fallback: symbol_hist then cross
        if np.isnan(std5).mean() > 0.5:
            sym_hist = _symbol_hist_roll_std(panel, window=int(prob_std_window), min_rows=int(prob_std_min_rows))
            std_map2 = sym_hist.to_dict()
            sh = last["symbol"].map(std_map2).astype(float).values
            std5 = np.where(np.isfinite(std5), std5, sh)
            if np.isnan(std5).mean() > 0.5:
                cs = _cross_sectional_std(last["pred_ret_5d_pct"].values)
                std5 = np.where(np.isfinite(std5), std5, cs)
    elif prob_std_method == "symbol_hist":
        sym_hist = _symbol_hist_roll_std(panel, window=int(prob_std_window), min_rows=int(prob_std_min_rows))
        std_map = sym_hist.to_dict()
        std5 = last["symbol"].map(std_map).astype(float).values
        if np.isnan(std5).mean() > 0.5:
            cs = _cross_sectional_std(last["pred_ret_5d_pct"].values)
            std5 = np.where(np.isfinite(std5), std5, cs)
    elif prob_std_method == "cross":
        cs = _cross_sectional_std(last["pred_ret_5d_pct"].values)
        std5[:] = cs
    else:  # "none"
        pass
    last["pred_std_5d"] = std5
    last["prob_up_5d"] = prob_up_from_gaussian(last["pred_ret_5d_pct"].values, std5)

    wl = last[(last["close"] >= MIN_CLOSE) & (last["avg20_vol"] >= MIN_AVG20_VOL)].copy()
    wl["bias"] = np.where(wl["long_score"] >= wl["short_score"], "LONG", "SHORT")

    # Ranking: prioritize calibrated 1D probability, then 5D expected return
    wl = wl[["symbol","timestamp","close","avg20_vol",
             "prob_up_1d","pred_ret_1d_pct","pred_ret_3d_pct","pred_ret_5d_pct",
             "pred_std_5d","prob_up_5d",
             "D_atr14","D_cpr_width_pct","long_score","short_score","bias"]].sort_values(
        ["prob_up_1d","pred_ret_5d_pct"], ascending=[False, False]
    )
    wl.to_csv(WATCHLIST_OUT, index=False)
    print(f"Saved: {WATCHLIST_OUT} rows={len(wl)}")
    return wl

# ===================== SHAP (latest rows only) =====================
def compute_shap_latest(panel: pd.DataFrame, feats: List[str], model, top_k: int = TOP_SHAP_PER_SYMBOL, shap_max_symbols: Optional[int] = None):
    try: import shap
    except Exception: raise SystemExit("shap not installed. Run: pip install shap")
    latest = panel.groupby("symbol", as_index=False).tail(1).copy()
    latest["avg20_vol"] = panel.groupby("symbol")["volume"].tail(1).values if "avg20_vol" not in latest \
        else latest["avg20_vol"]
    latest = latest[(latest["close"] >= MIN_CLOSE) & (latest["avg20_vol"].fillna(0) >= MIN_AVG20_VOL)]
    if shap_max_symbols is not None and shap_max_symbols > 0 and len(latest) > shap_max_symbols:
        latest = latest.sample(n=int(shap_max_symbols), random_state=42)
    X = sanitize_feature_matrix(latest[feats].copy())
    print("Computing SHAP values on latest rows...")
    import shap as _shap_mod
    explainer = _shap_mod.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list) and len(shap_vals) >= 2:
        shap_arr = shap_vals[1]  # positive class for classifier
    else:
        shap_arr = shap_vals
    base_val = explainer.expected_value
    try: base_val = float(np.ravel(base_val)[-1]) if isinstance(base_val, (list, tuple, np.ndarray)) else float(base_val)
    except Exception: base_val = 0.0
    rows, global_abs = [], defaultdict(list)
    eta = ProgressETA(total=len(latest), label="SHAP latest")
    for i, (_, row) in enumerate(latest.iterrows()):
        sym = row["symbol"]
        try:
            pred1 = float(model.predict_proba(X.iloc[[i]])[:,1][0])
        except Exception:
            pred1 = float(model.predict(X.iloc[[i]])[0])
        vals = shap_arr[i]
        idx = np.argsort(-np.abs(vals))[:top_k]
        for rank, j in enumerate(idx, start=1):
            rows.append({"symbol": sym, "timestamp": str(row["timestamp"]),
                         "feature": feats[j], "shap_value": float(vals[j]),
                         "abs_shap": float(abs(vals[j])), "rank": rank,
                         "prob_up_1d": pred1, "base_value": base_val})
        for j, v in enumerate(vals):
            global_abs[feats[j]].append(abs(float(v)))
        eta.tick(sym)
    shap_df = pd.DataFrame(rows).sort_values(["symbol","rank"])
    g_rows = [{"feature": f, "mean_abs_shap": float(np.mean(v))} for f, v in global_abs.items()]
    global_df = pd.DataFrame(g_rows).sort_values("mean_abs_shap", ascending=False)
    return shap_df, global_df

def build_shap_cards(shap_df: pd.DataFrame, top_k: int = 6):
    cards: Dict[str, dict] = {}
    for sym, g in shap_df.groupby("symbol"):
        g = g.sort_values("rank").head(top_k)
        prob = g["prob_up_1d"].iloc[0] if "prob_up_1d" in g else None
        base = g["base_value"].iloc[0] if "base_value" in g else None
        items = []
        for _, r in g.iterrows():
            items.append({"indicator": r["feature"],
                          "contribution_pct": round(float(r["shap_value"]), 6),
                          "abs_contribution": round(float(r["abs_shap"]), 6),
                          "rank": int(r["rank"])})
        cards[sym] = {"date": str(g["timestamp"].iloc[0]),
                      "prob_up_1d": None if prob is None else round(float(prob), 6),
                      "base_value": None if base is None else round(float(base), 6),
                      "top_contributors": items}
    return cards

# ===================== Extra outputs =====================
def write_parquet_single(panel: pd.DataFrame, out_dir: str):
    pq_path = Path(out_dir) / "panel_cache.parquet"
    try:
        panel.to_parquet(pq_path, index=False)
        print(f"Saved Parquet (single): {pq_path}")
    except Exception as e:
        print(f"Parquet write failed ({e}); skipping.")

def write_by_year_csv(panel: pd.DataFrame, out_dir: str):
    by_year_dir = Path(out_dir) / "panel_by_year"
    by_year_dir.mkdir(exist_ok=True)
    years = panel["timestamp"].dt.year.unique()
    for y in sorted(years):
        dfy = panel[panel["timestamp"].dt.year == y]
        outp = by_year_dir / f"panel_{y}.csv"
        dfy.to_csv(outp, index=False)
        print(f"Saved year CSV: {outp} (rows={len(dfy)})")

def write_excel_compact(panel: pd.DataFrame, wl: pd.DataFrame, out_dir: str):
    xlsx_path = Path(out_dir) / "panel_compact.xlsx"
    g = panel.groupby("symbol")
    summary = pd.DataFrame({"symbol": g.size().index, "rows": g.size().values,
                            "first_date": g["timestamp"].min().values,
                            "last_date": g["timestamp"].max().values,
                            "avg_close": g["close"].mean().values,
                            "avg_volume": g["volume"].mean().values})
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            wl.to_excel(writer, sheet_name="Latest", index=False)
            summary.to_excel(writer, sheet_name="SymbolsSummary", index=False)
        print(f"Saved Excel compact: {xlsx_path}")
    except Exception as e:
        print(f"Excel write failed ({e}); install openpyxl to enable --excel-compact")

def write_excel_topN(panel: pd.DataFrame, out_dir: str, n: int):
    n = min(max(1, int(n)), EXCEL_MAX)
    xlsx_path = Path(out_dir) / "panel_topN.xlsx"
    try:
        panel.head(n).to_excel(xlsx_path, sheet_name="TopNRows", index=False)
        print(f"Saved Excel top-N ({n} rows): {xlsx_path}")
    except Exception as e:
        print(f"Excel write failed ({e}); install openpyxl to enable --excel-top-rows")

# ===================== Checkpoints =====================
import joblib
def maybe_load_model(path: Path):
    try:
        if path.exists():
            print(f"Loading checkpoint: {path}")
            return joblib.load(path)
    except Exception as e:
        print(f"Failed to load checkpoint {path}: {e}")
    return None

def save_model(path: Path, model):
    try:
        joblib.dump(model, path)
        print(f"Saved model checkpoint: {path}")
    except Exception as e:
        print(f"Failed to save checkpoint {path}: {e}")

def save_oos(path: Path, oos: np.ndarray, valid_idx: np.ndarray):
    try:
        df = pd.DataFrame({"panel_idx": valid_idx, "oos_pred": oos})
        df.to_csv(path, index=False)
        print(f"Saved OOS predictions: {path} (rows={df.shape[0]})")
    except Exception as e:
        print(f"Failed to save OOS {path}: {e}")

def load_oos(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to load OOS {path}: {e}")
    return None

# ===================== OOS SUMMARY (Global + Playbooks compact) =====================
def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0: return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    margin = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    return (center - margin, center + margin)

def quarter_key(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("Q")

def hac_se_mean(x: np.ndarray, lag: int | str = "auto") -> float:
    x = np.asarray(pd.Series(x).dropna().values, dtype=float)
    n = len(x)
    if n <= 1: return np.inf
    m = x.mean(); e = x - m
    L = max(1, int(round(1.2 * (n ** (1/3))))) if isinstance(lag, str) and lag.lower()=="auto" else int(max(0, lag))
    gamma0 = np.dot(e, e) / n
    s = gamma0
    for k in range(1, min(L, n-1) + 1):
        w = 1 - k/(L+1)
        gamma_k = np.dot(e[:-k], e[k:]) / n
        s += 2 * w * gamma_k
    var_mean = s / n
    return math.sqrt(var_mean) if var_mean > 0 else np.inf

def one_sided_pvalue_mean_gt0(x: np.ndarray, lag="auto"):
    x = np.asarray(pd.Series(x).dropna().values, dtype=float)
    if len(x) == 0: return 1.0
    se = hac_se_mean(x, lag=lag)
    if not np.isfinite(se) or se == 0: return 1.0
    z = x.mean() / se
    return float(0.5 * (1 - math.erf(z / math.sqrt(2))))

def build_condition_columns(df: pd.DataFrame) -> OrderedDict:
    c = lambda name: pd.to_numeric(df.get(name, np.nan), errors="coerce")
    conds = OrderedDict()
    # CPR relations
    for k in ["CPR_Yday_Above","CPR_Yday_Below","CPR_Yday_Inside","CPR_Yday_Overlap"]:
        conds[k] = (df.get(k, 0) == 1)
    for k in ["CPR_Tmr_Above","CPR_Tmr_Below","CPR_Tmr_Inside","CPR_Tmr_Overlap"]:
        conds[k] = (df.get(k, 0) == 1)
    # Structure & Day type
    for k in ["Struct_uptrend","Struct_downtrend","Struct_range"]:
        conds[k] = (df.get(k, 0) == 1)
    for k in ["DayType_bullish","DayType_bearish","DayType_inside"]:
        conds[k] = (df.get(k, 0) == 1)
    # Trend & Momentum
    conds["EMA_stack_20_50_100"] = df.get("D_ema_stack_20_50_100", False).fillna(False).astype(bool)
    conds["RSI7_gt_RSI14"] = df.get("D_rsi7_gt_rsi14", False).fillna(False).astype(bool)
    conds["RSI14_50_70"] = c("D_rsi14").between(50,70)
    adx = c("D_adx14"); pdi = c("D_pdi14"); mdi = c("D_mdi14")
    conds["ADX_gt20_PDIgtMDI"] = ((adx > 20) & (pdi > mdi))
    eang = c("D_ema20_angle_deg")
    conds["EMA20_angle_pos"] = eang > 0
    # CPR width bands
    cprw = c("D_cpr_width_pct")
    conds["CPR_width_narrow"] = cprw < 0.5
    conds["CPR_width_wide"] = cprw > 1.0
    # OBV / PA
    conds["OBV_slope_up"] = c("D_obv_slope") > 0
    conds["Price_OBV_rising"] = df.get("D_price_and_obv_rising", False).fillna(False).astype(bool)
    # NR / ATR
    conds["NR_day"] = df.get("D_nr_day", False).fillna(False).astype(bool)
    conds["range_to_atr14_lt1"] = c("D_range_to_atr14") < 1.0
    # MA / Swing / Breakout
    conds["close_gt_sma20"] = c("D_sma20") < df["close"]
    conds["HH_or_HL"] = (df.get("D_hh", False).fillna(False).astype(bool) |
                         df.get("D_hl", False).fillna(False).astype(bool))
    conds["close_gt_prev_high"] = df["close"] > c("D_prev_high")
    # Regime
    conds["golden_regime"] = df.get("D_golden_regime", False).fillna(False).astype(bool)
    conds["daily_weekly_up"] = ((c("D_daily_trend")==1) & (c("D_weekly_trend")==1))
    for k in list(conds.keys()):
        s = conds[k]
        conds[k] = s.fillna(False).astype(bool) if s.dtype != bool else s.fillna(False)
    return conds

def non_overlapping_oos_mask(panel: pd.DataFrame, horizon: str, base_mask: pd.Series) -> pd.Series:
    H = {"1d":1, "3d":3, "5d":5}[horizon]
    df = panel.loc[base_mask, ["symbol","timestamp"]].copy().sort_values(["symbol","timestamp"])
    df["rnk"] = df.groupby("symbol").cumcount()
    keep_idx = df.index[(df["rnk"] % H) == 0]
    out = pd.Series(False, index=panel.index)
    out.loc[keep_idx] = True
    return out

def evaluate_conditions_oos(panel: pd.DataFrame, horizon: str, oos_df: pd.DataFrame,
                            conds: OrderedDict, min_support: int,
                            hac_lag=HAC_LAG, thin_inference=THIN_INFERENCE,
                            max_combo_size: int = MAX_COMBO_SIZE,
                            min_support_combo: int = MIN_SUPPORT_COMBO,
                            min_support_combo_4: int = MIN_SUPPORT_COMBO_4,
                            min_support_combo_5: int = MIN_SUPPORT_COMBO_5):
    oos_mask = pd.Series(False, index=panel.index)
    oos_mask.loc[oos_df["panel_idx"].values] = oos_df["oos_pred"].notna().values
    if thin_inference and horizon in ("3d","5d"):
        thin_mask = non_overlapping_oos_mask(panel, horizon, oos_mask)
        oos_mask = oos_mask & thin_mask
    ret_col = f"ret_{horizon}_close_pct"
    valid_rows = panel.index[oos_mask]
    if len(valid_rows) == 0:
        return pd.DataFrame(columns=["size","combo","support","hit_rate","mean_ret_%","wilson_low","wilson_high",
                                     "stability_qtr_std","p_value","bh_threshold","fdr_ok"])
    rets = panel.loc[valid_rows, ret_col].astype(float)
    ts = panel.loc[valid_rows, "timestamp"]
    cond_cols = OrderedDict()
    for name, series in conds.items():
        cond_cols[name] = series.loc[valid_rows].fillna(False).astype(bool)

    def metrics(flag: pd.Series):
        n = int(flag.sum())
        if n == 0: return (0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        r = rets[flag].values.astype(float)
        hit = (r > 0).sum()
        mean = float(np.mean(r))
        wl, wh = wilson_ci(int(hit), int(n), 1.96)
        qkey = quarter_key(ts[flag]); qmeans = pd.Series(r).groupby(qkey.values).mean()
        stability = float(qmeans.std()) if len(qmeans) >= 2 else 0.0
        p = one_sided_pvalue_mean_gt0(r, lag=hac_lag)
        return (n, hit/n, mean, wl, wh, p, stability)

    def min_support_for_size(size:int) -> int:
        if size >= 5: return max(min_support, min_support_combo, min_support_combo_5)
        elif size == 4: return max(min_support, min_support_combo, min_support_combo_4)
        elif size >= 2: return max(min_support, min_support_combo)
        else: return max(min_support, 0)

    names = list(cond_cols.keys())
    arrs = {k: cond_cols[k].values for k in names}
    rows_by_size: Dict[int, List[dict]] = {s: [] for s in range(1, max_combo_size+1)}
    support_by_combo: Dict[Tuple[str, ...], int] = {}

    # Singles
    for name, col in cond_cols.items():
        n, hr, mean, wl, wh, p, stab = metrics(col)
        support_by_combo[(name,)] = n
        if n >= min_support_for_size(1):
            rows_by_size[1].append({"size":1, "combo":name, "support":n, "hit_rate":hr, "mean_ret_%":mean,
                                    "wilson_low":wl, "wilson_high":wh, "stability_qtr_std":stab, "p_value":p})

    # Higher-order combos
    def add_combo(size:int, combo: Tuple[str, ...]):
        req = min_support_for_size(size)
        for sub in itertools.combinations(combo, size-1):
            sub_support = support_by_combo.get(tuple(sorted(sub)))
            if sub_support is None or sub_support < req: return
        flags = np.logical_and.reduce([arrs[c] for c in combo])
        mask_series = pd.Series(flags, index=valid_rows)
        n, hr, mean, wl, wh, p, stab = metrics(mask_series)
        support_by_combo[tuple(sorted(combo))] = n
        if n >= req:
            rows_by_size[size].append({"size":size, "combo":" & ".join(combo), "support":n, "hit_rate":hr,
                                       "mean_ret_%":mean, "wilson_low":wl, "wilson_high":wh,
                                       "stability_qtr_std":stab, "p_value":p})

    for size in range(2, max_combo_size+1):
        for combo in itertools.combinations(names, size):
            add_combo(size, combo)

    all_rows: List[dict] = []
    for s in range(1, max_combo_size+1):
        all_rows.extend(rows_by_size.get(s, []))
    if all_rows:
        dfw = pd.DataFrame(all_rows).sort_values("p_value").reset_index(drop=True)
        m = len(dfw)
        dfw["bh_threshold"] = (np.arange(1, m+1)/m) * FDR_ALPHA
        dfw["fdr_ok"] = dfw["p_value"] <= dfw["bh_threshold"]
        dfw = dfw.sort_values(["size","mean_ret_%","hit_rate","support"], ascending=[True, False, False, False])
        return dfw
    else:
        return pd.DataFrame(columns=["size","combo","support","hit_rate","mean_ret_%","wilson_low","wilson_high",
                                     "stability_qtr_std","p_value","bh_threshold","fdr_ok"])

def support_bar_for_size(size:int) -> int:
    if size >= 5: return max(MIN_SUPPORT_GLOBAL, MIN_SUPPORT_COMBO, MIN_SUPPORT_COMBO_5)
    elif size == 4: return max(MIN_SUPPORT_GLOBAL, MIN_SUPPORT_COMBO, MIN_SUPPORT_COMBO_4)
    elif size >= 2: return max(MIN_SUPPORT_GLOBAL, MIN_SUPPORT_COMBO)
    else: return max(MIN_SUPPORT_GLOBAL, 0)

def filter_approved_ww(ww: pd.DataFrame) -> pd.DataFrame:
    if ww is None or ww.empty: return pd.DataFrame()
    ww = ww.copy(); ww["size"] = ww["size"].astype(int)
    return ww[(ww["fdr_ok"]==True) & (ww["support"] >= ww["size"].map(support_bar_for_size))]

def latest_flags_for_conditions(panel: pd.DataFrame, conds: OrderedDict) -> pd.DataFrame:
    last_idx = panel.groupby("symbol").tail(1).index
    latest = panel.loc[last_idx, ["symbol"]].copy()
    for name, s in conds.items():
        latest[name] = s.loc[last_idx].astype(bool).values
    return latest.reset_index(drop=True)

def actionable_from_ww5d(panel: pd.DataFrame, wl: pd.DataFrame,
                         ww5: pd.DataFrame, conds: OrderedDict, out_csv: str):
    if ww5 is None or ww5.empty or wl is None or wl.empty: return
    ww5_ok = filter_approved_ww(ww5)
    if ww5_ok.empty:
        print("No approved 5D combos (after gates); actionable_5d not written.")
        return
    latest_flags = latest_flags_for_conditions(panel, conds)
    ww5_ok = ww5_ok.sort_values(["size","mean_ret_%","hit_rate","support"], ascending=[True, False, False, False])
    combo_list: List[Tuple[str, ...]] = []
    for _, r in ww5_ok.iterrows():
        combo = tuple([x.strip() for x in str(r["combo"]).split("&")])
        combo_list.append(combo)

    def combo_fires(row, combo: Tuple[str,...]) -> bool:
        for k in combo:
            if k not in row.index or not bool(row[k]): return False
        return True

    hits = []
    for _, r in latest_flags.iterrows():
        sym = r["symbol"]; hit_count = 0
        for combo in combo_list:
            if combo_fires(r, combo): hit_count += 1
        hits.append({"symbol": sym, "global_hits": hit_count})
    hits_df = pd.DataFrame(hits)
    out = wl.copy().merge(hits_df, on="symbol", how="left")
    out["global_hits"] = out["global_hits"].fillna(0).astype(int)
    out["actionable"] = out["global_hits"] > 0
    out = out.sort_values(["actionable","global_hits","pred_ret_5d_pct","prob_up_5d"], ascending=[False, False, False, False])
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} (rows={len(out)})")

# ===================== Recent accuracy reports =====================
def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2: return np.nan
    sx = np.nanstd(x); sy = np.nanstd(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0: return np.nan
    return float(np.corrcoef(x, y)[0,1])

def _recent_accuracy_for_horizon(panel: pd.DataFrame, oos_path: Path, horizon: str, last_days: int) -> dict:
    oos_df = load_oos(oos_path)
    if oos_df is None or oos_df.empty:
        return {"horizon": horizon, "window_days": last_days, "rows": 0,
                "hit_rate_%": np.nan, "mae_%": np.nan, "rmse_%": np.nan,
                "mean_pred_%": np.nan, "mean_real_%": np.nan, "pearson_r": np.nan,
                "wilson_low": np.nan, "wilson_high": np.nan}
    idx = oos_df["panel_idx"].values
    pred = pd.to_numeric(oos_df["oos_pred"], errors="coerce").values
    in_panel = np.isin(idx, panel.index.values)
    idx = idx[in_panel]; pred = pred[in_panel]
    max_ts = pd.to_datetime(panel["timestamp"]).max()
    if pd.isna(max_ts):
        return {"horizon": horizon, "window_days": last_days, "rows": 0,
                "hit_rate_%": np.nan, "mae_%": np.nan, "rmse_%": np.nan,
                "mean_pred_%": np.nan, "mean_real_%": np.nan, "pearson_r": np.nan,
                "wilson_low": np.nan, "wilson_high": np.nan}
    start_ts = max_ts - pd.Timedelta(days=int(last_days))
    ts = panel.loc[idx, "timestamp"].values
    in_window = (ts >= np.datetime64(start_ts)) & (ts <= np.datetime64(max_ts))
    idx = idx[in_window]; pred = pred[in_window]
    if len(idx) == 0:
        return {"horizon": horizon, "window_days": last_days, "rows": 0,
                "hit_rate_%": np.nan, "mae_%": np.nan, "rmse_%": np.nan,
                "mean_pred_%": np.nan, "mean_real_%": np.nan, "pearson_r": np.nan,
                "wilson_low": np.nan, "wilson_high": np.nan}
    ret_col = f"ret_{horizon}_close_pct"
    real = pd.to_numeric(panel.loc[idx, ret_col], errors="coerce").values
    valid = np.isfinite(pred) & np.isfinite(real)
    pred = pred[valid]; real = real[valid]
    if len(pred) == 0:
        return {"horizon": horizon, "window_days": last_days, "rows": 0,
                "hit_rate_%": np.nan, "mae_%": np.nan, "rmse_%": np.nan,
                "mean_pred_%": np.nan, "mean_real_%": np.nan, "pearson_r": np.nan,
                "wilson_low": np.nan, "wilson_high": np.nan}
    sign_pred = np.sign(pred); sign_real = np.sign(real)
    non_ties = (sign_pred != 0) & (sign_real != 0)
    hits = (sign_pred[non_ties] == sign_real[non_ties]).sum()
    total = int(non_ties.sum())
    hit_rate = (100.0 * hits / total) if total > 0 else np.nan
    abs_err = np.abs(real - pred)
    mae = float(np.mean(abs_err)) if len(abs_err) else np.nan
    rmse = float(np.sqrt(np.mean((real - pred)**2))) if len(abs_err) else np.nan
    mean_pred = float(np.mean(pred)) if len(pred) else np.nan
    mean_real = float(np.mean(real)) if len(real) else np.nan
    pearson_r = _safe_corr(pred, real)
    wl, wh = wilson_ci(hits, total, 1.96) if total > 0 else (np.nan, np.nan)
    return {"horizon": horizon, "window_days": last_days, "rows": int(len(pred)),
            "hit_rate_%": hit_rate, "mae_%": mae, "rmse_%": rmse,
            "mean_pred_%": mean_pred, "mean_real_%": mean_real,
            "pearson_r": pearson_r, "wilson_low": wl, "wilson_high": wh}

def write_recent_accuracy_report(panel: pd.DataFrame, oos1_path: Path, oos3_path: Path, oos5_path: Path,
                                 out_dir: str, last_days: int = 30) -> pd.DataFrame:
    rows = []
    rows.append(_recent_accuracy_for_horizon(panel, oos1_path, "1d", last_days))
    rows.append(_recent_accuracy_for_horizon(panel, oos3_path, "3d", last_days))
    rows.append(_recent_accuracy_for_horizon(panel, oos5_path, "5d", last_days))
    df = pd.DataFrame(rows)
    outp = Path(out_dir) / f"recent_accuracy_last_{int(last_days)}d.csv"
    df.to_csv(outp, index=False)
    print(f"Saved recent accuracy report: {outp}")
    return df

# ===================== Symbol-wise OOS accuracy =====================
def _symbol_accuracy_oos(panel: pd.DataFrame, oos_df: pd.DataFrame, horizon: str, thin_inference: bool = True) -> pd.DataFrame:
    if oos_df is None or oos_df.empty:
        return pd.DataFrame(columns=["symbol","horizon","rows","hit_rate_%","mae_%","rmse_%",
                                     "mean_pred_%","mean_real_%","pearson_r",
                                     "wilson_low","wilson_high","stability_qtr_std",
                                     "start_date","end_date"])
    oos_mask = pd.Series(False, index=panel.index)
    oos_mask.loc[oos_df["panel_idx"].values] = oos_df["oos_pred"].notna().values
    if thin_inference and horizon in ("3d","5d"):
        thin_mask = non_overlapping_oos_mask(panel, horizon, oos_mask)
        oos_mask = oos_mask & thin_mask
    idx_all = panel.index[oos_mask]
    preds_all = pd.to_numeric(oos_df.set_index("panel_idx").loc[idx_all, "oos_pred"], errors="coerce")
    rets_all = pd.to_numeric(panel.loc[idx_all, f"ret_{horizon}_close_pct"], errors="coerce")
    ts_all = panel.loc[idx_all, "timestamp"]
    sym_all = panel.loc[idx_all, "symbol"]
    rows = []
    for sym, grp_idx in pd.Series(idx_all, index=idx_all).groupby(sym_all.values):
        idx = grp_idx.values
        pred = preds_all.loc[idx].values
        real = rets_all.loc[idx].values
        ts = ts_all.loc[idx]
        valid = np.isfinite(pred) & np.isfinite(real)
        pred = pred[valid]; real = real[valid]; ts = ts[valid]
        if len(pred) == 0:
            rows.append({"symbol":sym, "horizon":horizon, "rows":0,
                         "hit_rate_%":np.nan,"mae_%":np.nan,"rmse_%":np.nan,
                         "mean_pred_%":np.nan,"mean_real_%":np.nan,"pearson_r":np.nan,
                         "wilson_low":np.nan,"wilson_high":np.nan,"stability_qtr_std":np.nan,
                         "start_date":None,"end_date":None})
            continue
        sign_pred = np.sign(pred); sign_real = np.sign(real)
        non_ties = (sign_pred != 0) & (sign_real != 0)
        hits = int((sign_pred[non_ties] == sign_real[non_ties]).sum())
        total = int(non_ties.sum())
        hit_rate = (100.0 * hits / total) if total > 0 else np.nan
        abs_err = np.abs(real - pred)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean((real - pred)**2)))
        mean_pred = float(np.mean(pred))
        mean_real = float(np.mean(real))
        pearson_r = _safe_corr(pred, real)
        wl, wh = wilson_ci(hits, total, 1.96) if total > 0 else (np.nan, np.nan)
        qkey = quarter_key(ts)
        qmeans = pd.Series(real).groupby(qkey.values).mean()
        stability = float(qmeans.std()) if len(qmeans) >= 2 else 0.0
        rows.append({"symbol":sym, "horizon":horizon, "rows":int(len(pred)),
                     "hit_rate_%":hit_rate,"mae_%":mae,"rmse_%":rmse,
                     "mean_pred_%":mean_pred,"mean_real_%":mean_real,"pearson_r":pearson_r,
                     "wilson_low":wl,"wilson_high":wh,"stability_qtr_std":stability,
                     "start_date":str(pd.to_datetime(ts.min()).date()),
                     "end_date":str(pd.to_datetime(ts.max()).date())})
    df = pd.DataFrame(rows).sort_values(["symbol","horizon"])
    return df

def write_symbol_accuracy_oos(panel: pd.DataFrame, oos1_path: Path, oos3_path: Path, oos5_path: Path,
                              out_dir: str, thin_inference: bool = True) -> pd.DataFrame:
    o1 = load_oos(oos1_path); o3 = load_oos(oos3_path); o5 = load_oos(oos5_path)
    df1 = _symbol_accuracy_oos(panel, o1, "1d", thin_inference)
    df3 = _symbol_accuracy_oos(panel, o3, "3d", thin_inference)
    df5 = _symbol_accuracy_oos(panel, o5, "5d", thin_inference)
    df = pd.concat([df1, df3, df5], ignore_index=True)
    outp = Path(out_dir) / "symbol_accuracy_oos.csv"
    df.to_csv(outp, index=False)
    print(f"Saved symbol-wise accuracy report: {outp} (rows={len(df)})")
    return df

# ===================== Combined summary =====================
def indicator_dictionary() -> pd.DataFrame:
    rows = [
        ("CPR_Yday_Above", "CPR", "Today’s CPR above yesterday’s CPR", "cpr_yday: Above"),
        ("CPR_Yday_Below", "CPR", "Today’s CPR below yesterday’s CPR", "cpr_yday: Below"),
        ("Struct_uptrend", "Structure", "Swing structure in uptrend", "HH & HL"),
        ("DayType_bullish", "DayType", "Open above CPR top", "open > CPR_tc"),
        ("EMA_stack_20_50_100", "Trend", "EMA20 > EMA50 > EMA100 (stacked up)", "EMA hierarchy"),
        ("RSI7_gt_RSI14", "Momentum", "RSI7 > RSI14", "short-term > mid-term"),
        ("EMA20_angle_pos", "Momentum", "EMA20 slope positive", "angle(deg) > 0"),
        ("OBV_slope_up", "Volume", "OBV slope rising", "ΔOBV > 0"),
        ("NR_day", "Volatility", "Narrow range day (NR)", "range < prev window min"),
        ("close_gt_sma20", "MA", "Close > SMA20", "close > SMA20"),
        ("HH_or_HL", "Swing", "Higher High or Higher Low", "HH | HL"),
        ("close_gt_prev_high", "Breakout", "Close > previous high", "close > prev_high"),
        ("golden_regime", "Regime", "Close > SMA200 and SMA50 > SMA200", "golden stack"),
        ("daily_weekly_up", "Regime", "Daily & Weekly trends up", "daily=+1 & weekly=+1"),
    ]
    return pd.DataFrame(rows, columns=["condition_key","group","description","definition"])

def write_combined_watchlist_summary(out_dir: str,
                                     wl_path: str,
                                     actionable_path: str,
                                     symbol_acc_path: str,
                                     playbooks_path: Optional[str] = None):
    wl = pd.read_csv(wl_path)
    actionable = pd.DataFrame()
    if actionable_path and Path(actionable_path).exists():
        actionable = pd.read_csv(actionable_path)
    symacc = pd.DataFrame()
    if symbol_acc_path and Path(symbol_acc_path).exists():
        symacc = pd.read_csv(symbol_acc_path)

    acc1d = symacc[symacc["horizon"]=="1d"].copy() if not symacc.empty else pd.DataFrame(columns=["symbol"])
    acc5d = symacc[symacc["horizon"]=="5d"].copy() if not symacc.empty else pd.DataFrame(columns=["symbol"])
    acc1d = acc1d[["symbol","hit_rate_%","mae_%","rmse_%","pearson_r"]].rename(
        columns={"hit_rate_%":"hit_rate_1d_%","mae_%":"mae_1d_%","rmse_%":"rmse_1d_%","pearson_r":"pearson_r_1d"})
    acc5d = acc5d[["symbol","hit_rate_%","mae_%","rmse_%","pearson_r"]].rename(
        columns={"hit_rate_%":"hit_rate_5d_%","mae_%":"mae_5d_%","rmse_%":"rmse_5d_%","pearson_r":"pearson_r_5d"})

    out = wl.copy()
    if not actionable.empty:
        cols_keep = ["symbol","global_hits","actionable"]
        inter = actionable[cols_keep].copy()
        out = out.merge(inter, on="symbol", how="left")
    else:
        out["global_hits"] = 0
        out["actionable"] = False

    out = out.merge(acc1d, on="symbol", how="left")
    out = out.merge(acc5d, on="symbol", how="left")

    dict_df = indicator_dictionary()
    out["notes"] = ""

    cols_order = [
        "symbol","timestamp","close","avg20_vol","bias",
        "prob_up_1d","pred_ret_1d_pct","pred_ret_3d_pct","pred_ret_5d_pct","prob_up_5d",
        "global_hits","actionable",
        "hit_rate_1d_%","mae_1d_%","rmse_1d_%","pearson_r_1d",
        "hit_rate_5d_%","mae_5d_%","rmse_5d_%","pearson_r_5d",
        "D_atr14","D_cpr_width_pct","long_score","short_score",
        "notes"
    ]
    for c in cols_order:
        if c not in out.columns:
            out[c] = np.nan if c not in ["actionable","notes"] else (False if c=="actionable" else "")
    out = out[cols_order]
    out_csv = Path(out_dir) / "combined_watchlist_summary.csv"
    out_xlsx = Path(out_dir) / "combined_watchlist_summary.xlsx"
    out.to_csv(out_csv, index=False)
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="Combined", index=False)
            dict_df.to_excel(writer, sheet_name="IndicatorDictionary", index=False)
        print(f"Saved: {out_csv} and {out_xlsx} (rows={len(out)})")
    except Exception as e:
        print(f"Excel write failed ({e}); CSV written: {out_csv}")
    return out

# ===================== MAIN =====================
def _maybe_delete_artifacts():
    try:
        p = Path(PANEL_OUT); q = Path(PROCESSED_LIST)
        if p.exists(): p.unlink()
        if q.exists(): q.unlink()
        print("Deleted previous panel_cache.csv and processed_symbols.txt (rebuild requested).")
    except Exception as e:
        print(f"Cleanup warning: {e}")

def resolve_input_paths(args) -> List[Path]:
    accept_any = str(args.accept_any_daily).lower() in ("true","1","yes","y","t")
    if args.files:
        paths = [Path(p) for p in args.files]
        return paths
    if args.gui:
        picked = pick_files_or_dir_gui()
        if not picked:
            print("GUI selection cancelled. Falling back to --data-dir.")
        else:
            # If a folder is picked, expand to files in that folder
            if len(picked) == 1 and picked[0].is_dir():
                return _strict_file_list(str(picked[0]), args.symbols_like, args.limit_files, accept_any_daily=accept_any)
            else:
                return [p for p in picked if p.is_file()]
    # Fallback: data-dir
    return _strict_file_list(args.data_dir, args.symbols_like, args.limit_files, accept_any_daily=accept_any)

def main():
    args = parse_cli()
    out_dir = args.out_dir
    setup_paths(out_dir)
    global FOLDS, EMBARGO_DAYS, HAC_LAG, THIN_INFERENCE
    global MAX_COMBO_SIZE, MIN_SUPPORT_GLOBAL, MIN_SUPPORT_COMBO, MIN_SUPPORT_COMBO_4, MIN_SUPPORT_COMBO_5
    global CHUNK_SIZE, SHAP_MAX_SYMBOLS
    global N_EST_1D, N_EST_3D, N_EST_5D, EARLY_STOPPING_ROUNDS
    global PROB_STD_METHOD, PROB_STD_WINDOW, PROB_STD_MIN_ROWS, CLS_MARGIN

    FOLDS = int(args.cv_splits)
    EMBARGO_DAYS = int(args.embargo_days)
    HAC_LAG = args.hac_lag
    THIN_INFERENCE = str(args.thin_inference).lower() in ("true","1","yes","y","t")
    MAX_COMBO_SIZE = int(args.max_combo_size)
    MIN_SUPPORT_GLOBAL = int(args.min_support_global)
    MIN_SUPPORT_COMBO = int(args.min_support_combo)
    MIN_SUPPORT_COMBO_4 = int(args.min_support_combo_4)
    MIN_SUPPORT_COMBO_5 = int(args.min_support_combo_5)
    CHUNK_SIZE = int(args.chunk_size)
    SHAP_MAX_SYMBOLS = args.shap_max_symbols
    N_EST_1D = int(args.n_estimators_1d)
    N_EST_3D = int(args.n_estimators_3d)
    N_EST_5D = int(args.n_estimators_5d)
    EARLY_STOPPING_ROUNDS = int(args.early_stopping_rounds)
    PROB_STD_METHOD = args.prob_std_method
    PROB_STD_WINDOW = int(args.prob_std_window)
    PROB_STD_MIN_ROWS = int(args.prob_std_min_rows)
    CLS_MARGIN = float(args.cls_margin)
    WRITE_SYMBOL_ACC = str(args.write_symbol_accuracy).lower() in ("true","1","yes","y","t")

    if args.rebuild: _maybe_delete_artifacts()

    print(f"Outputs dir: {Path(out_dir).resolve()}")
    t0 = time.perf_counter()

    # 0) Panel load/collect: prioritize panel_path, otherwise resume if PANEL_OUT exists
    write_status("panel_build", "starting")
    panel: pd.DataFrame
    feats: List[str]
    if args.panel_path and Path(args.panel_path).exists():
        print(f"Loading panel from --panel-path: {args.panel_path}")
        panel = pd.read_csv(args.panel_path, low_memory=False)
        panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
        panel = panel.dropna(subset=["timestamp"]).sort_values(["symbol","timestamp"]).reset_index(drop=True)
        feats = [c for c in panel.columns if (
                    c.startswith("D_") or c.startswith("CPR_Yday_") or c.startswith("CPR_Tmr_")
                    or c.startswith("Struct_") or c.startswith("DayType_"))]
    elif (not args.rebuild) and Path(PANEL_OUT).exists():
        print(f"Loading existing panel: {PANEL_OUT}")
        panel = pd.read_csv(PANEL_OUT, low_memory=False)
        panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
        panel = panel.dropna(subset=["timestamp"]).sort_values(["symbol","timestamp"]).reset_index(drop=True)
        feats = [c for c in panel.columns if (
                    c.startswith("D_") or c.startswith("CPR_Yday_") or c.startswith("CPR_Tmr_")
                    or c.startswith("Struct_") or c.startswith("DayType_"))]
    else:
        paths = resolve_input_paths(args)
        panel, feats = collect_panel_from_paths(paths)

    write_status("panel_build", "done")
    print(f"Panel rows: {len(panel)} symbols: {panel['symbol'].nunique()} feats: {len(feats)}")

    # Early writes (skip if loaded from external path)
    if not (args.panel_path and Path(args.panel_path).exists()):
        write_parquet_single(panel, out_dir)
        write_by_year_csv(panel, out_dir)

    # 2) Train / Checkpoints
    m1_reg_path = Path(out_dir) / "model_1d_reg.joblib"
    m1_cls_path = Path(out_dir) / "model_1d_cls_calib.joblib"
    m3_path = Path(out_dir) / "model_3d_rf.joblib"
    m5_path = Path(out_dir) / "model_5d_rf.joblib"
    oos1_reg_path = Path(out_dir) / "oos_preds_1d_reg.csv"
    oos1_cls_path = Path(out_dir) / "oos_probs_1d_cls.csv"
    oos3_path = Path(out_dir) / "oos_preds_3d.csv"
    oos5_path = Path(out_dir) / "oos_preds_5d.csv"

    train_mode = args.train_1d_mode  # "cls"|"reg"|"both"

    m1_reg = None; m1_cls = None
    # 1D CLS
    if train_mode in ("cls","both"):
        m1_cls = maybe_load_model(m1_cls_path)
        if m1_cls is None or not oos1_cls_path.exists():
            write_status("train_1d_cls", "starting")
            m1_cls, oos1_prob, idx1 = train_1d_cls_calibrated(panel, feats, CLS_MARGIN,
                                                               n_estimators=N_EST_1D,
                                                               n_splits=FOLDS, embargo_days=EMBARGO_DAYS,
                                                               early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            write_status("train_1d_cls", "done")
            save_model(m1_cls_path, m1_cls)
            save_oos(oos1_cls_path, oos1_prob, idx1)
        else:
            write_status("train_1d_cls", "skipped (checkpoint)")
    # 1D REG (optional)
    if train_mode in ("reg","both"):
        m1_reg = maybe_load_model(m1_reg_path)
        if m1_reg is None or not oos1_reg_path.exists():
            write_status("train_1d_reg", "starting")
            m1_reg, oos1_reg, idx1reg = train_rf(panel, feats, "ret_1d_close_pct", "1D-REG",
                                                 n_estimators=N_EST_1D, refit_final=False,
                                                 n_splits=FOLDS, embargo_days=EMBARGO_DAYS,
                                                 early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            write_status("train_1d_reg", "done")
            save_model(m1_reg_path, m1_reg); save_oos(oos1_reg_path, oos1_reg, idx1reg)
        else:
            write_status("train_1d_reg", "skipped (checkpoint)")

    # 3D
    m3 = maybe_load_model(m3_path)
    if m3 is None or not oos3_path.exists():
        write_status("train_3d", "starting")
        try:
            m3, oos3, idx3 = train_rf(panel, feats, "ret_3d_close_pct", "3D",
                                      n_estimators=N_EST_3D, refit_final=False,
                                      n_splits=FOLDS, embargo_days=EMBARGO_DAYS,
                                      early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            write_status("train_3d", "done")
            save_model(m3_path, m3); save_oos(oos3_path, oos3, idx3)
        except Exception as e:
            write_status("train_3d", f"failed: {e}")
            print(f"WARNING: 3D training failed ({e}). Proceeding without 3D.")
            m3 = None
    else:
        write_status("train_3d", "skipped (checkpoint)")

    # 5D
    m5 = maybe_load_model(m5_path)
    if m5 is None or not oos5_path.exists():
        write_status("train_5d", "starting")
        try:
            m5, oos5, idx5 = train_rf(panel, feats, "ret_5d_close_pct", "5D",
                                      n_estimators=N_EST_5D, refit_final=False,
                                      n_splits=FOLDS, embargo_days=EMBARGO_DAYS,
                                      early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            write_status("train_5d", "done")
            save_model(m5_path, m5); save_oos(oos5_path, oos5, idx5)
        except Exception as e:
            write_status("train_5d", f"failed: {e}")
            print(f"WARNING: 5D training failed ({e}). Proceeding without 5D.")
            m5 = None
    else:
        write_status("train_5d", "skipped (checkpoint)")

    # 3) Watchlist (latest)
    write_status("watchlist", "starting")
    wl = nightly_watchlist(panel, feats, m1_cls, m1_reg, m3, m5,
                           oos5_path=oos5_path,
                           prob_std_method=PROB_STD_METHOD,
                           prob_std_window=PROB_STD_WINDOW,
                           prob_std_min_rows=PROB_STD_MIN_ROWS)
    write_status("watchlist", "done")

    # 4) SHAP (1D latest) — prefer classifier if present; else regression 1D
    shap_df = None
    shap_model = m1_cls if m1_cls is not None else (m1_reg if m1_reg is not None else None)
    if "1d" in COMPUTE_SHAP_FOR and shap_model is not None:
        write_status("shap_1d", "starting")
        shap_df, shap_global = compute_shap_latest(panel, feats, shap_model, top_k=TOP_SHAP_PER_SYMBOL,
                                                   shap_max_symbols=SHAP_MAX_SYMBOLS)
        pd.DataFrame(shap_df).to_csv(SHAP_CARDS_CSV, index=False)
        pd.DataFrame(shap_global).to_csv(SHAP_GLOBAL_SUMMARY, index=False)
        cards = build_shap_cards(shap_df, top_k=6)
        with open(SHAP_CARDS_JSON, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2)
        write_status("shap_1d", "done")
        print(f"Saved: {SHAP_CARDS_CSV}, {SHAP_GLOBAL_SUMMARY}, {SHAP_CARDS_JSON}")

    # 5) What Worked (multi-horizon)
    try:
        write_status("what_works", "starting")
        conds = build_condition_columns(panel)
        ww1 = pd.DataFrame(); ww3 = pd.DataFrame(); ww5 = pd.DataFrame()
        oos1_use = oos1_reg_path if (train_mode in ("reg","both") and Path(oos1_reg_path).exists()) \
                   else (oos1_cls_path if Path(oos1_cls_path).exists() else None)
        if oos1_use is not None:
            ww1 = evaluate_conditions_oos(panel, "1d", pd.read_csv(oos1_use), conds, MIN_SUPPORT_GLOBAL,
                                          hac_lag=HAC_LAG, thin_inference=False,
                                          max_combo_size=MAX_COMBO_SIZE,
                                          min_support_combo=MIN_SUPPORT_COMBO,
                                          min_support_combo_4=MIN_SUPPORT_COMBO_4,
                                          min_support_combo_5=MIN_SUPPORT_COMBO_5)
            try:
                with pd.ExcelWriter(GLOBAL_WORKS_1D_XLSX, engine="openpyxl") as writer:
                    ww1.to_excel(writer, sheet_name="WhatWorks_1D", index=False)
                print(f"Saved: {GLOBAL_WORKS_1D_XLSX}")
            except Exception as e:
                ww1.to_csv(Path(GLOBAL_WORKS_1D_XLSX).with_suffix(".csv"), index=False)
                print(f"Excel write failed ({e}); wrote CSV fallback for 1D WhatWorked.")
        if Path(oos3_path).exists():
            ww3 = evaluate_conditions_oos(panel, "3d", pd.read_csv(oos3_path), conds, MIN_SUPPORT_GLOBAL,
                                          hac_lag=HAC_LAG, thin_inference=THIN_INFERENCE,
                                          max_combo_size=MAX_COMBO_SIZE,
                                          min_support_combo=MIN_SUPPORT_COMBO,
                                          min_support_combo_4=MIN_SUPPORT_COMBO_4,
                                          min_support_combo_5=MIN_SUPPORT_COMBO_5)
            try:
                with pd.ExcelWriter(GLOBAL_WORKS_3D_XLSX, engine="openpyxl") as writer:
                    ww3.to_excel(writer, sheet_name="WhatWorks_3D", index=False)
                print(f"Saved: {GLOBAL_WORKS_3D_XLSX}")
            except Exception as e:
                ww3.to_csv(Path(GLOBAL_WORKS_3D_XLSX).with_suffix(".csv"), index=False)
                print(f"Excel write failed ({e}); wrote CSV fallback for 3D WhatWorked.")
        if Path(oos5_path).exists():
            ww5 = evaluate_conditions_oos(panel, "5d", pd.read_csv(oos5_path), conds, MIN_SUPPORT_GLOBAL,
                                          hac_lag=HAC_LAG, thin_inference=THIN_INFERENCE,
                                          max_combo_size=MAX_COMBO_SIZE,
                                          min_support_combo=MIN_SUPPORT_COMBO,
                                          min_support_combo_4=MIN_SUPPORT_COMBO_4,
                                          min_support_combo_5=MIN_SUPPORT_COMBO_5)
            try:
                with pd.ExcelWriter(GLOBAL_WORKS_5D_XLSX, engine="openpyxl") as writer:
                    ww5.to_excel(writer, sheet_name="WhatWorks_5D", index=False)
                print(f"Saved: {GLOBAL_WORKS_5D_XLSX}")
            except Exception as e:
                ww5.to_csv(Path(GLOBAL_WORKS_5D_XLSX).with_suffix(".csv"), index=False)
                print(f"Excel write failed ({e}); wrote CSV fallback for 5D WhatWorked.")

        write_status("what_works", "done")

        # 6) Optional actionable overlay for 5D
        if WRITE_ACTIONABLE_5D and ww5 is not None and not ww5.empty and wl is not None and not wl.empty:
            actionable_from_ww5d(panel, wl, ww5, conds, ACTIONABLE_5D_OUT)
    except Exception as e:
        write_status("what_works", f"failed: {e}")
        print(f"WARNING: WhatWorks failed ({e})")

    # 7) Optional Excel exports
    if args.excel_compact: write_excel_compact(panel, wl, out_dir)
    if args.excel_top_rows: write_excel_topN(panel, out_dir, args.excel_top_rows)

    # 8) Recent accuracy (last N calendar days)
    try:
        write_recent_accuracy_report(panel,
                                     oos1_reg_path if Path(oos1_reg_path).exists() else oos1_cls_path,
                                     oos3_path, oos5_path, out_dir, last_days=args.last_month_days)
    except Exception as e:
        print(f"WARNING: Recent-accuracy report failed ({e})")

    # 9) Symbol-wise accuracy (full OOS)
    sym_acc_path = Path(out_dir) / "symbol_accuracy_oos.csv"
    if WRITE_SYMBOL_ACC:
        try:
            write_symbol_accuracy_oos(panel,
                                      oos1_reg_path if Path(oos1_reg_path).exists() else oos1_cls_path,
                                      oos3_path, oos5_path, out_dir, thin_inference=THIN_INFERENCE)
        except Exception as e:
            print(f"WARNING: Symbol-accuracy report failed ({e})")

    # 10) Combined summary
    try:
        write_combined_watchlist_summary(
            out_dir=out_dir,
            wl_path=WATCHLIST_OUT,
            actionable_path=ACTIONABLE_5D_OUT,
            symbol_acc_path=str(sym_acc_path),
            playbooks_path=SYMBOL_PLAYBOOKS_XLSX
        )
    except Exception as e:
        print(f"WARNING: Combined summary failed ({e})")

    dt_total = time.perf_counter() - t0
    m, s = divmod(int(dt_total), 60); h, m = divmod(m, 60)
    print(f"Total time: {h:02d}:{m:02d}:{s:02d}")
    write_status("done", "all outputs written")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — partial results saved (panel_cache, processed_symbols, model checkpoints). Re-run to resume.")
