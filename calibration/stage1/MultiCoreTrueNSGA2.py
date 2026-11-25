# -*- coding: utf-8 -*-
"""
GA_RRI_NSCAII_Flexible_Matching_v1.py — One-Land-Use GA for RRI with Parallel NSGA-II

Flex modes for pairing simulation with measurements:
- MATCH_MODE = "daily"   → pick one timestamp per day (nearest to target HH:MM:SS within tolerance)
- MATCH_MODE = "hourly"  → align hour-by-hour (tolerant rounding to top-of-hour if needed)
- MATCH_MODE = "satellite" → align simulation to irregular satellite timestamps (SWOT/ICESat)

Sources supported (enable/disable independently):
- Observed hourly water level (OBS)
- SWOT water level (irregular timestamps)
- ICESat water level (irregular timestamps)

Calibration driver (objectives) uses a single PRIMARY_SOURCE you choose, but the script also
computes/logs metrics for *all* enabled sources every evaluation for diagnostics.

Outputs:
- JSONL per-evaluation details (params, paths, matched pairs snapshot path, per-source metrics)
- TSV log (compact, per-evaluation, includes per-source metric columns)
- Archived hydro outputs and aligned pairs files per evaluation

Adjust the CONFIG section only.
"""

import os, re, shutil, subprocess, json, stat
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime
import time
import signal
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# =============================
# ========== CONFIG ===========
# =============================
ROOT_DIR      = Path(r"/mnt/c/users/mjgenoguin/Desktop/Dissertation/01_Objective1")
#ROOT_DIR      = Path(r"/home/WEM-st124985/Objective1")
PARTICLE_DIR  = ROOT_DIR / "Nan_5LULC"
RRI_INPUT     = PARTICLE_DIR / "RRI_Input.txt"
RRI_EXE_CANDS = ["RRI_Run"]  # name of RRI executable in working dir

# --- Data source toggles ---
USE_OBS     = True
USE_SWOT    = False
USE_ICESAT  = False

# Choose which source drives the GA objectives: "OBS" | "SWOT" | "ICESAT"
PRIMARY_SOURCE = "OBS"

# Mode for matching timestamps: "daily" | "hourly" | "satellite"
MATCH_MODE = "daily"

# --- Observed hourly (tab: datetime, value) ---
OBS_PATH       = ROOT_DIR / r"Observed/2023_Nan_N1_H.txt"
OBS_HAS_HEADER = True
OBS_SEP        = "\t"
OBS_DT_COL     = 0
OBS_VAL_COL    = 1
OBS_DT_FORMAT  = "%m/%d/%Y %H:%M:%S"  # adjust if needed

# --- SWOT (irregular timestamps) ---
SWOT_PATH        = ROOT_DIR / r"Observed/SWOT_WL.txt"  # datetime + value
SWOT_HAS_HEADER  = True
SWOT_SEP         = "\t"
SWOT_DT_COL      = 0
SWOT_VAL_COL     = 1
SWOT_DT_FORMAT   = "%Y-%m-%d %H:%M:%S"

# --- ICESat (irregular timestamps) ---
ICESAT_PATH        = ROOT_DIR / r"Observed/ICESat_WL.txt"
ICESAT_HAS_HEADER  = True
ICESAT_SEP         = "\t"
ICESAT_DT_COL      = 0
ICESAT_VAL_COL     = 1
ICESAT_DT_FORMAT   = "%Y-%m-%d %H:%M:%S"

# --- Archiving & logs ---
ARCHIVE_DIR   = PARTICLE_DIR / "Archive"
JSONL_LOG     = ARCHIVE_DIR / "GA_run_log_5LULC.jsonl"
TSV_LOG       = ROOT_DIR / "evaluation_log_5LULC.tsv"
CHECKPOINT_DIR  = ROOT_DIR / "Checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "ga_checkpoint_5LULC.pkl"

# --- Runtime budget (optional) ---
GA_MAX_SECONDS  = int(os.getenv("GA_MAX_SECONDS", "0"))   # 0 = unlimited
GRACE_SECONDS   = int(os.getenv("GA_GRACE_SECONDS", "300"))

# --- Checkpoint behavior ---
# If DELETE_CHECKPOINT_ON_SUCCESS = 1, checkpoint is removed at the end of a clean run.
# Default 0 keeps the checkpoint file as an archive of elite history.
DELETE_CHECKPOINT_ON_SUCCESS = bool(int(os.getenv("GA_DELETE_CKPT_ON_SUCCESS", "0")))

# --- GA hyperparameters ---
TARGET_LU     = 5
POP_SIZE      = 10
GENERATIONS   = 20
CX_PROB       = 0.6
MUT_PROB      = 0.2
MUT_SIGMA     = 0.05
RNG_SEED      = 42

# --- Seeding / warm-start behavior ---
# USE_SEEDING = 0 (default) → ignore vec0 and start from purely random population.
# USE_SEEDING = 1          → use vec0 as a seed: one exact copy, some locals around it,
#                             and the rest globally random within BOUNDS.
USE_SEEDING      = bool(int(os.getenv("GA_USE_SEEDING", "1")))
SEED_LOCAL_FRAC  = float(os.getenv("GA_SEED_LOCAL_FRAC", "0.7"))
SEED_LOCAL_SCALE = float(os.getenv("GA_SEED_LOCAL_SCALE", "0.05"))

PARALLEL_WORKERS = 3
WORKERS_ROOT     = PARTICLE_DIR.parent / "Workers_Nan"
CLEAN_HYDROS_BEFORE_RUN = True

# Lines to edit in RRI_Input.txt (1-based)
EDIT_LINE_RANGE = range(18, 37)
DO_NOT_TOUCH    = {19, 20, 24, 27, 31}
LOCKED_LUS      = None

CAL_PARAMS = [
    "ns_river", "ns_slope", "soildepth", "gammaa", "kv", "sf",
    "ka", "gammam", "beta", "ksg", "gammag", "kg0", "fg", "rgl"
]

PARAM_HINTS = {
    "ns_river":  ["ns_river"],
    "ns_slope":  ["ns_slope"],
    "soildepth": ["soildepth"],
    "gammaa":    ["gammaa"],
    "kv":        ["ksv"],
    "sf":        ["faif"],
    "ka":        ["ka"],
    "gammam":    ["gammam"],
    "beta":      ["beta"],
    "ksg":       ["kgv"],
    "gammag":    ["gammag"],
    "kg0":       ["tg"],
    "fg":        ["fpg"],
    "rgl":       ["init_cond_gw"],
}

# Bounds (widened as per your previous preference)
BOUNDS = {
    "ns_river":  (0.12, 0.26),
    "ns_slope":  (0.15, 0.90),
    "soildepth": (0.04,  0.80),
    "gammaa":    (0.38,  0.68),
    "gammam":    (0.00,  0.48),
    "kv":        (0.0,   0.10),
    "ka":        (0.0,   0.40),
    "sf":        (0.08,  0.38),
    "beta":      (0.0,   4.50),
    "ksg":       (0.0,   0.45),
    "gammag":    (0.20,  0.48),
    "kg0":       (0.00, 0.005),
    "fg":        (0.02,  0.15),
    "rgl":       (0.45,  1.00),
}

# === TIME AXIS & SCORING WINDOW ===
# SIM_START = reference for hydro_hr.txt first column (seconds since SIM_START)
SIM_START   = pd.Timestamp("2023-05-01 00:00")
SCORE_START = pd.Timestamp("2023-05-01 00:00")
SCORE_END   = pd.Timestamp("2023-10-31 00:00")

# === DAILY MATCH SETTINGS ===
DAILY_PICK_HOUR   = 0   # choose the hour of day yOu trust for daily comparison
DAILY_PICK_MINUTE = 0
DAILY_PICK_SECOND = 0
DAILY_TOLERANCE   = "120min"  # acceptable deviation around target time

# === HOURLY MATCH SETTINGS ===
HOURLY_TOLERANCE  = "5min"     # allowed jitter around exact top-of-hour

# === SATELLITE MATCH SETTINGS (per-source tolerances) ===
SAT_TOLERANCE_SIM_NEAR_SWOT   = "120min"
SAT_TOLERANCE_SIM_NEAR_ICESAT = "120min"

# =============================
# ========= HELPERS ===========
# =============================
FLOAT_TOKEN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?")
NUM_RE = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][+-]?\d+)?$")

def ensure_paths():
    for p in [PARTICLE_DIR, RRI_INPUT]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing: {p}")
    if USE_OBS and not Path(OBS_PATH).exists():
        raise FileNotFoundError(f"Missing OBS file: {OBS_PATH}")
    if USE_SWOT and not Path(SWOT_PATH).exists():
        raise FileNotFoundError(f"Missing SWOT file: {SWOT_PATH}")
    if USE_ICESAT and not Path(ICESAT_PATH).exists():
        raise FileNotFoundError(f"Missing ICESat file: {ICESAT_PATH}")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    WORKERS_ROOT.mkdir(parents=True, exist_ok=True)

def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

def write_lines_atomic(path: Path, lines: List[str]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, path)

def extract_name_from_comment(line: str) -> Optional[str]:
    if "#" not in line: return None
    tail = line.split("#", 1)[1].lower()
    for key, hints in PARAM_HINTS.items():
        if any(h in tail for h in hints):
            return key
    return None

def find_numeric_tokens_before_comment(line: str):
    code = line.split("#", 1)[0]
    toks = []
    for m in FLOAT_TOKEN.finditer(code):
        s, e = m.span()
        v = float(m.group(0).replace("D", "e").replace("d", "e"))
        toks.append((s, e, v))
    return toks, code

def replace_token_at_index(line: str, col_index_zero: int, new_val: float) -> str:
    tokens, code = find_numeric_tokens_before_comment(line)
    if not tokens: return line
    s_parts = []; last = 0
    for i, (a, b, _old) in enumerate(tokens):
        s_parts.append(code[last:a])
        if i == col_index_zero:
            s_parts.append(f"{new_val:.6g}")
        else:
            s_parts.append(code[a:b])
        last = b
    s_parts.append(code[last:])
    tail = "" if "#" not in line else "#" + line.split("#", 1)[1]
    return "".join(s_parts) + tail

def replace_first_numeric(line: str, new_val: float) -> str:
    tokens, code = find_numeric_tokens_before_comment(line)
    if not tokens: return line
    a, b, _ = tokens[0]
    s = code[:a] + f"{new_val:.6g}" + code[b:]
    tail = "" if "#" not in line else "#" + line.split("#", 1)[1]
    return s + tail

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def in_bounds(name: str, val: float) -> float:
    lo, hi = BOUNDS.get(name, (0.0, 1.0))
    return float(clamp(val, lo, hi))

def append_log(row: dict):
    with open(JSONL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

# =============================
# ===== File ↔ vector I/O =====
# =============================

def detect_params(lines: List[str]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for ln in EDIT_LINE_RANGE:
        if ln in DO_NOT_TOUCH or ln > len(lines): continue
        nm = extract_name_from_comment(lines[ln-1])
        names[ln] = nm
    return names

def read_vector_from_file(lines: List[str], names_by_line: Dict[int, str]) -> Tuple[List[str], np.ndarray, List[Tuple[int,int]]]:
    """
    Read a parameter vector from RRI_Input lines.

    For multi-LULC parameters (ns_slope, soildepth, ksv, ka, gammam, beta, kgv,
    gammag, tg, fpg, init_cond_gw, etc.), we create one GA gene per land-use
    column (Paddy..Water).

    For scalar parameters like ns_river (if not in DO_NOT_TOUCH), we only take
    the first numeric value.
    """
    param_order: List[str] = []
    vals: List[float] = []
    spans: List[Tuple[int, int]] = []

    for ln in EDIT_LINE_RANGE:
        if ln in DO_NOT_TOUCH or ln > len(lines):
            continue
        nm = names_by_line.get(ln)
        if not nm or nm not in CAL_PARAMS:
            continue

        line = lines[ln - 1]
        tokens, _ = find_numeric_tokens_before_comment(line)
        if not tokens:
            continue

        if nm == "ns_river":
            # Scalar parameter: first numeric only
            v = float(tokens[0][2])
            param_order.append(nm)
            vals.append(in_bounds(nm, v))
            spans.append((ln, -1))   # -1 means "no LU index / first numeric"
        else:
            # Multi-LULC parameter: one gene per column (Paddy..Water)
            for idx, (_a, _b, v) in enumerate(tokens):
                param_order.append(nm)
                vals.append(in_bounds(nm, float(v)))
                spans.append((ln, idx))  # idx = 0..(num_lu-1)

    return param_order, np.array(vals, dtype=float), spans

def _base_value_for(lines: List[str], target_name: str, target_lu: int) -> Optional[float]:
    for ln in EDIT_LINE_RANGE:
        if ln in DO_NOT_TOUCH or ln > len(lines): continue
        nm = extract_name_from_comment(lines[ln-1])
        if nm == target_name:
            tokens, _ = find_numeric_tokens_before_comment(lines[ln-1])
            idx = target_lu - 1
            if 0 <= idx < len(tokens):
                return float(tokens[idx][2])
    return None

def write_vector_to_lines(base_lines: List[str],
                          names: List[str],
                          vec: np.ndarray,
                          spans: List[Tuple[int, int]]) -> Tuple[List[str], List[float]]:
    """
    Write a GA vector back into a copy of RRI_Input lines.

    - Applies bounds per (parameter, LU).
    - Enforces gammam < gammaa per-LULC.
    - Enforces kv vs ka mutual exclusivity per-LULC.
    """
    out = list(base_lines)
    vals = np.array(vec, dtype=float).copy()

    # 1) Bounds per gene
    for i, name in enumerate(names):
        lo, hi = BOUNDS.get(name, (0.0, 1.0))
        vals[i] = clamp(vals[i], lo, hi)

    # 2) Build mapping: (param, LU) -> index in vals
    index_by_param_lu: Dict[Tuple[str, int], int] = {}
    max_lu = 0
    for i, (name, (ln, tok_idx)) in enumerate(zip(names, spans)):
        if tok_idx >= 0:
            lu = tok_idx + 1  # LU index: 1..num_LU
            index_by_param_lu[(name, lu)] = i
            if lu > max_lu:
                max_lu = lu

    # 3) Constraint: gammam < gammaa per-LULC
    if max_lu > 0:
        for lu in range(1, max_lu + 1):
            ia = index_by_param_lu.get(("gammaa", lu))
            im = index_by_param_lu.get(("gammam", lu))
            if ia is None or im is None:
                continue
            a = float(vals[ia])
            m = float(vals[im])
            if not (m < a):
                mid = 0.5 * (a + m)
                eps = max(1e-6, 0.01 * max(abs(a), abs(m), 1e-3))
                a_new = in_bounds("gammaa", mid + 0.5 * eps)
                m_new = in_bounds("gammam", mid - 0.5 * eps)
                vals[ia] = a_new
                vals[im] = m_new

    # 4) Constraint: kv vs ka mutual exclusivity per-LULC
    if max_lu > 0:
        for lu in range(1, max_lu + 1):
            ikv = index_by_param_lu.get(("kv", lu))
            ika = index_by_param_lu.get(("ka", lu))

            # Base kv from template, in case this LU is not in GA vector
            base_kv = _base_value_for(base_lines, "kv", lu)
            kv_val = float(vals[ikv]) if ikv is not None else (base_kv if base_kv is not None else 0.0)
            ka_val = float(vals[ika]) if ika is not None else 0.0

            if (kv_val > 0.0) and (ka_val > 0.0):
                if ikv is not None and ika is not None:
                    # Keep the larger, zero the smaller
                    if kv_val >= ka_val:
                        vals[ika] = 0.0
                    else:
                        vals[ikv] = 0.0
                else:
                    # If kv is positive in template and only ka is in GA: force ka = 0
                    if base_kv is not None and base_kv > 0.0 and ika is not None:
                        vals[ika] = 0.0

    # 5) Write back to text lines
    used_vals: List[float] = []
    for (ln, tok_idx), name, newv in zip(spans, names, vals):
        newv = float(newv)
        if ln > len(out):
            used_vals.append(newv)
            continue
        line = out[ln - 1]
        if tok_idx == -1:
            # Scalar (e.g., ns_river): first numeric on the line
            out[ln - 1] = replace_first_numeric(line, newv)
        else:
            # Multi-LULC: specific column on that line
            out[ln - 1] = replace_token_at_index(line, tok_idx, newv)
        used_vals.append(newv)

    return out, used_vals


# =============================
# ======== RRI Runner =========
# =============================

def _file_type(p: Path) -> str:
    try:
        out = subprocess.run(["/usr/bin/file", str(p)], capture_output=True, text=True)
        return out.stdout.strip() if out.stdout else ""
    except Exception:
        return ""

def _cleanup_hydros(cwd: Path):
    if CLEAN_HYDROS_BEFORE_RUN:
        for p in cwd.glob("**/hydro*.txt"):
            try: p.unlink()
            except Exception: pass

def run_rri_with_log(cwd: Path, run_log: Path):
    with open(run_log, "w", encoding="utf-8") as rlog:
        def w(msg): rlog.write(msg + "\n")
        w(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] START in {cwd}")
        exe = None
        for cand in RRI_EXE_CANDS:
            p = cwd / cand
            if p.exists():
                exe = p; break
        if exe is None:
            w(f"[error] RRI executable not found ({', '.join(RRI_EXE_CANDS)})")
            raise FileNotFoundError("RRI_Run not found")
        try:
            if exe.suffix != ".exe":
                mode = exe.stat().st_mode
                if not bool(mode & stat.S_IXUSR):
                    exe.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    w("[execbit] chmod +x applied")
        except Exception as e:
            w(f"[warn] chmod failed: {e}")
        ft = _file_type(exe)
        if ft: w(f"[file] {ft}")
        _cleanup_hydros(cwd)
        w(f"[run] Launching: {exe}")
        proc = subprocess.Popen([str(exe)], cwd=str(cwd),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            rlog.write(line)
        code = proc.wait()
        w(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] END (code {code})")
        if code != 0:
            raise subprocess.CalledProcessError(code, [str(exe)])
        hydros = list(cwd.glob("**/hydro*.txt"))
        if not hydros:
            w("[error] Finished but no hydro*.txt produced")
            raise RuntimeError("No hydro*.txt produced")

# =============================
# ======= I/O: series =========
# =============================

def _read_two_col_series(path: Path, sep: str, has_header: bool, dt_col: int, val_col: int, dt_format: Optional[str]) -> pd.Series:
    df = pd.read_csv(path, sep=sep, header=0 if has_header else None)
    if df.shape[1] <= max(dt_col, val_col):
        raise ValueError(f"{path} must have datetime and value columns")
    dt_raw = df.iloc[:, dt_col].astype(str).str.strip()
    val    = pd.to_numeric(df.iloc[:, val_col], errors="coerce")
    dt = pd.to_datetime(dt_raw, format=dt_format, errors="coerce") if dt_format else pd.to_datetime(dt_raw, errors="coerce")
    s = pd.Series(val.values, index=dt).sort_index()
    return s[s.index.notna()]

def load_observed_series() -> Optional[pd.Series]:
    if not USE_OBS: return None
    return _read_two_col_series(OBS_PATH, OBS_SEP, OBS_HAS_HEADER, OBS_DT_COL, OBS_VAL_COL, OBS_DT_FORMAT)

def load_swot_series() -> Optional[pd.Series]:
    if not USE_SWOT: return None
    return _read_two_col_series(SWOT_PATH, SWOT_SEP, SWOT_HAS_HEADER, SWOT_DT_COL, SWOT_VAL_COL, SWOT_DT_FORMAT)

def load_icesat_series() -> Optional[pd.Series]:
    if not USE_ICESAT: return None
    return _read_two_col_series(ICESAT_PATH, ICESAT_SEP, ICESAT_HAS_HEADER, ICESAT_DT_COL, ICESAT_VAL_COL, ICESAT_DT_FORMAT)

def load_hydro_with_time(path: Path, sim_start: pd.Timestamp) -> pd.Series:
    t_sec, sim1 = [], []
    if path.exists():
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 2:
                    try:
                        t = float(p[0]); v = float(p[1])
                        t_sec.append(t); sim1.append(v)
                    except Exception:
                        continue
    if not t_sec:
        return pd.Series(dtype=float)
    idx = sim_start + pd.to_timedelta(np.array(t_sec, dtype=float), unit="s")
    return pd.Series(np.array(sim1, dtype=float), index=pd.to_datetime(idx)).sort_index()

# =============================
# ======= Match helpers =======
# =============================

def pick_daily_time_tolerant(s: pd.Series, hour: int, minute: int, second: int, tolerance: str) -> pd.Series:
    """
    Pick a single value per day, closest to target HH:MM:SS within ±tolerance.
    Uses positional masks to avoid index-alignment errors.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)

    # Ensure DatetimeIndex and sorted
    s = s.sort_index()
    idx = pd.DatetimeIndex(s.index)

    # Build per-row target times (same year/month/day, chosen HH:MM:SS)
    targets = pd.to_datetime({
        "year":  idx.year,
        "month": idx.month,
        "day":   idx.day,
        "hour":  hour,
        "minute": minute,
        "second": second
    })

    # Compute ± tolerance window (POSitional mask)
    tol = pd.Timedelta(tolerance)
    delta = (idx - targets)  # TimedeltaIndex
    delta_arr = delta.to_numpy()  # 1D ndarray
    mask_arr = (delta_arr >= -tol) & (delta_arr <= tol)

    # Filter by position, then collapse duplicates per target (per-day) by taking first
    s_f = s.iloc[mask_arr]
    gkey = pd.DatetimeIndex(targets.values[mask_arr])
    # groupby expects same length; we group by the target timestamp of each kept row
    return s_f.groupby(gkey).first().sort_index()


def pick_hourly_tolerant(s: pd.Series, tolerance: str) -> pd.Series:
    """
    Snap to nearest hour within ±tolerance, using positional masks to avoid index alignment issues.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)

    s = s.sort_index()
    idx = pd.DatetimeIndex(s.index)
    rounded = idx.round("h")

    tol = pd.Timedelta(tolerance)
    delta = (idx - rounded)
    delta_arr = delta.to_numpy()
    mask_arr = (delta_arr >= -tol) & (delta_arr <= tol)

    s_f = s.iloc[mask_arr]
    # Collapse to one per hour by the rounded timestamp used as grouping key
    gkey = pd.DatetimeIndex(rounded.values[mask_arr])
    return s_f.groupby(gkey).first().sort_index()


def match_sim_to_satellite(sim: pd.Series, sat: pd.Series, tolerance: str) -> pd.DataFrame:
    """Return DataFrame with index as sat timestamps; columns: obs, sim (nearest within tol)."""
    if sim is None or sim.empty or sat is None or sat.empty:
        return pd.DataFrame(columns=["obs","sim"]).astype({"obs": float, "sim": float})
    tol = pd.Timedelta(tolerance)
    sim = sim.sort_index(); sat = sat.sort_index()
    # For each sat time, find nearest sim time within tol
    # Use pandas merge_asof for efficient nearest join
    df_sat = sat.rename("obs").to_frame()
    df_sim = sim.rename("sim").to_frame()
    out = pd.merge_asof(df_sat.reset_index().rename(columns={"index":"t"}),
                        df_sim.reset_index().rename(columns={"index":"t"}),
                        on="t", direction="nearest", tolerance=tol)
    out = out.dropna(subset=["sim"]).set_index("t")[["obs","sim"]]
    return out.sort_index()

# =============================
# ========= METRICS ===========
# =============================

def _finite_align(o: np.ndarray, s: np.ndarray):
    mask = np.isfinite(o) & np.isfinite(s)
    return o[mask], s[mask], mask.sum()

def _nse(o: np.ndarray, s: np.ndarray) -> float:
    o, s, n = _finite_align(o, s)
    if n < 2: return -1e9
    denom = float(np.sum((o - np.nanmean(o))**2))
    return 1.0 - float(np.nansum((o - s)**2))/denom if denom > 0 else -1e9

def _rmse(o: np.ndarray, s: np.ndarray) -> float:
    o, s, n = _finite_align(o, s)
    if n == 0: return 1e9
    return float(np.sqrt(np.nanmean((o - s)**2)))

def _mae(o: np.ndarray, s: np.ndarray) -> float:
    o, s, n = _finite_align(o, s)
    if n == 0: return 1e9
    return float(np.nanmean(np.abs(o - s)))

def _bias(o: np.ndarray, s: np.ndarray) -> float:
    o, s, n = _finite_align(o, s)
    if n == 0: return 1e9
    return float(np.nanmean(s - o))

def _corr(o: np.ndarray, s: np.ndarray) -> float:
    o, s, n = _finite_align(o, s)
    if n < 2: return float("nan")
    if np.nanstd(o) == 0 or np.nanstd(s) == 0:
        return float("nan")
    return float(np.corrcoef(o, s)[0, 1])

def _kge(o: np.ndarray, s: np.ndarray) -> float:
    o, s, n = _finite_align(o, s)
    if n < 2: return -1e9
    mu_o, mu_s = np.nanmean(o), np.nanmean(s)
    sd_o, sd_s = np.nanstd(o),  np.nanstd(s)
    if sd_o == 0 or np.isnan(sd_o) or np.isnan(sd_s) or mu_o == 0 or np.isnan(mu_o) or np.isnan(mu_s):
        return -1e9
    r = _corr(o, s)
    if np.isnan(r): return -1e9
    alpha = sd_s / sd_o
    beta  = mu_s / mu_o
    return float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))

# compute metrics for a dataframe with columns obs, sim

def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"NSE": -1e9, "RMSE": 1e9, "MAE": 1e9, "Bias": 1e9, "KGE": -1e9, "Corr": float("nan"), "n_pairs": 0}
    o = df["obs"].values
    s = df["sim"].values
    return {
        "NSE":  _nse(o, s),
        "RMSE": _rmse(o, s),
        "MAE":  _mae(o, s),
        "Bias": _bias(o, s),
        "KGE":  _kge(o, s),
        "Corr": _corr(o, s),
        "n_pairs": int(len(df))
    }

# =============================
# ====== WORKER INIT ==========
# =============================
_WORKDIR = None

def _init_worker(base_particle_dir: str, workers_root: str):
    global _WORKDIR, PARTICLE_DIR, RRI_INPUT
    pid = os.getpid()
    base = Path(base_particle_dir)
    root = Path(workers_root)
    root.mkdir(parents=True, exist_ok=True)
    workdir = root / f"Worker_p{pid}"
    if not workdir.exists():
        def _ignore(dirpath, names):
            ignored = set()
            if Path(dirpath).name.lower() == "archive":
                ignored = set(names)
            return ignored
        shutil.copytree(base, workdir, ignore=_ignore)
    _WORKDIR = workdir
    PARTICLE_DIR = workdir
    RRI_INPUT = workdir / "RRI_Input.txt"

# =============================
# ===== EVALUATE (WORKER) =====
# =============================

def _safe_copy(src: Path, dst: Path) -> Optional[str]:
    try:
        shutil.copy2(src, dst)
        return str(dst)
    except Exception:
        return None

def _archive_hydros(cwd: Path, archive_dir: Path, gen: int, ind: int, stamp: str) -> Dict[str, Optional[str]]:
    archived = {"hydro_hr": None, "hydro": None}
    candidates = list(cwd.glob("**/hydro*.txt"))
    def pick_exact(name):
        for p in candidates:
            if p.name.lower() == name:
                return p
        return None
    picks = {"hydro_hr": pick_exact("hydro_hr.txt"),
             "hydro":    pick_exact("hydro.txt")}
    if picks["hydro_hr"] is None:
        hits = sorted(cwd.glob("**/hydro_hr*.txt"))
        if hits: picks["hydro_hr"] = hits[0]
    for key, src in picks.items():
        if src and src.exists():
            dst = archive_dir / f"{key}_gen{gen:03d}_ind{ind:03d}_{stamp}.txt"
            archived[key] = _safe_copy(src, dst)
    return archived

def _build_pairs(sim_full: pd.Series) -> Dict[str, Dict[str, object]]:
    """
    Return per-source dict: {SRC: {df, metrics, aligned_path}}
    df has columns [obs, sim].
    """
    out: Dict[str, Dict[str, object]] = {}

    # Helper to window & save
    def finalize(src_name: str, df_pairs: pd.DataFrame) -> Dict[str, object]:
        if df_pairs is None:
            df_pairs = pd.DataFrame(columns=["obs","sim"]) 
        # Score window
        df_win = df_pairs.loc[(df_pairs.index >= SCORE_START) & (df_pairs.index <= SCORE_END)]
        metrics = compute_metrics(df_win)
        return {"df": df_win, "metrics": metrics}

    # Build per-source pairs according to MATCH_MODE
    if USE_OBS:
        obs = load_observed_series()
        if MATCH_MODE == "daily":
            obs_d = pick_daily_time_tolerant(obs, DAILY_PICK_HOUR, DAILY_PICK_MINUTE, DAILY_PICK_SECOND, DAILY_TOLERANCE)
            sim_d = pick_daily_time_tolerant(sim_full, DAILY_PICK_HOUR, DAILY_PICK_MINUTE, DAILY_PICK_SECOND, DAILY_TOLERANCE)
            df = pd.concat({"obs": obs_d, "sim": sim_d}, axis=1).dropna()
        elif MATCH_MODE == "hourly":
            obs_h = pick_hourly_tolerant(obs, HOURLY_TOLERANCE)
            sim_h = pick_hourly_tolerant(sim_full, HOURLY_TOLERANCE)
            df = pd.concat({"obs": obs_h, "sim": sim_h}, axis=1).dropna()
        elif MATCH_MODE == "satellite":
            # In satellite mode, OBS is not used for primary matching, but we can still compute hourly-like if desired.
            # Here, we still try to match hourly tolerant (diagnostic only).
            obs_h = pick_hourly_tolerant(obs, HOURLY_TOLERANCE)
            sim_h = pick_hourly_tolerant(sim_full, HOURLY_TOLERANCE)
            df = pd.concat({"obs": obs_h, "sim": sim_h}, axis=1).dropna()
        else:
            df = pd.DataFrame(columns=["obs","sim"])
        out["OBS"] = finalize("OBS", df)

    if USE_SWOT:
        swot = load_swot_series()
        df = match_sim_to_satellite(sim_full, swot, SAT_TOLERANCE_SIM_NEAR_SWOT)
        out["SWOT"] = finalize("SWOT", df)

    if USE_ICESAT:
        icesat = load_icesat_series()
        df = match_sim_to_satellite(sim_full, icesat, SAT_TOLERANCE_SIM_NEAR_ICESAT)
        out["ICESAT"] = finalize("ICESAT", df)

    return out

def evaluate_worker(vec: np.ndarray, names: List[str], template_lines: List[str], spans, gen: int, idx: int):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    edited_path = PARTICLE_DIR / f"RRI_Input_eval_gen{gen:03d}_ind{idx:03d}_{stamp}.txt"
    rri_log     = ARCHIVE_DIR / f"RRI_run_gen{gen:03d}_ind{idx:03d}_{stamp}.log"

    # Edit RRI_Input.txt
    original = read_lines(RRI_INPUT)
    new_lines, used_vals = write_vector_to_lines(original, names, vec, spans)

    snap = ARCHIVE_DIR / f"RRI_Input_EDIT_gen{gen:03d}_ind{idx:03d}_{stamp}.txt"
    try:
        with open(snap, "w", encoding="utf-8") as f:
            f.write("=== BEFORE (worker RRI_Input.txt) ===\n")
            f.writelines(original)
            f.write("\n=== AFTER (edited for this run) ===\n")
            f.writelines(new_lines)
    except Exception:
        pass

    # Run RRI
    try:
        write_lines_atomic(RRI_INPUT, new_lines)
        run_rri_with_log(PARTICLE_DIR, rri_log)
    except Exception as e:
        write_lines_atomic(RRI_INPUT, original)
        row = {
            "gen": gen, "ind": idx, "timestamp": stamp,
            "error": f"{e}",
            "edited_input_snapshot": str(snap),
            "rri_run_log": str(rri_log),
            "particle_dir": str(PARTICLE_DIR)
        }
        return (-1e9, 1e9, 1e9, row)
    finally:
        try:
            with open(edited_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception:
            pass
        write_lines_atomic(RRI_INPUT, original)

    # Read outputs & build pairs
    hydro_candidates = sorted(PARTICLE_DIR.glob("**/hydro*.txt"))
    hydro_used = None
    sim_full = pd.Series(dtype=float)

    archived_hydros = _archive_hydros(PARTICLE_DIR, ARCHIVE_DIR, gen, idx, stamp)

    if hydro_candidates:
        hr = [p for p in hydro_candidates if p.name.lower() == "hydro_hr.txt"]
        hydro_used = hr[0] if hr else hydro_candidates[0]
        sim_full = load_hydro_with_time(hydro_used, SIM_START)

    per_source = _build_pairs(sim_full)

    # Pick primary metrics for GA objectives
    if PRIMARY_SOURCE not in per_source:
        # No data -> penalize
        nse = -1e9; rmse = 1e9; bias = 1e9
        primary_metrics = {"NSE": nse, "RMSE": rmse, "Bias": bias}
        primary_pairs_path = None
    else:
        primary = per_source[PRIMARY_SOURCE]
        met = primary["metrics"]
        nse, rmse, bias = met["NSE"], met["RMSE"], met["Bias"]
        # Save aligned pairs
        pairs_path = ARCHIVE_DIR / f"pairs_{PRIMARY_SOURCE}_gen{gen:03d}_ind{idx:03d}_{stamp}.tsv"
        try:
            primary["df"].to_csv(pairs_path, sep="\t", float_format="%.6f", date_format="%Y-%m-%d %H:%M:%S")
            primary_pairs_path = str(pairs_path)
        except Exception:
            primary_pairs_path = None
        primary_metrics = met

    # Also write per-source pairs (diagnostics)
    diag_paths = {}
    for src_name, bundle in per_source.items():
        if src_name == PRIMARY_SOURCE: 
            diag_paths[src_name] = primary_pairs_path
            continue
        p = ARCHIVE_DIR / f"pairs_{src_name}_gen{gen:03d}_ind{idx:03d}_{stamp}.tsv"
        try:
            bundle["df"].to_csv(p, sep="\t", float_format="%.6f", date_format="%Y-%m-%d %H:%M:%S")
            diag_paths[src_name] = str(p)
        except Exception:
            diag_paths[src_name] = None

    # Flatten params for JSONL with correct LU index
    flat_params = []
    for (name, (ln, tok_idx)), val in zip(zip(names, spans), used_vals):
        # LU index: 0 for scalar (e.g., ns_river), or 1..num_LU based on column
        lu = 0 if tok_idx < 0 else (tok_idx + 1)
        flat_params.append((name, lu, float(val)))

    # Compose JSON line
    row = {
        "gen": gen, "ind": idx, "timestamp": stamp,
        "PRIMARY_SOURCE": PRIMARY_SOURCE,
        "MATCH_MODE": MATCH_MODE,
        "primary_metrics": primary_metrics,
        "per_source_metrics": {k: v["metrics"] for k, v in per_source.items()},
        "params": flat_params,
        "edited_input_snapshot": str(snap),
        "edited_input_copy": str(edited_path),
        "rri_run_log": str(rri_log),
        "hydro_used": str(hydro_used) if hydro_used else None,
        "archived_hydros": archived_hydros,
        "pairs_paths": {**diag_paths, PRIMARY_SOURCE: primary_pairs_path},
        "particle_dir": str(PARTICLE_DIR)
    }

    return float(nse), float(rmse), float(bias), row

# =============================
# ======= GA PRIMITIVES =======
# =============================

def init_population(n_ind: int, names: List[str], vec0: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """
    Create initial population.

    - If USE_SEEDING == False (default): purely random within BOUNDS (original behavior).
    - If USE_SEEDING == True and vec0 is provided:
        * 1 exact copy of vec0 (clipped to BOUNDS),
        * SEED_LOCAL_FRAC of the remaining individuals sampled as Gaussian
          perturbations around vec0 (scale = SEED_LOCAL_SCALE * (hi - lo)),
        * the rest sampled uniformly within BOUNDS.
    """
    lo = np.array([BOUNDS[n][0] for n in names], dtype=float)
    hi = np.array([BOUNDS[n][1] for n in names], dtype=float)
    dim = len(names)

    # Default: original fully random behavior
    if (not USE_SEEDING) or (vec0 is None) or (len(vec0) != dim):
        return [lo + np.random.rand(dim) * (hi - lo) for _ in range(n_ind)]

    pop: List[np.ndarray] = []

    # 1) Exact seed individual
    seed = np.clip(np.array(vec0, dtype=float), lo, hi)
    pop.append(seed)

    # Remaining individuals
    remaining = max(0, n_ind - 1)
    n_local = int(round(remaining * SEED_LOCAL_FRAC))

    # 2) Local perturbations around seed
    if n_local > 0:
        scale = SEED_LOCAL_SCALE * (hi - lo)
        for _ in range(n_local):
            noise = np.random.normal(loc=0.0, scale=1.0, size=dim)
            ind = seed + noise * scale
            ind = np.clip(ind, lo, hi)
            pop.append(ind)

    # 3) Global randoms for exploration
    while len(pop) < n_ind:
        ind = lo + np.random.rand(dim) * (hi - lo)
        pop.append(ind)

    return pop

def uniform_crossover(a: np.ndarray, b: np.ndarray, names: List[str], p: float = 0.5) -> np.ndarray:
    child = a.copy()
    mask = (np.random.rand(len(a)) < p)
    child[mask] = b[mask]
    for i, n in enumerate(names):
        lo, hi = BOUNDS[n]
        child[i] = np.clip(child[i], lo, hi)
    return child

def gaussian_mutation(x: np.ndarray, names: List[str], indpb: float = MUT_PROB, sigma: float = MUT_SIGMA) -> np.ndarray:
    y = x.copy()
    for i, n in enumerate(names):
        if np.random.rand() < indpb:
            lo, hi = BOUNDS[n]
            span = (hi - lo)
            y[i] += np.random.normal(0, sigma) * span
            y[i] = np.clip(y[i], lo, hi)
    return y

# =============================
# === NSGA-II utilities =======
# =============================

def _dominates(fa, fb):
    leq = all(a <= b for a, b in zip(fa, fb))
    lt  = any(a <  b for a, b in zip(fa, fb))
    return leq and lt

def fast_nondominated_sort(F):
    N = len(F)
    S = [set() for _ in range(N)]
    n = [0]*N
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q: continue
            if _dominates(F[p], F[q]): S[p].add(q)
            elif _dominates(F[q], F[p]): n[p] += 1
        if n[p] == 0: fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0: Q.append(q)
        i += 1
        fronts.append(Q)
    if not fronts[-1]: fronts.pop()
    return fronts

def crowding_distance(front, F):
    if not front: return {}
    m = len(F[0])
    dist = {i: 0.0 for i in front}
    for j in range(m):
        order = sorted(front, key=lambda i: F[i][j])
        dist[order[0]]  = float("inf")
        dist[order[-1]] = float("inf")
        lo = F[order[0]][j]; hi = F[order[-1]][j]
        rng = hi - lo
        if rng == 0: continue
        for k in range(1, len(order)-1):
            prev_i = order[k-1]; next_i = order[k+1]
            dist[order[k]] += (F[next_i][j] - F[prev_i][j]) / rng
    return dist

def _front_ranks(fronts):
    rank = {}
    for r, fr in enumerate(fronts):
        for i in fr: rank[i] = r
    return rank

def crowded_tournament(pop, fits, fronts, distances, tsize=2):
    import random
    n = len(pop)
    if n == 0:
        raise ValueError("empty population")
    if n == 1:
        return 0
    tsize = min(tsize, n)
    cand = random.sample(range(n), tsize)
    rank = _front_ranks(fronts)
    def better(i, j):
        ri, rj = rank[i], rank[j]
        if ri != rj: return ri < rj
        return distances.get(i, 0.0) > distances.get(j, 0.0)
    best = cand[0]
    for c in cand[1:]:
        if better(c, best): best = c
    return best

# =============================
# ============ STATE ==========
# =============================
_STATE = None

def get_state():
    """
    Initialize template, parameter order, initial vector, and spans.

    For multi-LULC: one gene per (parameter, LU) as defined by read_vector_from_file().
    """
    global _STATE
    if _STATE is None:
        template = read_lines(RRI_INPUT)

        # Detect which RRI_Input lines correspond to which parameter names
        names_by_line = detect_params(template)

        # Respect CAL_PARAMS order, but only keep parameters that actually appear
        ordered = [
            nm for nm in CAL_PARAMS
            if any(
                (names_by_line.get(ln) == nm)
                for ln in EDIT_LINE_RANGE
                if ln not in DO_NOT_TOUCH
            )
        ]
        if not ordered:
            # Fallback if nothing detected (very unlikely)
            ordered = ["ns_slope", "soildepth"]

        filtered = {ln: n for ln, n in names_by_line.items() if n in ordered}

        # This now builds a full multi-LULC vector
        param_order, vec0, spans = read_vector_from_file(template, filtered)

        _STATE = (template, param_order, vec0, spans)

    return _STATE


# =============================
# ============ RUN ============
# =============================
STOP_REQUESTED = False

def _handle_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[SIGNAL] Caught signal {signum}; will save checkpoint at next safe point.")

def save_checkpoint(gen_next, names, pop, rng_state, notes=None):
    """
    Save GA state so we can resume later, and also accumulate an elite history.

    The payload keeps:
    - gen_next: next generation index to run
    - names:    parameter name order
    - pop:      current population (as float arrays)
    - rng_state: NumPy RNG state
    - notes.elite_history: list of elite entries (one per generation),
      each containing metrics + parameter vector of the best individual.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Recover any existing elite history from an older checkpoint
    existing_notes = {}
    elite_history = []
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                old_payload = pickle.load(f)
            existing_notes = old_payload.get("notes", {}) or {}
            elite_history = list(existing_notes.get("elite_history", []))
        except Exception:
            existing_notes = {}
            elite_history = []

    # Merge new notes (typically carries "elite_entry")
    if notes is None:
        notes = {}
    elite_entry = notes.get("elite_entry")
    if elite_entry is not None:
        elite_history.append(elite_entry)
        existing_notes["elite_history"] = elite_history
        existing_notes["last_elite"] = elite_entry

    payload = {
        "gen_next": gen_next,
        "names": names,
        "pop": [np.array(ind, dtype=float) for ind in pop],
        "rng_state": rng_state,
        "timestamp": datetime.now().isoformat(),
        "notes": existing_notes,
    }
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"[CKPT] Saved checkpoint for Gen {gen_next} -> {CHECKPOINT_FILE}")

def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return None, None, None, None
    with open(CHECKPOINT_FILE, "rb") as f:
        payload = pickle.load(f)
    print(f"[CKPT] Loaded checkpoint from {CHECKPOINT_FILE} (Gen {payload.get('gen_next')})")
    return payload.get("gen_next"), payload.get("names"), payload.get("pop"), payload.get("rng_state")

def main():
    ensure_paths()
    np.random.seed(RNG_SEED)

    print(f"PRIMARY_SOURCE={PRIMARY_SOURCE}  |  MATCH_MODE={MATCH_MODE}")
    if MATCH_MODE == "daily":
        print(f"Daily evaluation at {DAILY_PICK_HOUR:02d}:{DAILY_PICK_MINUTE:02d}:{DAILY_PICK_SECOND:02d} (±{DAILY_TOLERANCE})")
    elif MATCH_MODE == "hourly":
        print(f"Hourly evaluation with tolerance ±{HOURLY_TOLERANCE}")
    else:
        print(f"Satellite-timestamp evaluation (SWOT tol={SAT_TOLERANCE_SIM_NEAR_SWOT}, ICESat tol={SAT_TOLERANCE_SIM_NEAR_ICESAT})")

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT,  _handle_stop)

    template, names, vec0, spans = get_state()
    print(f"Target LU: {TARGET_LU}")
    print("Tuning parameters:", ", ".join(names))

    # Initial population
    # - If USE_SEEDING == 0: purely random within BOUNDS (original behavior).
    # - If USE_SEEDING == 1: warm-start around vec0 (the parameters read from RRI_Input.txt).
    pop = init_population(POP_SIZE, names, vec0=vec0)

    gen_start, ckpt_names, ckpt_pop, ckpt_rng = load_checkpoint()
    if gen_start is not None:
        if ckpt_names != names:
            print("[CKPT] Parameter order changed; realigning by name.")
            name_to_idx = {nm: i for i, nm in enumerate(ckpt_names)}
            mapped = []
            for ind in ckpt_pop:
                mapped.append(np.array([ind[name_to_idx[n]] for n in names], dtype=float))
            ckpt_pop = mapped
        pop = ckpt_pop
        if ckpt_rng is not None:
            np.random.set_state(ckpt_rng)
        print(f"[RESUME] Continuing from Gen {gen_start}")
    else:
        gen_start = 1

    write_header = not TSV_LOG.exists()
    with open(TSV_LOG, "a", encoding="utf-8") as log:
        if write_header:
            header = [
                "Gen","Phase","Ind",
                # Primary metrics
                "NSE","RMSE","Bias",
                # Diagnostic metrics
                "OBS_NSE","OBS_RMSE","OBS_Bias","OBS_MAE","OBS_KGE","OBS_r","OBS_n",
                "SWOT_NSE","SWOT_RMSE","SWOT_Bias","SWOT_MAE","SWOT_KGE","SWOT_r","SWOT_n",
                "ICESAT_NSE","ICESAT_RMSE","ICESAT_Bias","ICESAT_MAE","ICESAT_KGE","ICESAT_r","ICESAT_n",
                "Target_LU"
            ] + names
            log.write("\t".join(header) + "\n")

    deadline = None
    if GA_MAX_SECONDS > 0:
        deadline = time.time() + max(0, GA_MAX_SECONDS - GRACE_SECONDS)
        print(f"[BUDGET] Will stop before: {GRACE_SECONDS}s grace")

    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS,
                             initializer=_init_worker,
                             initargs=(str(PARTICLE_DIR), str(WORKERS_ROOT))) as executor:

        for gen in range(gen_start, GENERATIONS+1):
            # ---- Evaluate current population ----
            fits_pop = [None] * len(pop)
            futures = {}
            for ind_idx, ind in enumerate(pop, start=1):
                fut = executor.submit(evaluate_worker, ind.copy(), names, template, spans, gen, ind_idx)
                futures[fut] = ("pop", ind_idx, ind)

            with open(TSV_LOG, "a", encoding="utf-8") as log:
                for fut in as_completed(futures):
                    phase, ind_idx, ind_vec = futures[fut]
                    nse, rmse, bias, row = fut.result()
                    if phase == "pop":
                        fits_pop[ind_idx-1] = (nse, rmse, bias)
                    append_log(row)

                    # Pull per-source metrics
                    ps = row.get("per_source_metrics", {})
                    def g(src, k):
                        v = ps.get(src, {}).get(k)
                        return ("nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.6f}" if isinstance(v, float) else v)
                    line = [gen, phase, ind_idx,
                            f"{nse:.6f}", f"{rmse:.6f}", f"{bias:.6f}",
                            g("OBS","NSE"), g("OBS","RMSE"), g("OBS","Bias"), g("OBS","MAE"), g("OBS","KGE"), g("OBS","Corr"), ps.get("OBS",{}).get("n_pairs",0),
                            g("SWOT","NSE"), g("SWOT","RMSE"), g("SWOT","Bias"), g("SWOT","MAE"), g("SWOT","KGE"), g("SWOT","Corr"), ps.get("SWOT",{}).get("n_pairs",0),
                            g("ICESAT","NSE"), g("ICESAT","RMSE"), g("ICESAT","Bias"), g("ICESAT","MAE"), g("ICESAT","KGE"), g("ICESAT","Corr"), ps.get("ICESAT",{}).get("n_pairs",0),
                            TARGET_LU] + [f"{v:.8g}" for v in ind_vec]
                    log.write("\t".join(map(str, line)) + "\n")
                log.flush()

            # ---- NSGA-II selection on (NSE, RMSE, |Bias|) of PRIMARY_SOURCE ----
            F_pop = [(-nse, rmse, abs(bias)) for (nse, rmse, bias) in fits_pop]
            fronts_pop = fast_nondominated_sort(F_pop)
            crowd_pop = {}
            for fr in fronts_pop:
                crowd_pop.update(crowding_distance(fr, F_pop))

            # ---- Create offspring ----
            mu = len(pop); lam = mu
            offspring = []
            while len(offspring) < lam:
                i = crowded_tournament(pop, fits_pop, fronts_pop, crowd_pop, tsize=2)
                j = crowded_tournament(pop, fits_pop, fronts_pop, crowd_pop, tsize=2)
                a = pop[i]; b = pop[j]
                child = a.copy()
                if np.random.rand() < CX_PROB and i != j:
                    child = uniform_crossover(a, b, names, p=0.5)
                child = gaussian_mutation(child, names)
                offspring.append(child)

            # ---- Evaluate offspring ----
            fits_off = [None] * len(offspring)
            futures = {}
            for ind_idx, ind in enumerate(offspring, start=1):
                global_ind = POP_SIZE + ind_idx
                fut = executor.submit(evaluate_worker, ind.copy(), names, template, spans, gen, global_ind)
                futures[fut] = ("offspring", ind_idx, ind, global_ind)

            with open(TSV_LOG, "a", encoding="utf-8") as log:
                for fut in as_completed(futures):
                    phase, local_idx, ind_vec, global_ind = futures[fut]
                    nse, rmse, bias, row = fut.result()
                    fits_off[local_idx-1] = (nse, rmse, bias)
                    append_log(row)
                    ps = row.get("per_source_metrics", {})
                    def g(src, k):
                        v = ps.get(src, {}).get(k)
                        return ("nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.6f}" if isinstance(v, float) else v)
                    line = [gen, phase, local_idx,
                            f"{nse:.6f}", f"{rmse:.6f}", f"{bias:.6f}",
                            g("OBS","NSE"), g("OBS","RMSE"), g("OBS","Bias"), g("OBS","MAE"), g("OBS","KGE"), g("OBS","Corr"), ps.get("OBS",{}).get("n_pairs",0),
                            g("SWOT","NSE"), g("SWOT","RMSE"), g("SWOT","Bias"), g("SWOT","MAE"), g("SWOT","KGE"), g("SWOT","Corr"), ps.get("SWOT",{}).get("n_pairs",0),
                            g("ICESAT","NSE"), g("ICESAT","RMSE"), g("ICESAT","Bias"), g("ICESAT","MAE"), g("ICESAT","KGE"), g("ICESAT","Corr"), ps.get("ICESAT",{}).get("n_pairs",0),
                            TARGET_LU] + [f"{v:.8g}" for v in ind_vec]
                    log.write("\t".join(map(str, line)) + "\n")
                log.flush()

            # ---- Environmental selection ----
            merged = pop + offspring
            mfits  = fits_pop + fits_off
            F_all  = [(-nse, rmse, abs(bias)) for (nse, rmse, bias) in mfits]

            fronts = fast_nondominated_sort(F_all)
            new_pop = []
            for fr in fronts:
                if len(new_pop) + len(fr) <= POP_SIZE:
                    new_pop.extend([merged[i] for i in fr])
                else:
                    d = crowding_distance(fr, F_all)
                    fr_sorted = sorted(fr, key=lambda i: d[i], reverse=True)
                    remain = POP_SIZE - len(new_pop)
                    new_pop.extend([merged[i] for i in fr_sorted[:remain]])
                    break
            pop = new_pop

            # ---- Progress + checkpoint ----
            first_front = fronts[0]
            # Choose a "best" individual from the first front using the same objective vector F_all
            # F_all[i] = (-NSE, RMSE, |Bias|), so lexicographic min is our compromise best
            best_idx = min(first_front, key=lambda i: F_all[i])
            best = mfits[best_idx]

            # Record best metrics and parameters for this generation
            # IMPORTANT: use 'merged', not 'pop', because best_idx refers to merged (parents+offspring)
            best_params = {nm: float(merged[best_idx][i]) for i, nm in enumerate(names)}
            print(f"[Gen {gen:03d}] Best NSE={best[0]:.4f} RMSE={best[1]:.4f} |Bias|={abs(best[2]):.4f}")

            elite_entry = {
                "gen": gen,
                "primary_source": PRIMARY_SOURCE,
                "metrics": {"NSE": float(best[0]), "RMSE": float(best[1]), "Bias": float(best[2])},
                "params": best_params,
            }

            save_checkpoint(gen_next=gen+1,
                            names=names,
                            pop=pop,
                            rng_state=np.random.get_state(),
                            notes={"elite_entry": elite_entry})

            elite_entry = {
                "gen": gen,
                "primary_source": PRIMARY_SOURCE,
                "metrics": {"NSE": float(best[0]), "RMSE": float(best[1]), "Bias": float(best[2])},
                "params": best_params,
            }

            save_checkpoint(gen_next=gen+1,
                            names=names,
                            pop=pop,
                            rng_state=np.random.get_state(),
                            notes={"elite_entry": elite_entry})

            if STOP_REQUESTED:
                print("[CKPT] Stop requested by signal; exiting after checkpoint.")
                return
            if deadline is not None and time.time() > deadline:
                print("[CKPT] Reached runtime budget; exiting after checkpoint.")
                return

    # Optionally clean up checkpoint after a successful, uninterrupted run.
    if CHECKPOINT_FILE.exists() and DELETE_CHECKPOINT_ON_SUCCESS:
        try:
            CHECKPOINT_FILE.unlink()
            print("[CKPT] Completed run; removed checkpoint file.")
        except Exception:
            pass

    print(f"\nJSONL log: {JSONL_LOG}")
    print(f"TSV log:   {TSV_LOG}")

if __name__ == "__main__":
    main()
