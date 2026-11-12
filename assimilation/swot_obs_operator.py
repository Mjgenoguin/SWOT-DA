# =========================================================
# SpatialDA/PythonCode/swot_obs_operator.py
# SWOT Cycle-Cells (parquet) → Cell-baseline anomalies → Model sampling (hr_*.out) → calc_v
#
# Designed to be imported by your DA main driver.
# Windows-friendly. No DA_Latest coupling beyond folder conventions.
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Union

import numpy as np
import pandas as pd
import os
import re
import datetime as _dt


# -----------------------------
# Config dataclasses
# -----------------------------

@dataclass
class SWOTQC:
    """Quality-control knobs for cycle-cell parquet."""
    min_n_nodes: int = 1
    max_node_q_median: Optional[float] = None
    max_wse_std: Optional[float] = None
    require_reliable_baseline: bool = False   # if True: keep only baseline.reliable == True


@dataclass
class ParticleLayout:
    """
    Defines how particle outputs are organized.
    Default expects:
      <particles_root>/Particle00001/RRI/out/hr_000021.out
    """
    particle_glob: str = "Particle*"
    rri_out_rel: Path = Path("RRI") / "out"
    hr_prefix: str = "hr_"
    hr_suffix: str = ".out"
    nodata_value: float = -0.1
    strict_particles: bool = True  # if True: raise if missing hr file for any particle


# -----------------------------
# Utilities
# -----------------------------

def _ensure_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def list_cycle_parquets(swot_dir: Union[str, Path]) -> List[Path]:
    swot_dir = _ensure_path(swot_dir)
    files = sorted(swot_dir.glob("cycle_*_cells.parquet"))
    return files


def build_swot_cell_baseline(
    swot_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    wse_field: str = "wse_median",
    qc: Optional[SWOTQC] = None,
    baseline_stat: str = "median",
) -> pd.DataFrame:
    """
    Build per-cell baseline from all cycle parquet files:
      baseline indexed by (row,col), includes mu_swot and diagnostics.

    Returns baseline DataFrame and optionally writes parquet if out_path provided.
    """
    swot_dir = _ensure_path(swot_dir)
    qc = qc or SWOTQC()

    files = list_cycle_parquets(swot_dir)
    if not files:
        raise FileNotFoundError(f"No cycle parquet files found in: {swot_dir}")

    keep_cols = [
        "row", "col", "cell_id", "lon_cell", "lat_cell",
        "wse_median", "wse_mean", "wse_std", "wse_u_median",
        "n_nodes", "node_q_min", "node_q_median",
        "cycle_start_ict", "cycle_end_ict", "time_min_ict", "time_max_ict"
    ]

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols].copy()
        df["source_file"] = f.name
        dfs.append(df)

    swot_all = pd.concat(dfs, ignore_index=True)

    # basic required fields
    if "row" not in swot_all.columns or "col" not in swot_all.columns:
        raise ValueError("Cycle parquet must include 'row' and 'col' columns.")
    if wse_field not in swot_all.columns:
        raise ValueError(f"Cycle parquet missing wse_field='{wse_field}'")

    swot_all = swot_all.dropna(subset=["row", "col", wse_field]).copy()
    swot_all["row"] = swot_all["row"].astype(int)
    swot_all["col"] = swot_all["col"].astype(int)

    # QC
    if "n_nodes" in swot_all.columns and qc.min_n_nodes is not None:
        swot_all = swot_all[swot_all["n_nodes"] >= qc.min_n_nodes]
    if qc.max_node_q_median is not None and "node_q_median" in swot_all.columns:
        swot_all = swot_all[swot_all["node_q_median"] <= qc.max_node_q_median]
    if qc.max_wse_std is not None and "wse_std" in swot_all.columns:
        swot_all = swot_all[swot_all["wse_std"] <= qc.max_wse_std]

    group_keys = ["row", "col"]

    agg: Dict[str, Tuple[str, Union[str, callable]]] = {
        "n_obs": (wse_field, "size"),
        "mu_swot_mean": (wse_field, "mean"),
        "mu_swot_median": (wse_field, "median"),
        "std_swot": (wse_field, "std"),
        "min_swot": (wse_field, "min"),
        "max_swot": (wse_field, "max"),
    }
    if "lon_cell" in swot_all.columns:
        agg["lon_cell_med"] = ("lon_cell", "median")
    if "lat_cell" in swot_all.columns:
        agg["lat_cell_med"] = ("lat_cell", "median")
    if "cell_id" in swot_all.columns:
        agg["cell_id_mode"] = ("cell_id", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0])

    baseline = (
        swot_all
        .groupby(group_keys, as_index=False)
        .agg(**agg)
    )

    baseline_stat = baseline_stat.lower()
    if baseline_stat == "median":
        baseline["mu_swot"] = baseline["mu_swot_median"]
    elif baseline_stat == "mean":
        baseline["mu_swot"] = baseline["mu_swot_mean"]
    else:
        raise ValueError("baseline_stat must be 'mean' or 'median'")

    baseline["reliable"] = baseline["n_obs"] >= 3

    if out_path is not None:
        out_path = _ensure_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        baseline.to_parquet(out_path, index=False)

    return baseline


def load_cycle_cells_with_anomaly(
    cycle_path: Union[str, Path],
    baseline: Union[pd.DataFrame, str, Path],
    wse_field: str = "wse_median",
    qc: Optional[SWOTQC] = None,
) -> pd.DataFrame:
    """
    Load one cycle parquet and compute obs anomaly:
      obs_anom = wse_field - mu_swot  (joined by row,col)

    Returns df sorted by (row,col) with at least:
      row,col,wse_field,mu_swot,obs_anom (+ diagnostics if present)
    """
    qc = qc or SWOTQC()
    cycle_path = _ensure_path(cycle_path)

    if isinstance(baseline, (str, Path)):
        baseline_df = pd.read_parquet(_ensure_path(baseline))
    else:
        baseline_df = baseline

    cycle = pd.read_parquet(cycle_path)

    if "row" not in cycle.columns or "col" not in cycle.columns:
        raise ValueError("Cycle parquet must include 'row' and 'col'")
    if wse_field not in cycle.columns:
        raise ValueError(f"Cycle parquet missing wse_field='{wse_field}'")

    needed = [
        "row", "col", wse_field,
        "n_nodes", "node_q_median", "node_q_min",
        "wse_std", "wse_u_median",
        "cycle_start_ict", "cycle_end_ict", "time_min_ict", "time_max_ict",
        "cell_id", "lon_cell", "lat_cell"
    ]
    cycle = cycle[[c for c in needed if c in cycle.columns]].copy()

    cycle["row"] = cycle["row"].astype(int)
    cycle["col"] = cycle["col"].astype(int)
    cycle = cycle.dropna(subset=[wse_field]).copy()

    # QC
    if "n_nodes" in cycle.columns and qc.min_n_nodes is not None:
        cycle = cycle[cycle["n_nodes"] >= qc.min_n_nodes]
    if qc.max_node_q_median is not None and "node_q_median" in cycle.columns:
        cycle = cycle[cycle["node_q_median"] <= qc.max_node_q_median]
    if qc.max_wse_std is not None and "wse_std" in cycle.columns:
        cycle = cycle[cycle["wse_std"] <= qc.max_wse_std]

    base_key = baseline_df[["row", "col", "mu_swot", "n_obs", "std_swot", "reliable"]].copy()
    df = cycle.merge(base_key, on=["row", "col"], how="left")

    missing_mu = int(df["mu_swot"].isna().sum())
    total = int(len(df))

    # If EVERYTHING is missing baseline, this is not "no obs" — it's a baseline mismatch.
    if total > 0 and missing_mu == total:
        raise ValueError(
            f"Baseline join failed: 0/{total} cycle cells matched baseline (mu_swot all NaN). "
            f"Likely baseline was built with different QC/filtering or from a different set of cycle files. "
            f"Rebuild baseline from the SAME cycle_*_cells.parquet corpus and QC settings."
        )

    # If only SOME are missing, drop them but warn loudly (keeps DA scientifically defensible)
    if missing_mu > 0:
        print(f"[SWOT][WARN] {missing_mu}/{total} cycle cells missing baseline mu_swot; dropping unmatched cells.")

    df = df.dropna(subset=["mu_swot"]).copy()


    if qc.require_reliable_baseline and "reliable" in df.columns:
        df = df[df["reliable"] == True].copy()

    df["obs_anom"] = df[wse_field] - df["mu_swot"]
    df = df.sort_values(["row", "col"]).reset_index(drop=True)
    return df

def _ensure_particles_first(calc_v: np.ndarray, n_particles: int, n_cells: int) -> np.ndarray:
    """
    Ensure calc_v is shaped (n_particles, n_cells).
    Accepts either (n_particles, n_cells) or (n_cells, n_particles).
    """
    if calc_v.shape == (n_particles, n_cells):
        return calc_v
    if calc_v.shape == (n_cells, n_particles):
        return calc_v.T
    raise ValueError(
        f"Unexpected calc_v shape {calc_v.shape}. Expected "
        f"({n_particles},{n_cells}) or ({n_cells},{n_particles})."
    )

def _model_anomaly_from_ensemble(calc_v_abs: np.ndarray):
    """
    Option A: model anomaly is particle value minus ensemble mean at the same time.
    calc_v_abs must be (n_particles, n_cells).
    Returns: (calc_v_anom, mu_model)
    """
    mu_model = np.nanmean(calc_v_abs, axis=0)          # (n_cells,)
    calc_v_anom = calc_v_abs - mu_model[None, :]       # (n_particles, n_cells)
    return calc_v_anom, mu_model

def _infer_cycle_dt_from_name(cycle_name: str) -> Optional["_dt.datetime"]:
    """
    Infer cycle datetime (ICT) from filename like: cycle_YYYYMMDD_HHMM_cells.parquet
    Returns naive datetime (local/ICT semantics are fine for month/DOY selection).
    """
    m = re.search(r"cycle_(\d{8})_(\d{4})", str(cycle_name))
    if not m:
        return None
    yyyymmdd = m.group(1)
    hhmm = m.group(2)
    try:
        return _dt.datetime.strptime(yyyymmdd + hhmm, "%Y%m%d%H%M")
    except Exception:
        return None


def _mu_model_from_baseline(
    obs_df: pd.DataFrame,
    model_baseline: dict,
    cycle_dt: Optional["_dt.datetime"],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-cell mu_model from model baseline arrays using (row,col)->flat_idx->river index mapping.

    Returns:
      mu_model: (nCells_kept,) float array
      keep_mask: (nCells_original,) boolean mask indicating which obs rows are kept (if strict=1)
    """
    rows = obs_df["row"].to_numpy(dtype=int)
    cols = obs_df["col"].to_numpy(dtype=int)

    ncols = int(model_baseline["ncols"])
    flat = rows * ncols + cols

    river_flat_idx = model_baseline["river_flat_idx"]  # shape (n_river,)
    pos = np.searchsorted(river_flat_idx, flat)
    in_range = (pos >= 0) & (pos < river_flat_idx.shape[0])
    match = in_range & (river_flat_idx[pos.clip(0, river_flat_idx.shape[0]-1)] == flat)

    strict = int(model_baseline.get("strict", 1)) == 1

    # mode selection
    mode = str(model_baseline.get("mode", "OVERALL")).upper().strip()
    invalid_value = float(model_baseline.get("invalid_value", -0.1))
    invalid_tol = float(model_baseline.get("invalid_tol", 1e-12))

    meta = model_baseline.get("meta", {}) or {}
    wet_months = meta.get("wet_months", [5, 6, 7, 8, 9, 10])

    # Determine seasonal indices if needed
    month_idx = None
    doy_idx = None
    wetdry_idx = None
    if cycle_dt is not None:
        month_idx = int(cycle_dt.month) - 1            # 0..11
        doy_idx = int(cycle_dt.timetuple().tm_yday) - 1  # 0..365
        wetdry_idx = 1 if int(cycle_dt.month) in set(wet_months) else 0  # 0=dry, 1=wet

    # Create mu_model for ALL rows first (NaN where unmapped)
    mu_all = np.full(len(obs_df), np.nan, dtype=float)

    # Fill mapped rows
    if np.any(match):
        p = pos[match]

        if mode == "OVERALL":
            base = model_baseline["overall"]
            mu_all[match] = np.asarray(base[p], dtype=float)

        elif mode == "WETDRY":
            base = model_baseline.get("wetdry", None)
            if base is None:
                raise ValueError("[MODEL_BASELINE] Mode=WETDRY but WetDry array not loaded.")
            if wetdry_idx is None:
                # fallback: assume WET if no dt
                wetdry_idx = 1
            mu_all[match] = np.asarray(base[wetdry_idx, p], dtype=float)

        elif mode == "MONTHLY":
            base = model_baseline.get("monthly", None)
            if base is None:
                raise ValueError("[MODEL_BASELINE] Mode=MONTHLY but Monthly array not loaded.")
            if month_idx is None:
                month_idx = 0
            mu_all[match] = np.asarray(base[month_idx, p], dtype=float)

        elif mode == "DOY":
            base = model_baseline.get("doy", None)
            if base is None:
                raise ValueError("[MODEL_BASELINE] Mode=DOY but DOY array not loaded.")
            if doy_idx is None:
                doy_idx = 0
            doy_idx = max(0, min(int(doy_idx), base.shape[0] - 1))
            mu_all[match] = np.asarray(base[doy_idx, p], dtype=float)

        else:
            raise ValueError(f"[MODEL_BASELINE] Unknown Mode: {mode}")

    # Invalidate sentinel-like baseline (optional)
    invalid = np.isfinite(mu_all) & (np.abs(mu_all - invalid_value) <= invalid_tol)

    if strict:
        keep = match & (~invalid) & np.isfinite(mu_all)
        mu_kept = mu_all[keep]
        return mu_kept, keep
    else:
        # keep all; caller can fallback for NaN mu_model
        keep = np.ones(len(obs_df), dtype=bool)
        mu_all[invalid] = np.nan
        return mu_all[keep], keep

def sample_hr_at_cells(
    hr_path: Union[str, Path],
    rows: np.ndarray,
    cols: np.ndarray,
    nodata_value: float = -0.1,
    strict_nodata: bool = True,
) -> np.ndarray:
    """
    Load hr grid and sample at arrays rows/cols.
    """
    hr_path = _ensure_path(hr_path)
    grid = np.loadtxt(hr_path)

    nR, nC = grid.shape
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= nR or cols.max() >= nC:
        raise ValueError(f"(row,col) out of bounds for grid shape {grid.shape} at {hr_path}")

    vals = grid[rows, cols]
    if strict_nodata and np.any(vals == nodata_value):
        raise ValueError(f"NODATA encountered in sampled cells for {hr_path}")
    return vals


def particle_hr_path(
    particle_dir: Union[str, Path],
    hr_index: int,
    layout: Optional[ParticleLayout] = None,
) -> Path:
    """Return the saved RRI river-depth state for one comparison index."""
    layout = layout or ParticleLayout()
    particle_dir = _ensure_path(particle_dir)
    hr_index = int(hr_index)
    if hr_index < 0:
        raise ValueError(f"hr_index must be non-negative, got {hr_index}")
    if hr_index == 0:
        # The cycle-start state is the particle-specific RRI initial condition.
        return particle_dir / Path(layout.rri_out_rel).parent / "init" / "hr_init.out"
    target_name = f"{layout.hr_prefix}{hr_index:06d}{layout.hr_suffix}"
    return particle_dir / layout.rri_out_rel / target_name


def build_calc_v_from_particles(
    particles_root: Union[str, Path],
    hr_index: int,
    obs_df: pd.DataFrame,
    layout: Optional[ParticleLayout] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build calc_v by sampling each particle saved hr state at obs_df cell list.

    hr_index=0 uses RRI/init/hr_init.out; positive indices use
    RRI/out/hr_{hr_index:06d}.out.

    Returns:
      calc_v: shape (nCells, nParticlesUsed)
      particles_used: list of particle folder names (order matches columns)
    """
    particles_root = _ensure_path(particles_root)
    layout = layout or ParticleLayout()

    if "row" not in obs_df.columns or "col" not in obs_df.columns:
        raise ValueError("obs_df must include 'row' and 'col'")

    rows = obs_df["row"].to_numpy(dtype=int)
    cols = obs_df["col"].to_numpy(dtype=int)

    particle_dirs = sorted([p for p in particles_root.glob(layout.particle_glob) if p.is_dir()])
    if not particle_dirs:
        raise FileNotFoundError(f"No particle dirs found under: {particles_root} with glob {layout.particle_glob}")

    target_name = (
        "hr_init.out"
        if int(hr_index) == 0
        else f"{layout.hr_prefix}{int(hr_index):06d}{layout.hr_suffix}"
    )

    calc_cols: List[np.ndarray] = []
    used: List[str] = []
    missing: List[str] = []

    for pdir in particle_dirs:
        hr_path = particle_hr_path(pdir, hr_index, layout)
        if not hr_path.exists():
            missing.append(pdir.name)
            continue

        vals = sample_hr_at_cells(
            hr_path,
            rows=rows,
            cols=cols,
            nodata_value=layout.nodata_value,
            strict_nodata=True,
        )
        calc_cols.append(vals)
        used.append(pdir.name)

    if layout.strict_particles:
        # if strict: require all particle dirs to have the output
        if missing:
            raise FileNotFoundError(
                f"Missing {target_name} for {len(missing)} particle(s). "
                f"Example missing: {missing[:10]}"
            )

    if not calc_cols:
        raise RuntimeError(
            f"No particles produced valid {target_name} for sampling. "
            f"Check particles_root and hr_index."
        )

    # (nParticles, nCells) -> transpose to (nCells, nParticles)
    calc_v = np.array(calc_cols, dtype=float).T
    return calc_v, used


def make_pf_inputs_from_swot(
    cycle_path: Union[str, Path, List[Union[str, Path]]],
    baseline_path: Union[str, Path],  # legacy overall baseline (backward compatible)
    particles_root: Union[str, Path],
    hr_index: int,
    wse_field: str = "wse_median",
    qc: Optional[SWOTQC] = None,
    layout: Optional[ParticleLayout] = None,

    # --- NEW: decoupled SWOT baseline selection ---
    baseline_mode: str = "OVERALL",   # OVERALL | WETDRY
    baseline_overall: Optional[Union[str, Path]] = None,
    baseline_wet: Optional[Union[str, Path]] = None,
    baseline_dry: Optional[Union[str, Path]] = None,
    wet_months: Optional[List[int]] = None,

    # --- R / sigma knobs ---
    sigma_field: str = "wse_u_median",
    sigma_agg: str = "MEDIAN",          # MEDIAN or MAX
    sigma_min: float = 0.05,            # meters
    sigma_max: float = -1.0,            # meters; negative disables
    sigma_inflation: float = 1.0,       # >=1 reduces trust
    sigma_rep: float = 0.0,             # meters representativeness added in quadrature
    model_baseline: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    End-to-end helper:
      cycle parquet + baseline -> obs_df (with obs_anom) -> calc_v from particles.

    Returns:
      obs_v: (nCells,)
      obs_t: (nCells,)  (time_max_ict string if available else cycle_start_ict else empty)
      calc_v: (nCells, nParticles)
      obs_df: dataframe used
      particles_used: list of particle folder names
    """
    qc = qc or SWOTQC()
    layout = layout or ParticleLayout()

    # Accept either a single parquet path OR a list of parquet paths (multiple SWOT passes in window)
    if isinstance(cycle_path, (list, tuple)):
        cycle_paths = [Path(p) for p in cycle_path]
    else:
        cycle_paths = [Path(cycle_path)]

    # --- baseline selection setup (NEW) ---
    baseline_mode = (baseline_mode or "OVERALL").strip().upper()
    if baseline_mode not in ("OVERALL", "WETDRY"):
        raise ValueError(f"[SWOT][BASELINE] baseline_mode must be OVERALL or WETDRY, got: {baseline_mode}")

    baseline_overall = baseline_overall or baseline_path
    wet_months = wet_months or [5, 6, 7, 8, 9, 10]
    wet_months_set = set(int(m) for m in wet_months)

    if baseline_mode == "WETDRY":
        if not baseline_wet or not baseline_dry:
            raise ValueError(
                "[SWOT][BASELINE] baseline_mode=WETDRY requires baseline_wet and baseline_dry paths."
            )

    dfs = []
    for cp in cycle_paths:
        # choose which SWOT baseline parquet to use for THIS cycle file
        base_to_use = baseline_overall
        base_tag = "OVERALL"

        if baseline_mode == "WETDRY":
            cycle_dt = _infer_cycle_dt_from_name(cp.name)
            if cycle_dt is None:
                # fallback: treat as wet (conservative, typically more samples)
                base_to_use = baseline_wet
                base_tag = "WET(fallback_no_dt)"
            else:
                is_wet = int(cycle_dt.month) in wet_months_set
                base_to_use = baseline_wet if is_wet else baseline_dry
                base_tag = "WET" if is_wet else "DRY"

        print(f"[SWOT][BASELINE] mode={baseline_mode} cycle={cp.name} -> {base_tag} baseline={base_to_use}")

        dfi = load_cycle_cells_with_anomaly(
            cycle_path=cp,
            baseline=base_to_use,
            wse_field=wse_field,
            qc=qc,
        )
        dfi["source_cycle"] = cp.name
        dfi["baseline_mode"] = baseline_mode
        dfi["baseline_used"] = os.path.basename(str(base_to_use))
        dfs.append(dfi)

    if len(dfs) == 0:
        obs_df = pd.DataFrame()
    else:
        obs_df = pd.concat(dfs, ignore_index=True)

        # If same (row,col) appears in multiple passes within window, collapse to one obs per cell
        if not obs_df.empty:
            sigma_agg = (sigma_agg or "MEDIAN").strip().upper()
            if sigma_agg not in ("MEDIAN", "MAX"):
                raise ValueError(f"sigma_agg must be MEDIAN or MAX, got: {sigma_agg}")

            _sigma_op = "median" if sigma_agg == "MEDIAN" else "max"
            _sigma_dict = {sigma_field: (sigma_field, _sigma_op)} if (sigma_field in obs_df.columns) else {}

            obs_df = (
                obs_df.groupby(["row", "col"], as_index=False)
                      .agg(
                          obs_anom=("obs_anom", "median"),
                          mu_swot=("mu_swot", "first"),
                          **({wse_field: (wse_field, "median")} if wse_field in obs_df.columns else {}),
                          **(_sigma_dict),
                          cell_id=("cell_id", "first") if "cell_id" in obs_df.columns else ("row", "first"),
                          lon_cell=("lon_cell", "first") if "lon_cell" in obs_df.columns else ("row", "first"),
                          lat_cell=("lat_cell", "first") if "lat_cell" in obs_df.columns else ("row", "first"),
                          n_nodes=("n_nodes", "max") if "n_nodes" in obs_df.columns else ("row", "first"),
                          time_min_ict=("time_min_ict", "max") if "time_min_ict" in obs_df.columns else ("row", "first"),
                          time_max_ict=("time_max_ict", "max") if "time_max_ict" in obs_df.columns else ("row", "first"),
                      )
                      .sort_values(["row", "col"])
                      .reset_index(drop=True)
            )

    # If no valid SWOT cells, do NOT attempt to sample hr outputs.
    if obs_df.empty:
        obs_v = np.array([], dtype=float)
        obs_t = np.array([], dtype=object)
        calc_v = np.empty((0, 0), dtype=float)
        sigma_v = np.array([], dtype=float)
        return obs_v, obs_t, calc_v, sigma_v, obs_df, []

    # ===================== PATCH: drop SWOT cells that map to model NODATA =====================
    # File: PythonCode/swot_obs_operator.py
    #
    # INSERT this block inside:
    #   def make_pf_inputs_from_swot(...):
    # RIGHT AFTER you finish building `obs_df_swot` (the dataframe from parquet)
    # AND RIGHT BEFORE this line (or equivalent):
    #   calc_v, particles_used = build_calc_v_from_particles(...)
    #
    # This prevents hard-crash when some SWOT cells land on inactive RRI mask cells (NODATA=-0.1).
    # ==========================================================================================

    # --- (A) helper: read hr_*.out as 2D float array (use your existing reader if present) ---
    def _read_hr_out_2d(hr_path: str) -> np.ndarray:
        """
        Reads an RRI hr_*.out file into a 2D numpy array (nrows, ncols).
        Assumes whitespace-delimited numeric grid, one row per line.
        """
        # Fast and robust for plain text grids
        arr = np.loadtxt(hr_path, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Unexpected hr grid shape from {hr_path}: {arr.shape}")
        return arr

    def _filter_obs_df_by_hr_mask(
        obs_df: "pd.DataFrame",
        ref_hr_path: str,
        nodata_value: float = -0.1,
        row_col_names: tuple = ("row", "col"),
    ) -> "pd.DataFrame":
        """
        Drops observations whose (row,col) map to NODATA in reference HR grid.
        """
        rname, cname = row_col_names
        if rname not in obs_df.columns or cname not in obs_df.columns:
            raise KeyError(f"obs_df missing required columns: {rname}, {cname}")

        hr = _read_hr_out_2d(ref_hr_path)

        rows = obs_df[rname].to_numpy(dtype=int)
        cols = obs_df[cname].to_numpy(dtype=int)

        # bounds check (defensive)
        nrows, ncols = hr.shape
        in_bounds = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)

        # mask validity: not nodata
        valid = np.zeros(len(obs_df), dtype=bool)
        valid[in_bounds] = hr[rows[in_bounds], cols[in_bounds]] != nodata_value

        dropped = int((~valid).sum())
        kept = int(valid.sum())

        if dropped > 0:
            print(
                f"[SWOT][WARN] Dropped {dropped}/{len(obs_df)} SWOT cells that map to model "
                f"NODATA ({nodata_value}) or out-of-bounds using ref grid: {ref_hr_path}"
            )

        # IMPORTANT: reset index so downstream vectorization aligns cleanly
        return obs_df.loc[valid].reset_index(drop=True)

    # --- (B) choose a reference HR file and filter obs_df_swot before sampling particles ---
    # NOTE: this assumes you already have `particles_dir` (Particles folder) and `present_hr_name`
    #       or can hardcode the hr file you are sampling for this assimilation step.
    #
    # If your code already has a variable for the HR filename (e.g., "hr_000024.out"),
    # use that. Otherwise set it explicitly below.

    # Example: if you are sampling "hr_000024.out" at this PF step:
    # Reference particle folder (use Particle00001 as baseline mask)
    _ref_hr_path = str(
        particle_hr_path(
            Path(particles_root) / "Particle00001",
            hr_index,
            layout,
        )
    )

    if os.path.exists(_ref_hr_path):
        obs_df = _filter_obs_df_by_hr_mask(
            obs_df,
            ref_hr_path=_ref_hr_path,
            nodata_value=layout.nodata_value,
            row_col_names=("row", "col"),
        )
    else:
        print(f"[SWOT][WARN] Reference HR not found for NODATA mask; skipping mask filter: {_ref_hr_path}")

    if len(obs_df) == 0:
        print("[SWOT][INFO] All SWOT cells dropped by model mask (no valid obs for this cycle).")
        obs_v = np.array([], dtype=float)
        obs_t = np.array([], dtype=object)
        calc_v = np.empty((0, 0), dtype=float)
        sigma_v = np.array([], dtype=float)
        return obs_v, obs_t, calc_v, sigma_v, obs_df, []
    # ===================== END PATCH ==========================================================


    # =========================================================
    # MODEL BASELINE filter + per-cell mu_model
    # (so model anomaly = hr - mu_model_baseline, matching obs anomaly definition)
    # =========================================================
    mu_model_baseline = None

    if (model_baseline is not None) and int(model_baseline.get("enable", 0)) == 1:
        # infer cycle date from first cycle file name (sufficient for OVERALL/WETDRY/MONTHLY/DOY selection)
        _dt0 = _infer_cycle_dt_from_name(cycle_paths[0].name) if len(cycle_paths) else None

        mu_vec, keep_mask = _mu_model_from_baseline(obs_df=obs_df, model_baseline=model_baseline, cycle_dt=_dt0)

        dropped = int((~keep_mask).sum())
        if dropped > 0:
            print(f"[MODEL_BASELINE][WARN] Dropped {dropped}/{len(obs_df)} SWOT cells not mappable to river baseline (Strict={model_baseline.get('strict', 1)}).")

        obs_df = obs_df.loc[keep_mask].reset_index(drop=True)
        mu_model_baseline = np.asarray(mu_vec, dtype=float)

        if len(obs_df) == 0:
            print("[MODEL_BASELINE][INFO] No SWOT cells remain after baseline mapping filter.")
            obs_v = np.array([], dtype=float)
            obs_t = np.array([], dtype=object)
            calc_v = np.empty((0, 0), dtype=float)
            sigma_v = np.array([], dtype=float)
            return obs_v, obs_t, calc_v, sigma_v, obs_df, []


    calc_v, particles_used = build_calc_v_from_particles(
        particles_root=particles_root,
        hr_index=hr_index,
        obs_df=obs_df,
        layout=layout,
    )

    obs_v = obs_df["obs_anom"].to_numpy(dtype=float)

    # --- sigma_v for DIAG_HETERO (still returned even if caller ignores it) ---
    if sigma_field not in obs_df.columns:
        # If caller requests DIAG_HETERO but field is missing, main.py will catch via R_Model.
        sigma_v = np.full(len(obs_df), np.nan, dtype=float)
    else:
        sigma_v = obs_df[sigma_field].to_numpy(dtype=float)

    # Apply inflation and representativeness: sigma_eff^2 = (infl*sigma)^2 + sigma_rep^2
    sigma_v = np.asarray(sigma_v, dtype=float) * float(sigma_inflation)
    if sigma_rep and float(sigma_rep) > 0.0:
        sigma_v = np.sqrt(sigma_v**2 + float(sigma_rep)**2)

    # Floor and cap
    if sigma_min is not None:
        sigma_v = np.maximum(sigma_v, float(sigma_min))
    if sigma_max is not None and float(sigma_max) > 0.0:
        sigma_v = np.minimum(sigma_v, float(sigma_max))


    # =========================================================
    # Model anomaly
    #   If MODEL_BASELINE enabled:  sim_anom = hr - mu_model_baseline
    #   Else (legacy):              sim_anom = hr - ensemble_mean
    # calc_v_abs_pf is (nParticles, nCells)
    # =========================================================
    # build_calc_v_from_particles returns (nCells, nParticles) -> transpose to (nParticles, nCells)
    calc_v_abs_pf = calc_v.T
   
    if mu_model_baseline is not None:
        mu_b = np.asarray(mu_model_baseline, dtype=float)

        # If any NaN slipped through (Strict=0), fallback per-cell to ensemble mean
        if np.any(~np.isfinite(mu_b)):
            mu_em = np.nanmean(calc_v_abs_pf, axis=0)
            mu_eff = np.where(np.isfinite(mu_b), mu_b, mu_em)
        else:
            mu_eff = mu_b

        calc_v_anom_pf = calc_v_abs_pf - mu_eff[None, :]
        mu_model = mu_eff
    else:
        calc_v_anom_pf, mu_model = _model_anomaly_from_ensemble(calc_v_abs_pf)

    # Return calc_v in the SAME shape your pipeline expects: (nCells, nParticles)
    calc_v = calc_v_anom_pf.T

    # Debug column for inspection
    try:
        obs_df["mu_model"] = mu_model
    except Exception:
        pass

    # obs_t: use time_max_ict if present (per-cell), else cycle_start_ict, else empty
    if "time_max_ict" in obs_df.columns:
        obs_t = obs_df["time_max_ict"].astype(str).to_numpy()
    elif "cycle_start_ict" in obs_df.columns:
        obs_t = obs_df["cycle_start_ict"].astype(str).to_numpy()
    else:
        obs_t = np.array([""] * len(obs_df), dtype=object)

    return obs_v, obs_t, calc_v, sigma_v, obs_df, particles_used

