# -*- coding: utf-8 -*-
"""
swot_rri_transfer_operator.py

Rollback-safe SWOT -> RRI observation operator for Objective2:Case3.

Purpose
-------
Convert absolute SWOT WSE into the RRI river-state coordinate used by the
particle filter, without changing ParticleFilter.py or the existing anomaly
operator.

Central observation operator
----------------------------
    h*_i = max(clip_min, SWOT_WSE_i + c_i_2023wet)

where c_i_2023wet is the frozen cell-specific median offset developed from the
2023 wet-season SWOT/offline-RRI training pairs.

Heteroscedastic transformed uncertainty
----------------------------------------
    sigma_swot_eff_i^2
        = (inflation * wse_u_i)^2 + sigma_repr^2

    sigma_total_i^2
        = sigma_swot_eff_i^2
          + sigma_transfer_idio_i^2
          + sigma_transfer_common^2

The pass-common transfer variance is deliberately treated diagonally in the
production candidate because the offline GLOBAL experiment showed that the
rank-1 correlated common-mode covariance over-flattened useful hydraulic
information.

Important
---------
- This module does NOT refit the transfer function.
- This module does NOT use gauges.
- This module does NOT modify the historical anomaly operator.
- Cells without frozen 2023-wet transfer support are dropped; there is no
  global fallback offset.
- RRI particle values remain the native absolute river-state `hr` values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import configparser
import json
import os

import numpy as np
import pandas as pd

from swot_obs_operator import (
    SWOTQC,
    ParticleLayout,
    build_calc_v_from_particles,
    particle_hr_path,
)


@dataclass(frozen=True)
class RRITransferConfig:
    parameter_csv: Path
    cell_sigma_csv: Path
    common_sigma_m: float
    strict_support: bool = True
    clip_min_m: float = 0.0


_TABLE_CACHE = {}


def _resolve_config_relative_path(path_text: str, base_dir: Union[str, Path]) -> Path:
    p = Path(str(path_text).strip()).expanduser()
    if not p.is_absolute():
        p = Path(base_dir) / p
    return p.resolve()


def load_rri_transfer_config(
    config_path: Union[str, Path],
    *,
    base_dir: Union[str, Path],
) -> RRITransferConfig:
    """Read [SWOT_RRI_TRANSFER] without touching ReadiniFile.py."""
    config_path = Path(config_path)

    ini = configparser.ConfigParser(interpolation=None)
    with open(config_path, "r", encoding="utf-8") as f:
        ini.read_file(f)

    sec = "SWOT_RRI_TRANSFER"
    if not ini.has_section(sec):
        raise ValueError(
            f"[{sec}] is required when [SWOT] ObservationOperator=RRI_TRANSFER."
        )

    parameter_csv = _resolve_config_relative_path(
        ini.get(sec, "ParameterCSV"), base_dir
    )
    cell_sigma_csv = _resolve_config_relative_path(
        ini.get(sec, "CellSigmaCSV"), base_dir
    )

    common_sigma_m = float(ini.get(sec, "CommonSigma"))
    strict_support = int(ini.get(sec, "StrictSupport", fallback="1")) == 1
    clip_min_m = float(ini.get(sec, "ClipMin", fallback="0.0"))

    if not parameter_csv.exists():
        raise FileNotFoundError(
            f"[SWOT_RRI_TRANSFER] ParameterCSV not found: {parameter_csv}"
        )
    if not cell_sigma_csv.exists():
        raise FileNotFoundError(
            f"[SWOT_RRI_TRANSFER] CellSigmaCSV not found: {cell_sigma_csv}"
        )
    if not np.isfinite(common_sigma_m) or common_sigma_m < 0.0:
        raise ValueError(
            f"[SWOT_RRI_TRANSFER] CommonSigma must be finite and >= 0, got {common_sigma_m}"
        )

    return RRITransferConfig(
        parameter_csv=parameter_csv,
        cell_sigma_csv=cell_sigma_csv,
        common_sigma_m=common_sigma_m,
        strict_support=strict_support,
        clip_min_m=clip_min_m,
    )


def _load_frozen_tables(cfg: RRITransferConfig):
    """Load and cache frozen 2023-wet offset/sigma tables."""
    key = (str(cfg.parameter_csv), str(cfg.cell_sigma_csv))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    p = pd.read_csv(cfg.parameter_csv)
    s = pd.read_csv(cfg.cell_sigma_csv)

    required_p = {"row0", "col0", "offset_2023wet"}
    required_s = {"row0", "col0", "sigma_idio_regularized_m"}

    missing_p = sorted(required_p - set(p.columns))
    missing_s = sorted(required_s - set(s.columns))

    if missing_p:
        raise ValueError(
            f"Transfer parameter CSV missing columns {missing_p}: {cfg.parameter_csv}"
        )
    if missing_s:
        raise ValueError(
            f"Transfer sigma CSV missing columns {missing_s}: {cfg.cell_sigma_csv}"
        )

    p = p.copy()
    s = s.copy()

    p["row0"] = pd.to_numeric(p["row0"], errors="coerce")
    p["col0"] = pd.to_numeric(p["col0"], errors="coerce")
    p["offset_2023wet"] = pd.to_numeric(p["offset_2023wet"], errors="coerce")

    s["row0"] = pd.to_numeric(s["row0"], errors="coerce")
    s["col0"] = pd.to_numeric(s["col0"], errors="coerce")
    s["sigma_idio_regularized_m"] = pd.to_numeric(
        s["sigma_idio_regularized_m"], errors="coerce"
    )

    p = p.dropna(subset=["row0", "col0", "offset_2023wet"]).copy()
    s = s.dropna(subset=["row0", "col0", "sigma_idio_regularized_m"]).copy()

    p["row0"] = p["row0"].astype(int)
    p["col0"] = p["col0"].astype(int)
    s["row0"] = s["row0"].astype(int)
    s["col0"] = s["col0"].astype(int)

    if p.duplicated(["row0", "col0"]).any():
        raise ValueError(f"Duplicate row0/col0 in transfer parameter CSV: {cfg.parameter_csv}")
    if s.duplicated(["row0", "col0"]).any():
        raise ValueError(f"Duplicate row0/col0 in transfer sigma CSV: {cfg.cell_sigma_csv}")

    p_keep = ["row0", "col0", "offset_2023wet"]
    for c in (
        "cell_key",
        "n_train",
        "offset_std_2023wet",
        "rri_hr_min_train",
        "rri_hr_max_train",
        "first_train",
        "last_train",
    ):
        if c in p.columns:
            p_keep.append(c)

    s_keep = ["row0", "col0", "sigma_idio_regularized_m"]
    for c in ("n_train", "sigma_idio_raw_m", "lambda_n"):
        if c in s.columns:
            s_keep.append(c)

    p = p[p_keep].copy()
    s = s[s_keep].copy()

    # Avoid duplicate n_train column names after merge.
    if "n_train" in s.columns:
        s = s.rename(columns={"n_train": "n_train_sigma"})

    _TABLE_CACHE[key] = (p, s)
    return p, s


def _load_cycle_cells_raw(
    cycle_path: Union[str, Path],
    *,
    wse_field: str,
    sigma_field: str,
    qc: SWOTQC,
) -> pd.DataFrame:
    """Load one SWOT cycle and apply the same raw observation QC as anomaly mode."""
    cycle_path = Path(cycle_path)
    cycle = pd.read_parquet(cycle_path)

    if "row" not in cycle.columns or "col" not in cycle.columns:
        raise ValueError(f"SWOT cycle missing row/col: {cycle_path}")
    if wse_field not in cycle.columns:
        raise ValueError(
            f"SWOT cycle missing wse_field='{wse_field}': {cycle_path}"
        )

    if bool(qc.require_reliable_baseline):
        raise ValueError(
            "[SWOT_RRI_TRANSFER] SWOT_RequireReliableBaseline=1 is not compatible "
            "with RRI_TRANSFER because this operator does not use the SWOT anomaly baseline. "
            "Use transfer support/QC instead."
        )

    needed = [
        "row", "col", wse_field, sigma_field,
        "n_nodes", "node_q_median", "node_q_min",
        "wse_std", "wse_u_median",
        "cycle_start_ict", "cycle_end_ict", "time_min_ict", "time_max_ict",
        "cell_id", "lon_cell", "lat_cell",
    ]
    # De-duplicate requested labels before column selection.
    # R_Sigma_Field is normally wse_u_median, which is also retained above as
    # a diagnostic field. Selecting the same label twice creates duplicate
    # DataFrame columns and causes pandas GroupBy median aggregation to fail.
    selected_cols = list(dict.fromkeys(c for c in needed if c in cycle.columns))
    df = cycle[selected_cols].copy()

    df["row"] = pd.to_numeric(df["row"], errors="coerce")
    df["col"] = pd.to_numeric(df["col"], errors="coerce")
    df[wse_field] = pd.to_numeric(df[wse_field], errors="coerce")
    df = df.dropna(subset=["row", "col", wse_field]).copy()

    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)

    if "n_nodes" in df.columns and qc.min_n_nodes is not None:
        df = df[pd.to_numeric(df["n_nodes"], errors="coerce") >= qc.min_n_nodes]
    if qc.max_node_q_median is not None and "node_q_median" in df.columns:
        q = pd.to_numeric(df["node_q_median"], errors="coerce")
        df = df[q <= qc.max_node_q_median]
    if qc.max_wse_std is not None and "wse_std" in df.columns:
        s = pd.to_numeric(df["wse_std"], errors="coerce")
        df = df[s <= qc.max_wse_std]

    df["source_cycle"] = cycle_path.name
    return df.reset_index(drop=True)


def _aggregate_cycle_cells(
    dfs: List[pd.DataFrame],
    *,
    wse_field: str,
    sigma_field: str,
    sigma_agg: str,
) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()

    obs_df = pd.concat(dfs, ignore_index=True)
    if obs_df.empty:
        return obs_df

    sigma_agg = str(sigma_agg or "MEDIAN").strip().upper()
    if sigma_agg not in ("MEDIAN", "MAX"):
        raise ValueError(
            f"[SWOT_RRI_TRANSFER] sigma_agg must be MEDIAN or MAX, got {sigma_agg}"
        )

    if sigma_field not in obs_df.columns:
        raise ValueError(
            f"[SWOT_RRI_TRANSFER] sigma field '{sigma_field}' missing from SWOT parquet. "
            "RRI_TRANSFER requires heteroscedastic wse_u-based uncertainty."
        )

    sigma_op = "median" if sigma_agg == "MEDIAN" else "max"

    agg = {
        wse_field: (wse_field, "median"),
        sigma_field: (sigma_field, sigma_op),
    }

    optional = {
        "cell_id": "first",
        "lon_cell": "first",
        "lat_cell": "first",
        "n_nodes": "max",
        "node_q_median": "median",
        "node_q_min": "median",
        "wse_std": "median",
        "time_min_ict": "max",
        "time_max_ict": "max",
        "cycle_start_ict": "max",
        "cycle_end_ict": "max",
        "source_cycle": "last",
    }

    for c, op in optional.items():
        if c in obs_df.columns:
            agg[c] = (c, op)

    out = (
        obs_df.groupby(["row", "col"], as_index=False)
        .agg(**agg)
        .sort_values(["row", "col"])
        .reset_index(drop=True)
    )
    return out


def _filter_by_reference_hr_mask(
    obs_df: pd.DataFrame,
    *,
    particles_root: Union[str, Path],
    hr_index: int,
    layout: ParticleLayout,
) -> pd.DataFrame:
    """Mirror the existing anomaly operator's reference-particle NODATA guard."""
    if obs_df.empty:
        return obs_df

    ref_path = particle_hr_path(
        Path(particles_root) / "Particle00001",
        hr_index,
        layout,
    )

    if not ref_path.exists():
        print(
            f"[SWOT_RRI_TRANSFER][WARN] Reference HR not found for NODATA mask; "
            f"skipping reference-mask filter: {ref_path}"
        )
        return obs_df

    grid = np.loadtxt(ref_path, dtype=float)
    if grid.ndim != 2:
        raise ValueError(
            f"[SWOT_RRI_TRANSFER] Unexpected reference hr grid shape: {grid.shape}"
        )

    rows = obs_df["row"].to_numpy(dtype=int)
    cols = obs_df["col"].to_numpy(dtype=int)
    nr, nc = grid.shape

    in_bounds = (
        (rows >= 0) & (rows < nr)
        & (cols >= 0) & (cols < nc)
    )

    valid = np.zeros(len(obs_df), dtype=bool)
    valid[in_bounds] = (
        np.isfinite(grid[rows[in_bounds], cols[in_bounds]])
        & (grid[rows[in_bounds], cols[in_bounds]] != layout.nodata_value)
    )

    dropped = int((~valid).sum())
    if dropped > 0:
        print(
            f"[SWOT_RRI_TRANSFER][WARN] Dropped {dropped}/{len(obs_df)} cells "
            f"outside active RRI hr mask using {ref_path}"
        )

    return obs_df.loc[valid].reset_index(drop=True)


def make_pf_inputs_from_swot_rri_transfer(
    *,
    cycle_path: Union[str, Path, List[Union[str, Path]]],
    particles_root: Union[str, Path],
    hr_index: int,
    transfer_cfg: RRITransferConfig,
    wse_field: str = "wse_median",
    qc: Optional[SWOTQC] = None,
    layout: Optional[ParticleLayout] = None,
    sigma_field: str = "wse_u_median",
    sigma_agg: str = "MEDIAN",
    sigma_min: float = 0.05,
    sigma_max: float = -1.0,
    sigma_inflation: float = 1.0,
    sigma_rep: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Build PF inputs in absolute RRI `hr` state space.

    Returns exactly the same six-object interface as make_pf_inputs_from_swot():
      obs_v, obs_t, calc_v, sigma_v, obs_df, particles_used

    Shapes:
      obs_v   : (nCells,)
      calc_v  : (nCells, nParticles)
      sigma_v : (nCells,)
    """
    qc = qc or SWOTQC()
    layout = layout or ParticleLayout()

    if isinstance(cycle_path, (list, tuple)):
        cycle_paths = [Path(p) for p in cycle_path]
    else:
        cycle_paths = [Path(cycle_path)]

    raw = [
        _load_cycle_cells_raw(
            cp,
            wse_field=wse_field,
            sigma_field=sigma_field,
            qc=qc,
        )
        for cp in cycle_paths
    ]

    obs_df = _aggregate_cycle_cells(
        raw,
        wse_field=wse_field,
        sigma_field=sigma_field,
        sigma_agg=sigma_agg,
    )

    if obs_df.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=object),
            np.empty((0, 0), dtype=float),
            np.array([], dtype=float),
            obs_df,
            [],
        )

    params, cell_sigma = _load_frozen_tables(transfer_cfg)

    p = params.rename(columns={"row0": "row", "col0": "col"})
    s = cell_sigma.rename(columns={"row0": "row", "col0": "col"})

    n_before_support = len(obs_df)
    obs_df = obs_df.merge(p, on=["row", "col"], how="left")
    obs_df = obs_df.merge(s, on=["row", "col"], how="left")

    support_ok = (
        np.isfinite(pd.to_numeric(obs_df["offset_2023wet"], errors="coerce"))
        & np.isfinite(
            pd.to_numeric(obs_df["sigma_idio_regularized_m"], errors="coerce")
        )
    )

    missing_support = int((~support_ok).sum())
    if missing_support > 0:
        print(
            f"[SWOT_RRI_TRANSFER] Strict 2023-wet support: dropping "
            f"{missing_support}/{n_before_support} current SWOT cells with no frozen "
            f"offset and/or transfer sigma. No fallback is used."
        )

    # The validated operator has no fallback. StrictSupport=0 is intentionally
    # rejected rather than silently inventing one.
    if (not transfer_cfg.strict_support) and missing_support > 0:
        raise ValueError(
            "[SWOT_RRI_TRANSFER] StrictSupport=0 requested, but no validated fallback "
            "operator exists. Set StrictSupport=1."
        )

    obs_df = obs_df.loc[support_ok].reset_index(drop=True)

    if obs_df.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=object),
            np.empty((0, 0), dtype=float),
            np.array([], dtype=float),
            obs_df,
            [],
        )

    # Require valid heteroscedastic SWOT uncertainty before expensive particle sampling.
    sigma_raw = pd.to_numeric(obs_df[sigma_field], errors="coerce").to_numpy(float)
    sigma_ok = np.isfinite(sigma_raw) & (sigma_raw >= 0.0)
    if not np.all(sigma_ok):
        dropped = int((~sigma_ok).sum())
        print(
            f"[SWOT_RRI_TRANSFER][WARN] Dropping {dropped}/{len(obs_df)} cells with "
            f"invalid {sigma_field}."
        )
        obs_df = obs_df.loc[sigma_ok].reset_index(drop=True)

    if obs_df.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=object),
            np.empty((0, 0), dtype=float),
            np.array([], dtype=float),
            obs_df,
            [],
        )

    obs_df = _filter_by_reference_hr_mask(
        obs_df,
        particles_root=particles_root,
        hr_index=hr_index,
        layout=layout,
    )

    if obs_df.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=object),
            np.empty((0, 0), dtype=float),
            np.array([], dtype=float),
            obs_df,
            [],
        )

    # Absolute RRI river-state particle matrix (nCells, nParticles).
    calc_v, particles_used = build_calc_v_from_particles(
        particles_root=particles_root,
        hr_index=hr_index,
        obs_df=obs_df,
        layout=layout,
    )

    calc_v = np.asarray(calc_v, dtype=float)
    if np.any(~np.isfinite(calc_v)):
        raise ValueError("[SWOT_RRI_TRANSFER] Non-finite particle hr encountered.")

    # Central transfer mean.
    wse = pd.to_numeric(obs_df[wse_field], errors="coerce").to_numpy(float)
    offset = pd.to_numeric(obs_df["offset_2023wet"], errors="coerce").to_numpy(float)
    obs_v_raw = wse + offset
    obs_v = np.maximum(float(transfer_cfg.clip_min_m), obs_v_raw)

    # Heteroscedastic uncertainty.
    sigma_raw = pd.to_numeric(obs_df[sigma_field], errors="coerce").to_numpy(float)
    sigma_swot_eff = np.sqrt(
        (float(sigma_inflation) * sigma_raw) ** 2
        + float(sigma_rep) ** 2
    )

    # Preserve legacy sigma floor/cap semantics on the SWOT measurement component.
    if sigma_min is not None:
        sigma_swot_eff = np.maximum(sigma_swot_eff, float(sigma_min))
    if sigma_max is not None and float(sigma_max) > 0.0:
        sigma_swot_eff = np.minimum(sigma_swot_eff, float(sigma_max))

    sigma_idio = pd.to_numeric(
        obs_df["sigma_idio_regularized_m"], errors="coerce"
    ).to_numpy(float)

    if np.any(~np.isfinite(sigma_idio)) or np.any(sigma_idio < 0.0):
        raise ValueError(
            "[SWOT_RRI_TRANSFER] Invalid regularized transfer sigma after support filter."
        )

    sigma_common = float(transfer_cfg.common_sigma_m)

    sigma_v = np.sqrt(
        sigma_swot_eff ** 2
        + sigma_idio ** 2
        + sigma_common ** 2
    )

    if np.any(~np.isfinite(sigma_v)) or np.any(sigma_v <= 0.0):
        raise ValueError("[SWOT_RRI_TRANSFER] Invalid transformed total sigma.")

    # Diagnostics/provenance columns.
    obs_df = obs_df.copy()
    obs_df["observation_operator"] = "RRI_TRANSFER"
    obs_df["transfer_wse_used_m"] = wse
    obs_df["transfer_offset_2023wet_m"] = offset
    obs_df["obs_transfer_hr_raw_m"] = obs_v_raw
    obs_df["obs_transfer_hr_used_m"] = obs_v
    obs_df["transfer_clip_min_m"] = float(transfer_cfg.clip_min_m)
    obs_df["sigma_swot_raw_m"] = sigma_raw
    obs_df["sigma_swot_effective_m"] = sigma_swot_eff
    obs_df["sigma_transfer_idio_m"] = sigma_idio
    obs_df["sigma_transfer_common_m"] = sigma_common
    obs_df["sigma_total_used_m"] = sigma_v

    if "time_max_ict" in obs_df.columns:
        obs_t = obs_df["time_max_ict"].astype(str).to_numpy()
    elif "cycle_start_ict" in obs_df.columns:
        obs_t = obs_df["cycle_start_ict"].astype(str).to_numpy()
    else:
        obs_t = np.array([""] * len(obs_df), dtype=object)

    return obs_v, obs_t, calc_v, sigma_v, obs_df, particles_used


def _safe_mkdir(path: Union[str, Path]):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_swot_rri_transfer_diagnostics(
    *,
    out_dir: Union[str, Path],
    present_time_txt: str,
    obs_df_swot: pd.DataFrame,
    obs_v_swot: np.ndarray,
    calc_v_swot: np.ndarray,
    sigma_v_swot: Optional[np.ndarray],
    particles_used,
    BestPn_1based: Optional[int] = None,
    Likelihood: Optional[np.ndarray] = None,
):
    """Save transfer-state diagnostics without labeling absolute hr as anomaly."""
    _safe_mkdir(out_dir)
    out_dir = Path(out_dir)

    obs_v_swot = np.asarray(obs_v_swot, dtype=float)
    calc_v_swot = np.asarray(calc_v_swot, dtype=float)

    cells = obs_df_swot.copy()
    cells["obs_v_used"] = obs_v_swot
    cells["observation_operator"] = "RRI_TRANSFER"

    if sigma_v_swot is not None:
        cells["sigma_v_used"] = np.asarray(sigma_v_swot, dtype=float)

    # Keep the generic *_SWOT_cells.csv name for downstream inventory/audits.
    cells_csv = out_dir / f"{present_time_txt}_SWOT_cells.csv"
    cells.to_csv(cells_csv, index=False)

    mat_npy = out_dir / f"{present_time_txt}_SWOT_calc_v_transfer_hr.npy"
    np.save(mat_npy, calc_v_swot)

    mapping = {
        "present_time_txt": str(present_time_txt),
        "observation_operator": "RRI_TRANSFER",
        "calc_v_shape": list(calc_v_swot.shape),
        "particles_used": list(particles_used) if particles_used is not None else None,
        "note": (
            "calc_v columns are particle order; values are absolute native RRI hr, "
            "not anomalies"
        ),
    }
    with open(out_dir / f"{present_time_txt}_SWOT_particles_used.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    if Likelihood is not None:
        w = np.asarray(Likelihood, dtype=float)
        ws = float(np.sum(w))
        if ws > 0.0:
            w = w / ws
        ess = float(1.0 / np.sum(w*w)) if np.sum(w*w) > 0.0 else float("nan")
    else:
        w = None
        ess = None

    pf_summary = {
        "present_time_txt": str(present_time_txt),
        "observation_operator": "RRI_TRANSFER",
        "BestPn_1based": int(BestPn_1based) if BestPn_1based is not None else None,
        "ESS": ess,
        "weights_norm": w.tolist() if w is not None else None,
    }
    with open(out_dir / f"{present_time_txt}_SWOT_pf_summary.json", "w", encoding="utf-8") as f:
        json.dump(pf_summary, f, indent=2)

    if BestPn_1based is not None and calc_v_swot.ndim == 2 and calc_v_swot.shape[0] > 0:
        best_idx = int(BestPn_1based) - 1
        if 0 <= best_idx < calc_v_swot.shape[1]:
            sim_best = calc_v_swot[:, best_idx]
            innovation = sim_best - obs_v_swot

            out = obs_df_swot.copy()
            out["obs_transfer_hr_used_m"] = obs_v_swot
            out["sim_rri_hr_best_m"] = sim_best
            out["innovation_best_m"] = innovation
            out["BestPn_1based"] = int(BestPn_1based)

            if sigma_v_swot is not None:
                sig = np.asarray(sigma_v_swot, dtype=float)
                out["sigma_total_used_m"] = sig
                out["weighted_innovation_best"] = innovation / np.maximum(sig, 1e-12)

            out.to_csv(
                out_dir / f"{present_time_txt}_SWOT_transfer_innovation_best.csv",
                index=False,
            )
