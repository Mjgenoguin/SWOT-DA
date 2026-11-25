# -*- coding: utf-8 -*-
"""
MultiCoreTrueNSGA2_ThreeStations_DepthProfile_Dedup.py

Three-station NSGA-II calibration of RRI river depth-profile geometry.

This driver preserves the working three-station NSGA-II, multiprocessing,
checkpointing, objective aggregation, and duplicate-evaluation logic of the
existing calibration script, but changes what an individual represents.

Genes
-----
1. upper_blend
   Additional bed-profile blend on NAN001 -> NAN002 relative to the supplied
   D50 seed geometry. 0 reproduces the D50 seed; +1 moves the reach to a
   straight endpoint-to-endpoint bed profile; -1 approximately reconstructs
   the pre-U50 profile because the D50 seed used 50% smoothing.

2. middle_blend
   Same definition on NAN002 -> NAN003. The D50 seed contains the previous
   S50 treatment, so 0 is S50, +1 is fully straight, and -1 approximately
   reconstructs the pre-S50 (AnchorRaw) profile.

3. downstream_slope_factor
   Multiplies the D50 main-stem bed drop below NAN003 over the adaptive core
   reach. 1 reproduces D50, values below 1 flatten the downstream bed further,
   and values above 1 steepen it. The adjustment is tapered back to the D50
   seed at the downstream end.

Connected streams
-----------------
For every changed main-stem confluence, the incremental bed correction relative
to the D50 seed is propagated upstream through connected tributary river cells
and tapered smoothly to zero. Tributary corrections are deterministic; they
are not additional NSGA-II genes.

Safety / rollback
-----------------
- The base RRI_Input.txt is never edited by this script.
- Each process-specific worker receives its own copied RRI model folder.
- The worker RRI_Input.txt is patched once to read ./riv/depth_NSGA2_working.txt.
- Every candidate depth raster is generated only inside that worker folder.
- The exact NAN001, NAN002, and NAN003 depth cells are preserved.
- Candidate river depths must remain positive.
- The main-stem downstream counter-slope cannot exceed the seed geometry by
  more than the configured tolerance.
- The known D50 geometry is the exact seed individual [0, 0, 1].
- Duplicate individuals reuse cached fitness and are not re-run in RRI.

The script does not modify Manning n or land-use parameters.
"""
from __future__ import annotations

import copy
from collections import deque
import json
import os
import pickle
import re
import shutil
import signal
import stat
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

ROOT_DIR = Path(r"/home/WEM-st124985/Objective1")
PARTICLE_DIR = Path(r"/home/WEM-st124985/Objective1/Nan_5LULC")
RRI_INPUT = PARTICLE_DIR / "RRI_Input.txt"
RRI_EXE_CANDS = [
    "0_rri_1_4_2_6_NonOMP_dt600s",
    "RRI_Run",
]

CALIBRATION_MODE = "depth_profile"
RUN_TAG = "DepthProfile_Nan5LULC_184d"

STATION_IDS: Tuple[str, ...] = ("NAN001", "NAN002", "NAN003")
STATION_WEIGHTS: Dict[str, float] = {
    "NAN001": 1.0,
    "NAN002": 1.0,
    "NAN003": 1.0,
}
MIN_PAIRS_PER_STATION = 30

# RRI location.txt is 1-based. Internal raster indices are zero-based.
LOCATION_FILE = PARTICLE_DIR / "location.txt"
STATION_GRID_LOCATIONS: Dict[str, Tuple[int, int]] = {
    "NAN001": (88, 331),
    "NAN002": (117, 332),
    "NAN003": (131, 326),
}
AUTO_WRITE_LOCATION_FILE = True
LOCATION_BACKUP_FILE = PARTICLE_DIR / "location_single_station_backup.txt"

REFRESH_WORKER_DIRS = True

# -----------------------------------------------------------------------------
# Observed water-level input
# -----------------------------------------------------------------------------
OBS_INPUT_MODE = "single_table"

OBS_STATION_FILES: Dict[str, Dict[str, object]] = {
    "NAN001": {
        "path": ROOT_DIR / "Observed/NAN001_WL_HII.txt",
        "sep": "\t",
        "has_header": True,
        "datetime_col": 0,
        "value_col": 1,
        "datetime_format": "%m/%d/%Y %H:%M:%S",
    },
    "NAN002": {
        "path": ROOT_DIR / "Observed/NAN002_WL_HII.txt",
        "sep": "\t",
        "has_header": True,
        "datetime_col": 0,
        "value_col": 1,
        "datetime_format": "%m/%d/%Y %H:%M:%S",
    },
    "NAN003": {
        "path": ROOT_DIR / "Observed/NAN003_WL_HII.txt",
        "sep": "\t",
        "has_header": True,
        "datetime_col": 0,
        "value_col": 1,
        "datetime_format": "%m/%d/%Y %H:%M:%S",
    },
}

OBS_SINGLE_TABLE = {
    "path": ROOT_DIR / "Observed/Nan_WL_HII_2023-2025_RID.txt",
    "sep": "\t",
    "has_header": True,
    "datetime_col": "datetime",
    "station_value_cols": {
        "NAN001": "NAN001",
        "NAN002": "NAN002",
        "NAN003": "NAN003",
    },
    "datetime_format": None,
}

# -----------------------------------------------------------------------------
# Simulated RRI hydrograph input
# -----------------------------------------------------------------------------
SIM_OUTPUT_MODE = "multi_column"

SIM_MULTI_FILE = "hydro_hr.txt"
SIM_TIME_COL = 0
SIM_STATION_VALUE_COLS: Dict[str, int] = {
    "NAN001": 1,
    "NAN002": 2,
    "NAN003": 3,
}

SIM_STATION_FILES: Dict[str, str] = {
    "NAN001": "hydro_hr_NAN001.txt",
    "NAN002": "hydro_hr_NAN002.txt",
    "NAN003": "hydro_hr_NAN003.txt",
}
SIM_SEPARATE_TIME_COL = 0
SIM_SEPARATE_VALUE_COL = 1

# -----------------------------------------------------------------------------
# Temporal matching and scoring
# -----------------------------------------------------------------------------
MATCH_MODE = "daily"

SIM_START = pd.Timestamp("2023-05-01 00:00")
SCORE_START = pd.Timestamp("2023-05-01 01:00")
SCORE_END = pd.Timestamp("2023-10-31 01:00")

# Match the existing 184-pair daily evaluation convention used for the
# May 1-Oct 31, 2023 benchmark runs.
DAILY_PICK_HOUR = 1
DAILY_PICK_MINUTE = 0
DAILY_PICK_SECOND = 0
DAILY_TOLERANCE = "120min"
HOURLY_TOLERANCE = "5min"

# -----------------------------------------------------------------------------
# Objective aggregation
# -----------------------------------------------------------------------------
ERROR_NORMALIZATION = "std"

# -----------------------------------------------------------------------------
# Geometry calibration
# -----------------------------------------------------------------------------
SEED_DEPTH_NAME = "depth_S50_U50_D50_AdaptiveConfluence.txt"
WORKING_DEPTH_NAME = "depth_NSGA2_working.txt"

ADEM_REL = Path("topo/adem_2s.asc")
ADIR_REL = Path("topo/adir_2s.asc")
WIDTH_REL = Path("riv/width.txt")
SEED_DEPTH_REL = Path("riv") / SEED_DEPTH_NAME
WORKING_DEPTH_REL = Path("riv") / WORKING_DEPTH_NAME

# D8 convention used by the present Upper Nan model.
D8_OFFSETS: Dict[int, Tuple[int, int]] = {
    1:   (0, 1),
    2:   (1, 1),
    4:   (1, 0),
    8:   (1, -1),
    16:  (0, -1),
    32:  (-1, -1),
    64:  (-1, 0),
    128: (-1, 1),
}

DESIRED_DOWNSTREAM_CORE_KM = 2.0
DESIRED_DOWNSTREAM_TAPER_KM = 2.0
ADAPTIVE_CORE_FRACTION = 0.60

TRIBUTARY_TAPER_KM = 2.0
MIN_CONFLUENCE_CORRECTION_M = 0.01
MAINSTEM_UPHILL_ALLOWANCE_M = 5.0

GENE_KEYS: Tuple[str, ...] = (
    "upper_blend",
    "middle_blend",
    "downstream_slope_factor",
)

# 0,0,1 reproduces the supplied D50 seed geometry.
INITIAL_VECTOR = np.asarray([0.0, 0.0, 1.0], dtype=float)

BOUNDS = {
    # -1 ~= undo the existing 50% smoothing; +1 = fully straight profile.
    "upper_blend": (-1.0, 2.5),
    "middle_blend": (-1.0, 2.5),

    # 1.0 = supplied D50 downstream slope; <1 flatter; >1 steeper.
    "downstream_slope_factor": (-3, 1.50),
}

# -----------------------------------------------------------------------------
# Logs, archives, checkpoint, and runtime
# -----------------------------------------------------------------------------
ARCHIVE_DIR = PARTICLE_DIR / f"Archive_ThreeStations_{RUN_TAG}_Dedup"
JSONL_LOG = ARCHIVE_DIR / f"GA_run_log_ThreeStations_{RUN_TAG}_Dedup.jsonl"
TSV_LOG = ROOT_DIR / f"evaluation_log_ThreeStations_{RUN_TAG}_Dedup.tsv"
CHECKPOINT_DIR = ROOT_DIR / "Checkpoints"
CHECKPOINT_FILE = (
    CHECKPOINT_DIR / f"ga_checkpoint_ThreeStations_{RUN_TAG}_Dedup.pkl"
)

GA_MAX_SECONDS = int(os.getenv("GA_MAX_SECONDS", "0"))
GRACE_SECONDS = int(os.getenv("GA_GRACE_SECONDS", "300"))
DELETE_CHECKPOINT_ON_SUCCESS = bool(
    int(os.getenv("GA_DELETE_CKPT_ON_SUCCESS", "0"))
)

PREFLIGHT_ONLY = bool(int(os.getenv("GA_PREFLIGHT_ONLY", "0")))
ARCHIVE_ALL_DEPTHS = bool(int(os.getenv("GA_ARCHIVE_DEPTHS", "0")))

# -----------------------------------------------------------------------------
# NSGA-II settings
# -----------------------------------------------------------------------------
# Small first-stage search because each RRI evaluation is expensive.
# All can be overridden from sbatch without editing this file.
POP_SIZE = int(os.getenv("GA_POP_SIZE", "10"))
GENERATIONS = int(os.getenv("GA_GENERATIONS", "100"))
CX_PROB = float(os.getenv("GA_CX_PROB", "0.6"))
MUT_PROB = float(os.getenv("GA_MUT_PROB", "0.5"))
MUT_SIGMA = float(os.getenv("GA_MUT_SIGMA", "0.08"))
RNG_SEED = int(os.getenv("GA_RNG_SEED", "42"))
TOURNAMENT_SIZE = 2

USE_SEEDING = bool(int(os.getenv("GA_USE_SEEDING", "1")))
SEED_LOCAL_FRAC = float(os.getenv("GA_SEED_LOCAL_FRAC", "0.75"))
SEED_LOCAL_SCALE = float(os.getenv("GA_SEED_LOCAL_SCALE", "0.08"))

PARALLEL_WORKERS = int(os.getenv("GA_PARALLEL_WORKERS", "10"))
WORKERS_ROOT = (
    PARTICLE_DIR.parent
    / f"Workers_Nan_5LULC_ThreeStations_{RUN_TAG}"
)
CLEAN_HYDROS_BEFORE_RUN = True

FINAL_BEST_DEPTH = (
    PARTICLE_DIR / "riv" / "depth_NSGA2_DepthProfile_Best.txt"
)
FINAL_PARETO_TSV = (
    ARCHIVE_DIR / "Final_Pareto_DepthProfile.tsv"
)

# =============================================================================
# GENERAL HELPERS
# =============================================================================

FLOAT_TOKEN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?")


def resolve_dataframe_column(dataframe: pd.DataFrame, spec: object) -> pd.Series:
    """Return a column by header name or zero-based integer position."""
    if isinstance(spec, str):
        if spec not in dataframe.columns:
            raise KeyError(
                f"Column {spec!r} was not found. Available columns: "
                f"{list(dataframe.columns)!r}"
            )
        return dataframe[spec]

    index = int(spec)
    if index < 0 or index >= dataframe.shape[1]:
        raise IndexError(
            f"Column index {index} is outside the table with "
            f"{dataframe.shape[1]} columns"
        )
    return dataframe.iloc[:, index]


def expected_location_text() -> str:
    lines = []
    for station in STATION_IDS:
        row, col = STATION_GRID_LOCATIONS[station]
        lines.append(f"{station}\t\t{row}\t{col}\n")
    return "".join(lines)


def prepare_three_station_location_file() -> None:
    """Back up the existing location file once, then enforce three stations."""
    expected = expected_location_text()
    current = ""
    if LOCATION_FILE.exists():
        current = LOCATION_FILE.read_text(encoding="utf-8", errors="ignore")

    def normalized(text: str) -> List[str]:
        return [" ".join(line.split()) for line in text.splitlines() if line.strip()]

    if normalized(current) == normalized(expected):
        return

    if not AUTO_WRITE_LOCATION_FILE:
        raise RuntimeError(
            f"{LOCATION_FILE} does not contain NAN001, NAN002, and NAN003 in "
            "the required order. Replace it with the supplied three-station "
            "location file or set AUTO_WRITE_LOCATION_FILE=True."
        )

    if current.strip() and not LOCATION_BACKUP_FILE.exists():
        LOCATION_BACKUP_FILE.write_text(current, encoding="utf-8")

    LOCATION_FILE.write_text(expected, encoding="utf-8")
    print(f"[LOCATION] Wrote three-station location file: {LOCATION_FILE}")


def ensure_paths() -> None:
    required = [
        PARTICLE_DIR,
        RRI_INPUT,
        PARTICLE_DIR / ADEM_REL,
        PARTICLE_DIR / ADIR_REL,
        PARTICLE_DIR / WIDTH_REL,
        PARTICLE_DIR / SEED_DEPTH_REL,
    ]
    for path in required:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    if OBS_INPUT_MODE == "separate_files":
        for station in STATION_IDS:
            config = OBS_STATION_FILES.get(station)
            if config is None:
                raise KeyError(f"No observation configuration for {station}")
            path = Path(config["path"])
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing observed file for {station}: {path}"
                )
    elif OBS_INPUT_MODE == "single_table":
        path = Path(OBS_SINGLE_TABLE["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Missing observed multi-station table: {path}"
            )
    else:
        raise ValueError(
            f"OBS_INPUT_MODE must be 'separate_files' or 'single_table', "
            f"got {OBS_INPUT_MODE!r}"
        )

    if SIM_OUTPUT_MODE not in {"multi_column", "separate_files"}:
        raise ValueError(
            f"SIM_OUTPUT_MODE must be 'multi_column' or 'separate_files', "
            f"got {SIM_OUTPUT_MODE!r}"
        )

    if POP_SIZE < 4:
        raise ValueError("GA_POP_SIZE must be at least 4")
    if GENERATIONS < 1:
        raise ValueError("GA_GENERATIONS must be at least 1")
    if PARALLEL_WORKERS < 1:
        raise ValueError("GA_PARALLEL_WORKERS must be at least 1")

    missing_weights = [s for s in STATION_IDS if s not in STATION_WEIGHTS]
    if missing_weights:
        raise KeyError(f"Missing station weights for: {missing_weights}")

    missing_locations = [
        s for s in STATION_IDS
        if s not in STATION_GRID_LOCATIONS
    ]
    if missing_locations:
        raise KeyError(f"Missing grid locations for: {missing_locations}")

    if any(float(STATION_WEIGHTS[s]) <= 0.0 for s in STATION_IDS):
        raise ValueError("All STATION_WEIGHTS must be positive")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    WORKERS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines(True)


def write_lines_atomic(path: Path, lines: Sequence[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(row: dict) -> None:
    with open(JSONL_LOG, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, allow_nan=True) + "\n")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def in_bounds(name: str, value: float) -> float:
    lower, upper = BOUNDS[name]
    return float(clamp(value, lower, upper))


def normalized_station_weights() -> Dict[str, float]:
    total = float(sum(float(STATION_WEIGHTS[s]) for s in STATION_IDS))
    if total <= 0.0:
        raise ValueError("Station-weight sum must be positive")
    return {s: float(STATION_WEIGHTS[s]) / total for s in STATION_IDS}


# =============================================================================
# DEPTH-PROFILE GEOMETRY GENERATION
# =============================================================================


def read_esri_ascii(path: Path) -> Tuple[dict, List[str], np.ndarray]:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        header_lines = [
            file.readline().rstrip("\n\r")
            for _ in range(6)
        ]
        header: Dict[str, float] = {}
        for line in header_lines:
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid ESRI header line in {path}: {line!r}"
                )
            header[parts[0].lower()] = float(parts[1])
        data = np.loadtxt(file, dtype=float)

    header["ncols"] = int(header["ncols"])
    header["nrows"] = int(header["nrows"])

    expected_shape = (
        int(header["nrows"]),
        int(header["ncols"]),
    )
    if data.shape != expected_shape:
        raise ValueError(
            f"{path}: expected raster shape {expected_shape}, "
            f"found {data.shape}"
        )
    return header, header_lines, data


def check_same_grid(reference: dict, comparison: dict, name: str) -> None:
    for key in (
        "ncols",
        "nrows",
        "xllcorner",
        "yllcorner",
        "cellsize",
    ):
        if not np.isclose(
            float(reference[key]),
            float(comparison[key]),
        ):
            raise ValueError(
                f"Grid mismatch for {name}: {key}: "
                f"{reference[key]} vs {comparison[key]}"
            )


def next_d8_cell(
    adir_grid: np.ndarray,
    cell: Tuple[int, int],
) -> Tuple[Optional[Tuple[int, int]], int]:
    row, col = cell
    direction = int(round(float(adir_grid[row, col])))
    if direction not in D8_OFFSETS:
        return None, direction
    drow, dcol = D8_OFFSETS[direction]
    return (row + drow, col + dcol), direction


def trace_to_target(
    adir_grid: np.ndarray,
    start_cell: Tuple[int, int],
    target_cell: Tuple[int, int],
    max_steps: int = 10000,
) -> List[Tuple[int, int]]:
    current = tuple(start_cell)
    target = tuple(target_cell)
    path: List[Tuple[int, int]] = []
    visited = set()
    nrows, ncols = adir_grid.shape

    for _ in range(max_steps):
        if current in visited:
            raise RuntimeError(f"D8 loop detected at {current}")
        row, col = current
        if not (0 <= row < nrows and 0 <= col < ncols):
            raise RuntimeError(f"D8 path left raster at {current}")

        visited.add(current)
        path.append(current)

        if current == target:
            return path

        nxt, direction = next_d8_cell(adir_grid, current)
        if nxt is None:
            raise RuntimeError(
                f"Target {target} not reached: D8 terminated at "
                f"{current} with direction {direction}"
            )
        current = nxt

    raise RuntimeError(
        f"Target {target} not reached from {start_cell} "
        f"within {max_steps} steps"
    )


def trace_downstream_with_terminal(
    adir_grid: np.ndarray,
    start_cell: Tuple[int, int],
    river_mask: Optional[np.ndarray] = None,
    max_steps: int = 10000,
) -> Tuple[List[Tuple[int, int]], dict]:
    current = tuple(start_cell)
    path: List[Tuple[int, int]] = []
    visited = set()
    nrows, ncols = adir_grid.shape

    terminal = {
        "reason": None,
        "cell": None,
        "direction": None,
    }

    for _ in range(max_steps):
        if current in visited:
            terminal = {
                "reason": "D8_LOOP",
                "cell": current,
                "direction": int(round(float(adir_grid[current]))),
            }
            break

        row, col = current
        if not (0 <= row < nrows and 0 <= col < ncols):
            terminal = {
                "reason": "LEFT_RASTER",
                "cell": current,
                "direction": None,
            }
            break

        # RRI defines river cells from width > 0. Once the D8 path leaves
        # that mask, stop the downstream geometry-control reach rather than
        # treating non-river raster cells as part of the calibrated channel.
        if river_mask is not None and not bool(river_mask[row, col]):
            terminal = {
                "reason": "LEFT_RIVER_MASK",
                "cell": current,
                "direction": int(round(float(adir_grid[current]))),
            }
            break

        visited.add(current)
        path.append(current)

        nxt, direction = next_d8_cell(adir_grid, current)
        if nxt is None:
            terminal = {
                "reason": "TERMINAL_OR_INVALID_D8",
                "cell": current,
                "direction": direction,
            }
            break

        nrow, ncol = nxt
        if not (0 <= nrow < nrows and 0 <= ncol < ncols):
            terminal = {
                "reason": "NEXT_CELL_OUTSIDE_RASTER",
                "cell": current,
                "direction": direction,
            }
            break
        current = nxt
    else:
        terminal = {
            "reason": "MAX_STEPS_REACHED",
            "cell": current,
            "direction": None,
        }

    return path, terminal


def raster_cell_center(
    header: dict,
    row0: int,
    col0: int,
) -> Tuple[float, float]:
    nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])
    lon = float(header["xllcorner"]) + (col0 + 0.5) * cellsize
    lat = (
        float(header["yllcorner"])
        + (nrows - row0 - 0.5) * cellsize
    )
    return lon, lat


def haversine_km(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    radius_km = 6371.0088
    lon1_r, lat1_r, lon2_r, lat2_r = map(
        np.radians,
        [lon1, lat1, lon2, lat2],
    )
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r)
        * np.cos(lat2_r)
        * np.sin(dlon / 2.0) ** 2
    )
    return float(
        radius_km
        * 2.0
        * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    )


def cell_distance_km(
    header: dict,
    cell_a: Tuple[int, int],
    cell_b: Tuple[int, int],
) -> float:
    lon1, lat1 = raster_cell_center(
        header, cell_a[0], cell_a[1]
    )
    lon2, lat2 = raster_cell_center(
        header, cell_b[0], cell_b[1]
    )
    return haversine_km(lon1, lat1, lon2, lat2)


def path_chainage_km(
    header: dict,
    cells: Sequence[Tuple[int, int]],
) -> np.ndarray:
    chainage = np.zeros(len(cells), dtype=float)
    for index in range(1, len(cells)):
        chainage[index] = (
            chainage[index - 1]
            + cell_distance_km(
                header,
                cells[index - 1],
                cells[index],
            )
        )
    return chainage


def upstream_river_neighbors(
    target_cell: Tuple[int, int],
    adir_grid: np.ndarray,
    river_mask: np.ndarray,
) -> List[Tuple[int, int]]:
    target_row, target_col = target_cell
    nrows, ncols = adir_grid.shape
    output: List[Tuple[int, int]] = []

    for drow in (-1, 0, 1):
        for dcol in (-1, 0, 1):
            if drow == 0 and dcol == 0:
                continue
            row = target_row + drow
            col = target_col + dcol
            if not (0 <= row < nrows and 0 <= col < ncols):
                continue
            if not bool(river_mask[row, col]):
                continue
            nxt, _direction = next_d8_cell(
                adir_grid,
                (row, col),
            )
            if nxt == target_cell:
                output.append((row, col))

    return output


def tributary_cells_for_confluence(
    junction_cell: Tuple[int, int],
    mainstem_set: set,
    adir_grid: np.ndarray,
    river_mask: np.ndarray,
    header: dict,
    max_distance_km: float,
) -> List[Tuple[Tuple[int, int], float]]:
    initial = [
        cell
        for cell in upstream_river_neighbors(
            junction_cell,
            adir_grid,
            river_mask,
        )
        if cell not in mainstem_set
    ]

    if not initial:
        return []

    queue = deque()
    visited = set()
    output: List[Tuple[Tuple[int, int], float]] = []

    for cell in initial:
        queue.append(
            (
                cell,
                cell_distance_km(
                    header,
                    junction_cell,
                    cell,
                ),
            )
        )

    while queue:
        cell, distance_km = queue.popleft()
        if cell in visited:
            continue
        visited.add(cell)

        if distance_km > max_distance_km:
            continue

        fraction = min(
            max(distance_km / max_distance_km, 0.0),
            1.0,
        )
        weight = 0.5 * (
            1.0 + np.cos(np.pi * fraction)
        )
        output.append((cell, float(weight)))

        for upstream_cell in upstream_river_neighbors(
            cell,
            adir_grid,
            river_mask,
        ):
            if upstream_cell in mainstem_set:
                continue
            edge_km = cell_distance_km(
                header,
                cell,
                upstream_cell,
            )
            queue.append(
                (
                    upstream_cell,
                    distance_km + edge_km,
                )
            )

    return output


def _station_cells_zero_based() -> Dict[str, Tuple[int, int]]:
    return {
        station: (row1 - 1, col1 - 1)
        for station, (row1, col1)
        in STATION_GRID_LOCATIONS.items()
    }


def _max_downstream_uphill_rise(
    bed_grid: np.ndarray,
    path_cells: Sequence[Tuple[int, int]],
) -> float:
    if len(path_cells) < 2:
        return 0.0
    values = np.asarray(
        [float(bed_grid[cell]) for cell in path_cells],
        dtype=float,
    )
    rises = values[1:] - values[:-1]
    return float(max(0.0, np.max(rises)))


def build_geometry_context(base_dir: Path) -> dict:
    base_dir = Path(base_dir)

    seed_path = base_dir / SEED_DEPTH_REL
    adem_path = base_dir / ADEM_REL
    adir_path = base_dir / ADIR_REL
    width_path = base_dir / WIDTH_REL

    seed_header, seed_header_lines, seed_depth = read_esri_ascii(seed_path)
    adem_header, _adem_lines, adem = read_esri_ascii(adem_path)
    adir_header, _adir_lines, adir = read_esri_ascii(adir_path)
    width_header, _width_lines, width = read_esri_ascii(width_path)

    check_same_grid(seed_header, adem_header, "ADEM")
    check_same_grid(seed_header, adir_header, "ADIR")
    check_same_grid(seed_header, width_header, "WIDTH")

    river_mask = (
        np.isfinite(width)
        & (width > 0.0)
    )

    nodata = float(seed_header.get("nodata_value", -9999.0))
    valid_seed = (
        np.isfinite(seed_depth)
        & ~np.isclose(seed_depth, nodata)
    )

    if not np.all(valid_seed[river_mask]):
        raise RuntimeError(
            "At least one width>0 river cell has invalid seed depth."
        )
    if np.any(seed_depth[river_mask] <= 0.0):
        raise RuntimeError(
            "The D50 seed contains non-positive river depth."
        )

    bed_seed = adem - seed_depth
    station_cells = _station_cells_zero_based()
    nan001 = station_cells["NAN001"]
    nan002 = station_cells["NAN002"]
    nan003 = station_cells["NAN003"]

    path_001_002 = trace_to_target(adir, nan001, nan002)
    path_002_003 = trace_to_target(adir, nan002, nan003)
    if not all(bool(river_mask[cell]) for cell in path_001_002):
        raise RuntimeError("NAN001->NAN002 D8 path leaves the width>0 river mask")
    if not all(bool(river_mask[cell]) for cell in path_002_003):
        raise RuntimeError("NAN002->NAN003 D8 path leaves the width>0 river mask")

    downstream_all, terminal_info = trace_downstream_with_terminal(
        adir,
        nan003,
        river_mask=river_mask,
    )
    if len(downstream_all) < 3:
        raise RuntimeError(
            "Fewer than three modeled D8 cells exist from NAN003 "
            "downstream; depth-profile calibration cannot continue."
        )

    chain_12 = path_chainage_km(seed_header, path_001_002)
    chain_23 = path_chainage_km(seed_header, path_002_003)
    chain_down = path_chainage_km(seed_header, downstream_all)

    available_km = float(chain_down[-1])
    desired_total_km = (
        DESIRED_DOWNSTREAM_CORE_KM
        + DESIRED_DOWNSTREAM_TAPER_KM
    )

    if available_km >= desired_total_km:
        target_core_km = DESIRED_DOWNSTREAM_CORE_KM
        target_total_km = desired_total_km
        adaptive_mode = False
    else:
        target_total_km = available_km
        target_core_km = min(
            DESIRED_DOWNSTREAM_CORE_KM,
            ADAPTIVE_CORE_FRACTION * available_km,
        )
        adaptive_mode = True

    core_candidates = np.flatnonzero(
        chain_down >= target_core_km
    )
    core_end_index = (
        int(core_candidates[0])
        if core_candidates.size
        else len(downstream_all) - 2
    )
    core_end_index = max(
        1,
        min(core_end_index, len(downstream_all) - 2),
    )

    if adaptive_mode:
        taper_end_index = len(downstream_all) - 1
    else:
        total_candidates = np.flatnonzero(
            chain_down >= target_total_km
        )
        taper_end_index = (
            int(total_candidates[0])
            if total_candidates.size
            else len(downstream_all) - 1
        )

    taper_end_index = max(
        core_end_index + 1,
        min(taper_end_index, len(downstream_all) - 1),
    )

    downstream_used = downstream_all[:taper_end_index + 1]
    chain_down_used = chain_down[:taper_end_index + 1]

    core_end_km = float(chain_down_used[core_end_index])
    taper_end_km = float(chain_down_used[-1])
    taper_length_km = taper_end_km - core_end_km
    if core_end_km <= 0.0 or taper_length_km <= 0.0:
        raise RuntimeError(
            "Invalid adaptive downstream core/taper geometry."
        )

    mainstem_001_003 = (
        path_001_002[:-1] + path_002_003
    )
    mainstem_all = (
        mainstem_001_003
        + downstream_all[1:]
    )
    mainstem_set = set(mainstem_all)

    # Precompute short tributary tapers once per worker.
    tributary_map: Dict[
        Tuple[int, int],
        List[Tuple[Tuple[int, int], float]],
    ] = {}
    for junction in mainstem_set:
        branches = tributary_cells_for_confluence(
            junction_cell=junction,
            mainstem_set=mainstem_set,
            adir_grid=adir,
            river_mask=river_mask,
            header=seed_header,
            max_distance_km=TRIBUTARY_TAPER_KM,
        )
        if branches:
            tributary_map[junction] = branches

    seed_mainstem_uphill = _max_downstream_uphill_rise(
        bed_seed,
        mainstem_all,
    )

    return {
        "base_dir": base_dir,
        "seed_path": seed_path,
        "header": seed_header,
        "header_lines": seed_header_lines,
        "adem": adem,
        "adir": adir,
        "width": width,
        "river_mask": river_mask,
        "seed_depth": seed_depth,
        "seed_bed": bed_seed,
        "station_cells": station_cells,
        "path_001_002": path_001_002,
        "path_002_003": path_002_003,
        "chain_12": chain_12,
        "chain_23": chain_23,
        "downstream_all": downstream_all,
        "downstream_used": downstream_used,
        "chain_down_used": chain_down_used,
        "core_end_index": core_end_index,
        "core_end_km": core_end_km,
        "taper_end_km": taper_end_km,
        "taper_length_km": taper_length_km,
        "available_downstream_km": available_km,
        "adaptive_mode": adaptive_mode,
        "terminal_info": terminal_info,
        "mainstem_all": mainstem_all,
        "mainstem_set": mainstem_set,
        "tributary_map": tributary_map,
        "seed_mainstem_uphill_m": seed_mainstem_uphill,
    }


def write_depth_token_preserving(
    source_path: Path,
    output_path: Path,
    replacement_map: Dict[Tuple[int, int], float],
    nrows: int,
    ncols: int,
    decimals: int = 5,
) -> None:
    source_path = Path(source_path)
    output_path = Path(output_path)

    lines = source_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()

    header_lines = lines[:6]
    data_lines = lines[6:]
    if len(data_lines) != nrows:
        raise ValueError(
            f"Expected {nrows} raster rows in {source_path}, "
            f"found {len(data_lines)}"
        )

    by_row: Dict[int, Dict[int, float]] = {}
    for (row, col), value in replacement_map.items():
        by_row.setdefault(int(row), {})[int(col)] = float(value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for line in header_lines:
            file.write(line + "\n")

        for row, line in enumerate(data_lines):
            replacements = by_row.get(row)
            if not replacements:
                file.write(line + "\n")
                continue

            tokens = line.split()
            if len(tokens) != ncols:
                raise ValueError(
                    f"Raster row {row} has {len(tokens)} values; "
                    f"expected {ncols}"
                )
            for col, value in replacements.items():
                tokens[col] = f"{value:.{decimals}f}"
            file.write(" ".join(tokens) + "\n")


def effective_geometry_values(vector: np.ndarray) -> List[float]:
    vector = np.asarray(vector, dtype=float)
    if len(vector) != len(GENE_KEYS):
        raise ValueError(
            f"Expected {len(GENE_KEYS)} geometry genes, got {len(vector)}"
        )

    used: List[float] = []
    for key, value in zip(GENE_KEYS, vector):
        lower, upper = BOUNDS[key]
        used.append(
            float(np.clip(float(value), lower, upper))
        )
    return used


def generate_depth_candidate(
    context: dict,
    vector: np.ndarray,
    output_path: Path,
) -> Tuple[List[float], dict]:
    upper_blend, middle_blend, downstream_factor = (
        effective_geometry_values(vector)
    )

    seed_bed = context["seed_bed"]
    seed_depth = context["seed_depth"]
    adem = context["adem"]
    river_mask = context["river_mask"]

    candidate_bed = seed_bed.copy()

    # ---------------------------------------------------------
    # Reach 1: NAN001 -> NAN002
    # ---------------------------------------------------------
    path12 = context["path_001_002"]
    chain12 = context["chain_12"]
    length12 = float(chain12[-1])
    z1 = float(seed_bed[path12[0]])
    z2 = float(seed_bed[path12[-1]])

    for index, cell in enumerate(path12[1:-1], start=1):
        fraction = float(chain12[index] / length12)
        z_reference = z1 + fraction * (z2 - z1)
        z_seed = float(seed_bed[cell])
        candidate_bed[cell] = (
            (1.0 - upper_blend) * z_seed
            + upper_blend * z_reference
        )

    # ---------------------------------------------------------
    # Reach 2: NAN002 -> NAN003
    # ---------------------------------------------------------
    path23 = context["path_002_003"]
    chain23 = context["chain_23"]
    length23 = float(chain23[-1])
    z2_mid = float(seed_bed[path23[0]])
    z3 = float(seed_bed[path23[-1]])

    for index, cell in enumerate(path23[1:-1], start=1):
        fraction = float(chain23[index] / length23)
        z_reference = z2_mid + fraction * (z3 - z2_mid)
        z_seed = float(seed_bed[cell])
        candidate_bed[cell] = (
            (1.0 - middle_blend) * z_seed
            + middle_blend * z_reference
        )

    # ---------------------------------------------------------
    # Reach 3: adaptive downstream control below NAN003
    # ---------------------------------------------------------
    down_cells = context["downstream_used"]
    down_chain = context["chain_down_used"]
    core_index = int(context["core_end_index"])
    core_km = float(context["core_end_km"])
    taper_km = float(context["taper_length_km"])

    z3_seed = float(seed_bed[down_cells[0]])
    z_core_seed = float(seed_bed[down_cells[core_index]])

    # Scaling the D50 drop gives factor=1 -> exactly D50.
    z_core_new = (
        z3_seed
        + downstream_factor
        * (z_core_seed - z3_seed)
    )
    core_offset = z_core_new - z_core_seed

    for index, cell in enumerate(down_cells):
        if index == 0:
            continue

        distance_km = float(down_chain[index])

        if index <= core_index:
            seed_drop_here = (
                float(seed_bed[cell]) - z3_seed
            )
            candidate_bed[cell] = (
                z3_seed
                + downstream_factor
                * seed_drop_here
            )
        else:
            taper_fraction = (
                (distance_km - core_km)
                / taper_km
            )
            taper_fraction = min(
                max(taper_fraction, 0.0),
                1.0,
            )
            weight = 0.5 * (
                1.0
                + np.cos(np.pi * taper_fraction)
            )
            candidate_bed[cell] = (
                float(seed_bed[cell])
                + core_offset * weight
            )

    # ---------------------------------------------------------
    # Confluence smoothing: propagate incremental main-stem
    # correction relative to the D50 seed into tributaries.
    # ---------------------------------------------------------
    tributary_updates: Dict[Tuple[int, int], float] = {}
    affected_confluences = 0

    for junction, branch_cells in context["tributary_map"].items():
        delta_junction = float(
            candidate_bed[junction]
            - seed_bed[junction]
        )
        if abs(delta_junction) < MIN_CONFLUENCE_CORRECTION_M:
            continue

        affected_confluences += 1
        for cell, weight in branch_cells:
            proposed_delta = delta_junction * float(weight)

            if cell in tributary_updates:
                old_delta = (
                    tributary_updates[cell]
                    - float(seed_bed[cell])
                )
                if abs(proposed_delta) <= abs(old_delta):
                    continue

            tributary_updates[cell] = (
                float(seed_bed[cell])
                + proposed_delta
            )

    for cell, new_bed in tributary_updates.items():
        candidate_bed[cell] = new_bed

    candidate_depth = adem - candidate_bed

    river_depths = candidate_depth[river_mask]
    if not np.all(np.isfinite(river_depths)):
        raise RuntimeError("Generated non-finite river depth")
    if np.any(river_depths <= 0.0):
        raise RuntimeError(
            "Generated non-positive river depth: "
            f"minimum={float(np.min(river_depths)):.6f} m"
        )

    # Exact station cells are intentionally fixed.
    for station, cell in context["station_cells"].items():
        if not np.isclose(
            candidate_depth[cell],
            seed_depth[cell],
            atol=1e-10,
        ):
            raise RuntimeError(
                f"Exact {station} depth changed unexpectedly"
            )

    candidate_uphill = _max_downstream_uphill_rise(
        candidate_bed,
        context["mainstem_all"],
    )
    allowed_uphill = (
        float(context["seed_mainstem_uphill_m"])
        + MAINSTEM_UPHILL_ALLOWANCE_M
    )
    if candidate_uphill > allowed_uphill:
        raise RuntimeError(
            "Candidate creates excessive downstream-rising main-stem bed: "
            f"candidate={candidate_uphill:.4f} m, "
            f"seed={context['seed_mainstem_uphill_m']:.4f} m, "
            f"allowed={allowed_uphill:.4f} m"
        )

    changed_mask = (
        river_mask
        & np.isfinite(candidate_depth)
        & np.isfinite(seed_depth)
        & ~np.isclose(
            candidate_depth,
            seed_depth,
            atol=1e-10,
        )
    )
    changed_rows, changed_cols = np.where(changed_mask)

    replacement_map = {
        (int(row), int(col)): float(
            candidate_depth[row, col]
        )
        for row, col in zip(changed_rows, changed_cols)
    }

    output_path = Path(output_path)
    if not replacement_map:
        shutil.copy2(context["seed_path"], output_path)
    else:
        write_depth_token_preserving(
            source_path=context["seed_path"],
            output_path=output_path,
            replacement_map=replacement_map,
            nrows=int(context["header"]["nrows"]),
            ncols=int(context["header"]["ncols"]),
            decimals=5,
        )

    bed_delta = (
        candidate_bed[river_mask]
        - seed_bed[river_mask]
    )

    diagnostics = {
        "genes": {
            key: float(value)
            for key, value in zip(
                GENE_KEYS,
                (
                    upper_blend,
                    middle_blend,
                    downstream_factor,
                ),
            )
        },
        "modified_river_cells_vs_D50": int(
            len(replacement_map)
        ),
        "affected_confluences": int(
            affected_confluences
        ),
        "tributary_cells_tapered": int(
            len(tributary_updates)
        ),
        "max_abs_bed_change_vs_D50_m": float(
            np.max(np.abs(bed_delta))
            if bed_delta.size
            else 0.0
        ),
        "minimum_river_depth_m": float(
            np.min(river_depths)
        ),
        "mainstem_max_uphill_seed_m": float(
            context["seed_mainstem_uphill_m"]
        ),
        "mainstem_max_uphill_candidate_m": float(
            candidate_uphill
        ),
        "available_downstream_km": float(
            context["available_downstream_km"]
        ),
        "downstream_core_end_km": float(
            context["core_end_km"]
        ),
        "downstream_taper_end_km": float(
            context["taper_end_km"]
        ),
        "downstream_adaptive_mode": bool(
            context["adaptive_mode"]
        ),
        "downstream_terminal": context["terminal_info"],
    }

    return [
        upper_blend,
        middle_blend,
        downstream_factor,
    ], diagnostics


def patch_worker_rri_depth_path(input_path: Path) -> None:
    lines = read_lines(input_path)
    matches = [
        index
        for index, line in enumerate(lines)
        if "./riv/" in line
        and "depth" in line.lower()
        and ".txt" in line.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one river depth-file path in RRI_Input.txt; "
            f"found {len(matches)}"
        )

    index = matches[0]
    newline = "\n" if lines[index].endswith("\n") else ""
    lines[index] = f"./riv/{WORKING_DEPTH_NAME}{newline}"
    write_lines_atomic(input_path, lines)


# =============================================================================
# RRI RUNNER
# =============================================================================


def _file_type(path: Path) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/file", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.stdout else ""
    except Exception:
        return ""


def _cleanup_hydros(cwd: Path) -> None:
    if not CLEAN_HYDROS_BEFORE_RUN:
        return
    for path in cwd.glob("**/hydro*.txt"):
        try:
            path.unlink()
        except Exception:
            pass


def run_rri_with_log(cwd: Path, run_log: Path) -> None:
    with open(run_log, "w", encoding="utf-8") as log:
        def write(message: str) -> None:
            log.write(message + "\n")

        write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] START in {cwd}")

        executable: Optional[Path] = None
        for candidate in RRI_EXE_CANDS:
            path = cwd / candidate
            if path.exists():
                executable = path
                break

        if executable is None:
            write(f"[error] RRI executable not found: {RRI_EXE_CANDS}")
            raise FileNotFoundError("RRI executable not found")

        try:
            if executable.suffix.lower() != ".exe":
                mode = executable.stat().st_mode
                if not bool(mode & stat.S_IXUSR):
                    executable.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    write("[execbit] chmod +x applied")
        except Exception as exc:
            write(f"[warn] chmod failed: {exc}")

        file_type = _file_type(executable)
        if file_type:
            write(f"[file] {file_type}")

        _cleanup_hydros(cwd)
        write(f"[run] Launching: {executable}")

        process = subprocess.Popen(
            [str(executable)],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is not None:
            for line in process.stdout:
                log.write(line)

        return_code = process.wait()
        write(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] END "
            f"(code {return_code})"
        )

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, [str(executable)])

        if not list(cwd.glob("**/hydro*.txt")):
            write("[error] RRI finished but no hydro*.txt was produced")
            raise RuntimeError("No hydrograph output was produced")


# =============================================================================
# OBSERVED AND SIMULATED SERIES
# =============================================================================

_OBSERVED_CACHE: Optional[Dict[str, pd.Series]] = None


def _read_observed_two_column(config: Dict[str, object]) -> pd.Series:
    path = Path(config["path"])
    dataframe = pd.read_csv(
        path,
        sep=str(config["sep"]),
        header=0 if bool(config["has_header"]) else None,
    )

    datetime_col = int(config["datetime_col"])
    value_col = int(config["value_col"])
    if dataframe.shape[1] <= max(datetime_col, value_col):
        raise ValueError(
            f"Observed file {path} does not contain requested columns "
            f"{datetime_col} and {value_col}"
        )

    raw_datetime = dataframe.iloc[:, datetime_col].astype(str).str.strip()
    raw_values = pd.to_numeric(dataframe.iloc[:, value_col], errors="coerce")
    datetime_format = config.get("datetime_format")

    if datetime_format:
        datetimes = pd.to_datetime(
            raw_datetime,
            format=str(datetime_format),
            errors="coerce",
        )
    else:
        datetimes = pd.to_datetime(raw_datetime, errors="coerce")

    valid = datetimes.notna() & raw_values.notna()
    series = pd.Series(
        raw_values.loc[valid].to_numpy(dtype=float),
        index=pd.DatetimeIndex(datetimes.loc[valid]),
        dtype=float,
    )
    return series.sort_index()


def load_all_observed_series() -> Dict[str, pd.Series]:
    global _OBSERVED_CACHE
    if _OBSERVED_CACHE is not None:
        return _OBSERVED_CACHE

    if OBS_INPUT_MODE == "separate_files":
        _OBSERVED_CACHE = {
            station: _read_observed_two_column(OBS_STATION_FILES[station])
            for station in STATION_IDS
        }
        return _OBSERVED_CACHE

    config = OBS_SINGLE_TABLE
    path = Path(config["path"])
    dataframe = pd.read_csv(
        path,
        sep=str(config["sep"]),
        header=0 if bool(config["has_header"]) else None,
    )

    raw_datetime = resolve_dataframe_column(
        dataframe, config["datetime_col"]
    ).astype(str).str.strip()
    datetime_format = config.get("datetime_format")
    if datetime_format:
        datetimes = pd.to_datetime(
            raw_datetime,
            format=str(datetime_format),
            errors="coerce",
        )
    else:
        datetimes = pd.to_datetime(raw_datetime, errors="coerce")

    output: Dict[str, pd.Series] = {}
    station_cols = config["station_value_cols"]
    for station in STATION_IDS:
        raw_values = pd.to_numeric(
            resolve_dataframe_column(dataframe, station_cols[station]),
            errors="coerce",
        )
        valid = datetimes.notna() & raw_values.notna()
        output[station] = pd.Series(
            raw_values.loc[valid].to_numpy(dtype=float),
            index=pd.DatetimeIndex(datetimes.loc[valid]),
            dtype=float,
        ).sort_index()

    _OBSERVED_CACHE = output
    return _OBSERVED_CACHE


def _find_output_file(cwd: Path, path_or_name: str) -> Optional[Path]:
    candidate = cwd / path_or_name
    if candidate.exists():
        return candidate

    matches = sorted(cwd.glob(f"**/{Path(path_or_name).name}"))
    return matches[0] if matches else None


def _read_whitespace_numeric_rows(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            try:
                row = [float(value.replace("D", "e").replace("d", "e")) for value in parts]
            except ValueError:
                continue
            rows.append(row)
    return rows


def load_simulated_station_series(cwd: Path) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    """Load all station simulations and return series plus source-file paths."""
    output: Dict[str, pd.Series] = {}
    source_paths: Dict[str, str] = {}

    if SIM_OUTPUT_MODE == "multi_column":
        path = _find_output_file(cwd, SIM_MULTI_FILE)
        if path is None:
            raise FileNotFoundError(
                f"Could not find multi-station hydrograph file {SIM_MULTI_FILE!r} in {cwd}"
            )

        rows = _read_whitespace_numeric_rows(path)
        required_col = max(
            [SIM_TIME_COL] + [SIM_STATION_VALUE_COLS[s] for s in STATION_IDS]
        )
        rows = [row for row in rows if len(row) > required_col]
        if not rows:
            raise ValueError(
                f"No usable rows found in {path}; required maximum column index is {required_col}"
            )

        seconds = np.asarray([row[SIM_TIME_COL] for row in rows], dtype=float)
        datetimes = SIM_START + pd.to_timedelta(seconds, unit="s")

        for station in STATION_IDS:
            value_col = int(SIM_STATION_VALUE_COLS[station])
            values = np.asarray([row[value_col] for row in rows], dtype=float)
            valid = np.isfinite(seconds) & np.isfinite(values)
            output[station] = pd.Series(
                values[valid],
                index=pd.DatetimeIndex(datetimes[valid]),
                dtype=float,
            ).sort_index()
            source_paths[station] = str(path)

        return output, source_paths

    for station in STATION_IDS:
        filename = SIM_STATION_FILES[station]
        path = _find_output_file(cwd, filename)
        if path is None:
            raise FileNotFoundError(
                f"Could not find simulated hydrograph for {station}: {filename!r}"
            )

        rows = _read_whitespace_numeric_rows(path)
        required_col = max(SIM_SEPARATE_TIME_COL, SIM_SEPARATE_VALUE_COL)
        rows = [row for row in rows if len(row) > required_col]
        if not rows:
            raise ValueError(f"No usable simulated rows for {station} in {path}")

        seconds = np.asarray(
            [row[SIM_SEPARATE_TIME_COL] for row in rows], dtype=float
        )
        values = np.asarray(
            [row[SIM_SEPARATE_VALUE_COL] for row in rows], dtype=float
        )
        datetimes = SIM_START + pd.to_timedelta(seconds, unit="s")
        valid = np.isfinite(seconds) & np.isfinite(values)
        output[station] = pd.Series(
            values[valid],
            index=pd.DatetimeIndex(datetimes[valid]),
            dtype=float,
        ).sort_index()
        source_paths[station] = str(path)

    return output, source_paths


# =============================================================================
# TEMPORAL MATCHING
# =============================================================================


def pick_daily_nearest(
    series: pd.Series,
    hour: int,
    minute: int,
    second: int,
    tolerance: str,
) -> pd.Series:
    """Pick the value nearest to the requested time for each date."""
    if series is None or series.empty:
        return pd.Series(dtype=float)

    series = series.dropna().sort_index()
    index = pd.DatetimeIndex(series.index)
    targets = index.normalize() + pd.Timedelta(
        hours=hour,
        minutes=minute,
        seconds=second,
    )
    difference_ns = np.abs((index - targets).asi8)
    tolerance_ns = pd.Timedelta(tolerance).value

    dataframe = pd.DataFrame(
        {
            "value": series.to_numpy(dtype=float),
            "target": targets,
            "difference_ns": difference_ns,
        },
        index=index,
    )
    dataframe = dataframe.loc[dataframe["difference_ns"] <= tolerance_ns]
    if dataframe.empty:
        return pd.Series(dtype=float)

    chosen_indices = dataframe.groupby("target")["difference_ns"].idxmin()
    chosen = dataframe.loc[chosen_indices].sort_values("target")
    return pd.Series(
        chosen["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(chosen["target"]),
        dtype=float,
    )


def pick_hourly_nearest(series: pd.Series, tolerance: str) -> pd.Series:
    """Snap each value to its nearest hour and retain the closest value per hour."""
    if series is None or series.empty:
        return pd.Series(dtype=float)

    series = series.dropna().sort_index()
    index = pd.DatetimeIndex(series.index)
    targets = index.round("h")
    difference_ns = np.abs((index - targets).asi8)
    tolerance_ns = pd.Timedelta(tolerance).value

    dataframe = pd.DataFrame(
        {
            "value": series.to_numpy(dtype=float),
            "target": targets,
            "difference_ns": difference_ns,
        },
        index=index,
    )
    dataframe = dataframe.loc[dataframe["difference_ns"] <= tolerance_ns]
    if dataframe.empty:
        return pd.Series(dtype=float)

    chosen_indices = dataframe.groupby("target")["difference_ns"].idxmin()
    chosen = dataframe.loc[chosen_indices].sort_values("target")
    return pd.Series(
        chosen["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(chosen["target"]),
        dtype=float,
    )


def build_station_pairs(
    observed: Dict[str, pd.Series],
    simulated: Dict[str, pd.Series],
) -> Dict[str, pd.DataFrame]:
    pairs: Dict[str, pd.DataFrame] = {}

    for station in STATION_IDS:
        obs_series = observed[station]
        sim_series = simulated[station]

        if MATCH_MODE == "daily":
            obs_matched = pick_daily_nearest(
                obs_series,
                DAILY_PICK_HOUR,
                DAILY_PICK_MINUTE,
                DAILY_PICK_SECOND,
                DAILY_TOLERANCE,
            )
            sim_matched = pick_daily_nearest(
                sim_series,
                DAILY_PICK_HOUR,
                DAILY_PICK_MINUTE,
                DAILY_PICK_SECOND,
                DAILY_TOLERANCE,
            )
        elif MATCH_MODE == "hourly":
            obs_matched = pick_hourly_nearest(obs_series, HOURLY_TOLERANCE)
            sim_matched = pick_hourly_nearest(sim_series, HOURLY_TOLERANCE)
        else:
            raise ValueError(
                f"MATCH_MODE must be 'daily' or 'hourly', got {MATCH_MODE!r}"
            )

        dataframe = pd.concat(
            {"obs": obs_matched, "sim": sim_matched},
            axis=1,
        ).dropna()
        dataframe = dataframe.loc[
            (dataframe.index >= SCORE_START) & (dataframe.index <= SCORE_END)
        ]
        pairs[station] = dataframe.sort_index()

    return pairs


# =============================================================================
# METRICS AND THREE-STATION OBJECTIVES
# =============================================================================


def _finite_align(observed: np.ndarray, simulated: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    mask = np.isfinite(observed) & np.isfinite(simulated)
    return observed[mask], simulated[mask], int(mask.sum())


def metric_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed, simulated, count = _finite_align(observed, simulated)
    if count < 2:
        return -1e9
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= 0.0:
        return -1e9
    return float(1.0 - np.sum((observed - simulated) ** 2) / denominator)


def metric_rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed, simulated, count = _finite_align(observed, simulated)
    if count == 0:
        return 1e9
    return float(np.sqrt(np.mean((observed - simulated) ** 2)))


def metric_mae(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed, simulated, count = _finite_align(observed, simulated)
    if count == 0:
        return 1e9
    return float(np.mean(np.abs(observed - simulated)))


def metric_bias(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed, simulated, count = _finite_align(observed, simulated)
    if count == 0:
        return 1e9
    return float(np.mean(simulated - observed))


def metric_corr(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed, simulated, count = _finite_align(observed, simulated)
    if count < 2:
        return float("nan")
    if np.std(observed) <= 0.0 or np.std(simulated) <= 0.0:
        return float("nan")
    return float(np.corrcoef(observed, simulated)[0, 1])


def metric_kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed, simulated, count = _finite_align(observed, simulated)
    if count < 2:
        return -1e9

    mean_obs = float(np.mean(observed))
    mean_sim = float(np.mean(simulated))
    std_obs = float(np.std(observed))
    std_sim = float(np.std(simulated))

    if std_obs <= 0.0 or mean_obs == 0.0:
        return -1e9

    correlation = metric_corr(observed, simulated)
    if not np.isfinite(correlation):
        return -1e9

    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs
    return float(
        1.0
        - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def station_error_scale(observed: np.ndarray) -> float:
    finite = observed[np.isfinite(observed)]
    if finite.size == 0:
        return 1.0

    if ERROR_NORMALIZATION == "std":
        scale = float(np.std(finite))
    elif ERROR_NORMALIZATION == "range":
        scale = float(np.max(finite) - np.min(finite))
    elif ERROR_NORMALIZATION == "mean_abs":
        scale = float(np.mean(np.abs(finite)))
    else:
        raise ValueError(
            "ERROR_NORMALIZATION must be 'std', 'range', or 'mean_abs'"
        )

    # Robust fallbacks for nearly constant or zero-reference observations.
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(finite) - np.min(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.mean(np.abs(finite)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    return scale


def compute_station_metrics(dataframe: pd.DataFrame) -> Dict[str, float]:
    if dataframe is None or dataframe.empty:
        return {
            "NSE": -1e9,
            "RMSE": 1e9,
            "NRMSE": 1e9,
            "MAE": 1e9,
            "Bias": 1e9,
            "NAbsBias": 1e9,
            "KGE": -1e9,
            "Corr": float("nan"),
            "ObsScale": 1.0,
            "n_pairs": 0,
        }

    observed = dataframe["obs"].to_numpy(dtype=float)
    simulated = dataframe["sim"].to_numpy(dtype=float)
    scale = station_error_scale(observed)
    rmse = metric_rmse(observed, simulated)
    bias = metric_bias(observed, simulated)

    return {
        "NSE": metric_nse(observed, simulated),
        "RMSE": rmse,
        "NRMSE": float(rmse / scale),
        "MAE": metric_mae(observed, simulated),
        "Bias": bias,
        "NAbsBias": float(abs(bias) / scale),
        "KGE": metric_kge(observed, simulated),
        "Corr": metric_corr(observed, simulated),
        "ObsScale": float(scale),
        "n_pairs": int(len(dataframe)),
    }


def aggregate_three_station_objectives(
    station_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    """
    Return aggregate objectives and a validity flag.

    Objective 1: maximize weighted mean NSE.
    Objective 2: minimize weighted mean normalized RMSE.
    Objective 3: minimize worst-station normalized absolute bias.
    """
    weights = normalized_station_weights()
    invalid_stations = [
        station
        for station in STATION_IDS
        if int(station_metrics[station]["n_pairs"]) < MIN_PAIRS_PER_STATION
        or not np.isfinite(float(station_metrics[station]["NSE"]))
        or not np.isfinite(float(station_metrics[station]["NRMSE"]))
        or not np.isfinite(float(station_metrics[station]["NAbsBias"]))
    ]

    if invalid_stations:
        return {
            "valid": False,
            "invalid_stations": invalid_stations,
            "MeanNSE": -1e9,
            "MeanNRMSE": 1e9,
            "WorstNAbsBias": 1e9,
        }

    mean_nse = float(
        sum(weights[s] * float(station_metrics[s]["NSE"]) for s in STATION_IDS)
    )
    mean_nrmse = float(
        sum(
            weights[s] * float(station_metrics[s]["NRMSE"])
            for s in STATION_IDS
        )
    )
    worst_nabs_bias = float(
        max(float(station_metrics[s]["NAbsBias"]) for s in STATION_IDS)
    )

    return {
        "valid": True,
        "invalid_stations": [],
        "MeanNSE": mean_nse,
        "MeanNRMSE": mean_nrmse,
        "WorstNAbsBias": worst_nabs_bias,
    }


# =============================================================================
# WORKER SETUP AND EVALUATION
# =============================================================================

_WORKDIR: Optional[Path] = None
_GEOM_CONTEXT: Optional[dict] = None


def _init_worker(base_particle_dir: str, workers_root: str) -> None:
    global _WORKDIR, _GEOM_CONTEXT, PARTICLE_DIR, RRI_INPUT

    process_id = os.getpid()
    base = Path(base_particle_dir)
    root = Path(workers_root)
    root.mkdir(parents=True, exist_ok=True)
    workdir = root / f"Worker_p{process_id}"

    if REFRESH_WORKER_DIRS and workdir.exists():
        shutil.rmtree(workdir)

    if not workdir.exists():
        def ignore_archive(dirpath: str, names: List[str]) -> set:
            if Path(dirpath).name.lower().startswith("archive"):
                return set(names)
            return set()

        shutil.copytree(
            base,
            workdir,
            ignore=ignore_archive,
        )

    _WORKDIR = workdir
    PARTICLE_DIR = workdir
    RRI_INPUT = workdir / "RRI_Input.txt"

    # Only worker copies are patched. The base RRI_Input.txt remains untouched.
    patch_worker_rri_depth_path(RRI_INPUT)

    _GEOM_CONTEXT = build_geometry_context(workdir)

    # Start with the known D50 geometry so the worker input is always valid.
    shutil.copy2(
        _GEOM_CONTEXT["seed_path"],
        workdir / WORKING_DEPTH_REL,
    )


def _safe_copy(source: Path, destination: Path) -> Optional[str]:
    try:
        shutil.copy2(source, destination)
        return str(destination)
    except Exception:
        return None


def archive_simulation_outputs(
    cwd: Path,
    generation: int,
    individual: int,
    timestamp: str,
) -> Dict[str, Optional[str]]:
    archived: Dict[str, Optional[str]] = {}

    if SIM_OUTPUT_MODE == "multi_column":
        path = _find_output_file(cwd, SIM_MULTI_FILE)
        if path is not None:
            destination = ARCHIVE_DIR / (
                f"{path.stem}_gen{generation:03d}_ind{individual:03d}_"
                f"{timestamp}{path.suffix}"
            )
            archived["MULTI"] = _safe_copy(path, destination)
        else:
            archived["MULTI"] = None
        return archived

    for station in STATION_IDS:
        path = _find_output_file(cwd, SIM_STATION_FILES[station])
        if path is None:
            archived[station] = None
            continue
        destination = ARCHIVE_DIR / (
            f"{station}_{path.stem}_gen{generation:03d}_"
            f"ind{individual:03d}_{timestamp}{path.suffix}"
        )
        archived[station] = _safe_copy(path, destination)

    return archived


def evaluate_worker(
    vector: np.ndarray,
    base_names: List[str],
    spans: List[Tuple[int, int]],
    gene_keys: List[str],
    generation: int,
    individual: int,
) -> Tuple[float, float, float, dict]:
    del base_names, spans  # retained in signature for shared NSGA-II machinery

    if _GEOM_CONTEXT is None:
        raise RuntimeError("Worker geometry context was not initialized")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_log = ARCHIVE_DIR / (
        f"RRI_run_gen{generation:03d}_ind{individual:03d}_{timestamp}.log"
    )
    geometry_log = ARCHIVE_DIR / (
        f"geometry_gen{generation:03d}_ind{individual:03d}_{timestamp}.json"
    )
    working_depth = PARTICLE_DIR / WORKING_DEPTH_REL

    used_values = effective_geometry_values(vector)
    geometry_diagnostics: dict = {}
    archived_depth: Optional[str] = None

    try:
        used_values, geometry_diagnostics = generate_depth_candidate(
            _GEOM_CONTEXT,
            np.asarray(vector, dtype=float),
            working_depth,
        )

        geometry_log.write_text(
            json.dumps(
                geometry_diagnostics,
                indent=2,
                allow_nan=True,
            ),
            encoding="utf-8",
        )

        if ARCHIVE_ALL_DEPTHS:
            archived_depth_path = ARCHIVE_DIR / (
                f"depth_gen{generation:03d}_ind{individual:03d}_"
                f"{timestamp}.txt"
            )
            archived_depth = _safe_copy(
                working_depth,
                archived_depth_path,
            )

        run_rri_with_log(PARTICLE_DIR, run_log)

    except Exception as exc:
        row = {
            "calibration_mode": CALIBRATION_MODE,
            "generation": generation,
            "individual": individual,
            "timestamp": timestamp,
            "error": str(exc),
            "aggregate_objectives": {
                "MeanNSE": -1e9,
                "MeanNRMSE": 1e9,
                "WorstNAbsBias": 1e9,
                "valid": False,
                "invalid_stations": list(STATION_IDS),
            },
            "station_metrics": {},
            "params": {
                key: float(value)
                for key, value in zip(gene_keys, used_values)
            },
            "geometry": geometry_diagnostics,
            "geometry_log": str(geometry_log),
            "generated_depth": str(working_depth),
            "archived_depth": archived_depth,
            "rri_run_log": str(run_log),
            "particle_dir": str(PARTICLE_DIR),
        }
        return -1e9, 1e9, 1e9, row

    archived_outputs = archive_simulation_outputs(
        PARTICLE_DIR,
        generation,
        individual,
        timestamp,
    )

    try:
        observed = load_all_observed_series()
        simulated, simulation_source_paths = (
            load_simulated_station_series(PARTICLE_DIR)
        )
        station_pairs = build_station_pairs(
            observed,
            simulated,
        )
        station_metrics = {
            station: compute_station_metrics(
                station_pairs[station]
            )
            for station in STATION_IDS
        }
        aggregate = aggregate_three_station_objectives(
            station_metrics
        )
    except Exception as exc:
        row = {
            "calibration_mode": CALIBRATION_MODE,
            "generation": generation,
            "individual": individual,
            "timestamp": timestamp,
            "error": f"Output/pairing failure: {exc}",
            "aggregate_objectives": {
                "MeanNSE": -1e9,
                "MeanNRMSE": 1e9,
                "WorstNAbsBias": 1e9,
                "valid": False,
                "invalid_stations": list(STATION_IDS),
            },
            "station_metrics": {},
            "params": {
                key: float(value)
                for key, value in zip(gene_keys, used_values)
            },
            "geometry": geometry_diagnostics,
            "geometry_log": str(geometry_log),
            "generated_depth": str(working_depth),
            "archived_depth": archived_depth,
            "rri_run_log": str(run_log),
            "archived_outputs": archived_outputs,
            "particle_dir": str(PARTICLE_DIR),
        }
        return -1e9, 1e9, 1e9, row

    pair_paths: Dict[str, Optional[str]] = {}
    for station in STATION_IDS:
        path = ARCHIVE_DIR / (
            f"pairs_{station}_gen{generation:03d}_ind{individual:03d}_"
            f"{timestamp}.tsv"
        )
        try:
            station_pairs[station].to_csv(
                path,
                sep="\t",
                float_format="%.6f",
                date_format="%Y-%m-%d %H:%M:%S",
            )
            pair_paths[station] = str(path)
        except Exception:
            pair_paths[station] = None

    mean_nse = float(aggregate["MeanNSE"])
    mean_nrmse = float(aggregate["MeanNRMSE"])
    worst_nabs_bias = float(aggregate["WorstNAbsBias"])

    row = {
        "calibration_mode": CALIBRATION_MODE,
        "generation": generation,
        "individual": individual,
        "timestamp": timestamp,
        "objective_definition": {
            "objective_1": "maximize equal-weight mean station NSE",
            "objective_2": "minimize equal-weight mean station normalized RMSE",
            "objective_3": "minimize worst-station normalized absolute bias",
            "error_normalization": ERROR_NORMALIZATION,
            "station_weights": normalized_station_weights(),
        },
        "aggregate_objectives": aggregate,
        "station_metrics": station_metrics,
        "params": {
            key: float(value)
            for key, value in zip(gene_keys, used_values)
        },
        "geometry": geometry_diagnostics,
        "geometry_log": str(geometry_log),
        "generated_depth": str(working_depth),
        "archived_depth": archived_depth,
        "rri_run_log": str(run_log),
        "simulation_source_paths": simulation_source_paths,
        "archived_outputs": archived_outputs,
        "pair_paths": pair_paths,
        "particle_dir": str(PARTICLE_DIR),
    }

    return mean_nse, mean_nrmse, worst_nabs_bias, row


# =============================================================================
# GENETIC OPERATORS
# =============================================================================


def init_population(
    number_of_individuals: int,
    base_names: Sequence[str],
    initial_vector: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    lower = np.asarray([BOUNDS[name][0] for name in base_names], dtype=float)
    upper = np.asarray([BOUNDS[name][1] for name in base_names], dtype=float)
    dimension = len(base_names)

    if (
        not USE_SEEDING
        or initial_vector is None
        or len(initial_vector) != dimension
    ):
        return [
            lower + np.random.rand(dimension) * (upper - lower)
            for _ in range(number_of_individuals)
        ]

    population: List[np.ndarray] = []
    seed = np.clip(np.asarray(initial_vector, dtype=float), lower, upper)
    population.append(seed.copy())

    remaining = max(0, number_of_individuals - 1)
    local_count = int(round(remaining * SEED_LOCAL_FRAC))
    local_scale = SEED_LOCAL_SCALE * (upper - lower)

    for _ in range(local_count):
        individual = seed + np.random.normal(size=dimension) * local_scale
        population.append(np.clip(individual, lower, upper))

    while len(population) < number_of_individuals:
        population.append(lower + np.random.rand(dimension) * (upper - lower))

    return population


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    base_names: Sequence[str],
    swap_probability: float = 0.5,
) -> np.ndarray:
    child = parent_a.copy()
    mask = np.random.rand(len(parent_a)) < swap_probability
    child[mask] = parent_b[mask]

    for index, name in enumerate(base_names):
        lower, upper = BOUNDS[name]
        child[index] = np.clip(child[index], lower, upper)

    return child


def gaussian_mutation(
    individual: np.ndarray,
    base_names: Sequence[str],
    gene_probability: float = MUT_PROB,
    sigma: float = MUT_SIGMA,
) -> np.ndarray:
    mutated = individual.copy()
    for index, name in enumerate(base_names):
        if np.random.rand() < gene_probability:
            lower, upper = BOUNDS[name]
            span = upper - lower
            mutated[index] += np.random.normal(0.0, sigma) * span
            mutated[index] = np.clip(mutated[index], lower, upper)
    return mutated


# =============================================================================
# NSGA-II
# =============================================================================


def dominates(objectives_a: Sequence[float], objectives_b: Sequence[float]) -> bool:
    less_or_equal = all(a <= b for a, b in zip(objectives_a, objectives_b))
    strictly_less = any(a < b for a, b in zip(objectives_a, objectives_b))
    return less_or_equal and strictly_less


def fast_nondominated_sort(objective_vectors: Sequence[Sequence[float]]) -> List[List[int]]:
    count = len(objective_vectors)
    dominates_set = [set() for _ in range(count)]
    domination_count = [0] * count
    fronts: List[List[int]] = [[]]

    for p in range(count):
        for q in range(count):
            if p == q:
                continue
            if dominates(objective_vectors[p], objective_vectors[q]):
                dominates_set[p].add(q)
            elif dominates(objective_vectors[q], objective_vectors[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: List[int] = []
        for p in fronts[front_index]:
            for q in dominates_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        front_index += 1
        fronts.append(next_front)

    if fronts and not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(
    front: Sequence[int],
    objective_vectors: Sequence[Sequence[float]],
) -> Dict[int, float]:
    if not front:
        return {}

    number_of_objectives = len(objective_vectors[0])
    distance = {index: 0.0 for index in front}

    if len(front) <= 2:
        for index in front:
            distance[index] = float("inf")
        return distance

    for objective_index in range(number_of_objectives):
        ordered = sorted(
            front,
            key=lambda index: objective_vectors[index][objective_index],
        )
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")

        minimum = objective_vectors[ordered[0]][objective_index]
        maximum = objective_vectors[ordered[-1]][objective_index]
        span = maximum - minimum
        if span == 0.0:
            continue

        for position in range(1, len(ordered) - 1):
            previous_index = ordered[position - 1]
            next_index = ordered[position + 1]
            current_index = ordered[position]
            distance[current_index] += (
                objective_vectors[next_index][objective_index]
                - objective_vectors[previous_index][objective_index]
            ) / span

    return distance


def front_ranks(fronts: Sequence[Sequence[int]]) -> Dict[int, int]:
    ranks: Dict[int, int] = {}
    for rank, front in enumerate(fronts):
        for index in front:
            ranks[index] = rank
    return ranks


def crowded_tournament(
    population: Sequence[np.ndarray],
    fronts: Sequence[Sequence[int]],
    distances: Dict[int, float],
    tournament_size: int = TOURNAMENT_SIZE,
) -> int:
    population_size = len(population)
    if population_size == 0:
        raise ValueError("Cannot select from an empty population")
    if population_size == 1:
        return 0

    size = min(tournament_size, population_size)
    candidates = np.random.choice(population_size, size=size, replace=False).tolist()
    ranks = front_ranks(fronts)

    def better(index_a: int, index_b: int) -> bool:
        rank_a = ranks[index_a]
        rank_b = ranks[index_b]
        if rank_a != rank_b:
            return rank_a < rank_b
        return distances.get(index_a, 0.0) > distances.get(index_b, 0.0)

    best = candidates[0]
    for candidate in candidates[1:]:
        if better(candidate, best):
            best = candidate
    return int(best)


def select_balanced_compromise(
    front: Sequence[int],
    objective_vectors: Sequence[Sequence[float]],
) -> int:
    """
    Select one representative Pareto solution using normalized distance to the
    ideal point. This avoids the original lexicographic rule in which NSE
    effectively dominated the reported elite solution.
    """
    if not front:
        raise ValueError("Cannot select a compromise from an empty front")
    if len(front) == 1:
        return int(front[0])

    matrix = np.asarray([objective_vectors[index] for index in front], dtype=float)
    minima = np.min(matrix, axis=0)
    maxima = np.max(matrix, axis=0)
    spans = maxima - minima
    spans[spans <= 0.0] = 1.0
    normalized = (matrix - minima) / spans
    distances = np.sqrt(np.sum(normalized ** 2, axis=1))
    return int(front[int(np.argmin(distances))])


# =============================================================================
# STATE AND CHECKPOINTS
# =============================================================================

_STATE = None
STOP_REQUESTED = False


def get_state():
    global _STATE
    if _STATE is None:
        base_names = list(GENE_KEYS)
        initial_vector = INITIAL_VECTOR.copy()

        # spans are no longer used, but are retained so the unchanged
        # NSGA-II/cache call signatures remain compatible.
        spans: List[Tuple[int, int]] = [
            (-1, -1)
            for _ in GENE_KEYS
        ]
        gene_keys = list(GENE_KEYS)
        _STATE = (
            None,
            base_names,
            initial_vector,
            spans,
            gene_keys,
        )
    return _STATE


def handle_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"[SIGNAL] Caught signal {signum}; the script will checkpoint at the "
        "next safe point."
    )


def save_checkpoint(
    generation_next: int,
    gene_keys: Sequence[str],
    population: Sequence[np.ndarray],
    population_fitness: Optional[
        Sequence[Tuple[float, float, float]]
    ],
    fitness_cache: Dict[
        Tuple[str, ...],
        Tuple[Tuple[float, float, float], dict],
    ],
    rng_state,
    elite_entry: Optional[dict] = None,
) -> None:
    existing_notes: Dict[str, object] = {}
    elite_history: List[dict] = []

    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "rb") as file:
                previous = pickle.load(file)
            existing_notes = previous.get("notes", {}) or {}
            elite_history = list(existing_notes.get("elite_history", []))
        except Exception:
            existing_notes = {}
            elite_history = []

    if elite_entry is not None:
        elite_history.append(elite_entry)
        existing_notes["elite_history"] = elite_history
        existing_notes["last_elite"] = elite_entry

    payload = {
        "calibration_mode": CALIBRATION_MODE,
        "generation_next": generation_next,
        "gene_keys": list(gene_keys),
        "population": [
            np.asarray(individual, dtype=float)
            for individual in population
        ],
        "population_fitness": (
            [
                tuple(float(value) for value in fitness)
                for fitness in population_fitness
            ]
            if population_fitness is not None
            else None
        ),
        "fitness_cache": fitness_cache,
        "numpy_rng_state": rng_state,
        "timestamp": datetime.now().isoformat(),
        "notes": existing_notes,
    }

    with open(CHECKPOINT_FILE, "wb") as file:
        pickle.dump(payload, file)
    print(
        f"[CKPT] Saved checkpoint for generation {generation_next}: "
        f"{CHECKPOINT_FILE}"
    )

def load_checkpoint():
    if not CHECKPOINT_FILE.exists():
        return None
    with open(CHECKPOINT_FILE, "rb") as file:
        payload = pickle.load(file)
    print(
        f"[CKPT] Loaded checkpoint from {CHECKPOINT_FILE} "
        f"(next generation {payload.get('generation_next')})"
    )
    return payload


# =============================================================================
# TSV LOGGING
# =============================================================================


def format_number(value) -> str:
    if value is None:
        return "nan"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.8g}"


def write_tsv_header(gene_keys: Sequence[str]) -> None:
    if TSV_LOG.exists():
        return

    header = [
        "CalibrationMode",
        "Generation",
        "Phase",
        "Individual",
        "EvaluationSource",
        "MeanNSE",
        "MeanNRMSE",
        "WorstNAbsBias",
        "AggregateValid",
        "InvalidStations",
    ]

    for station in STATION_IDS:
        header.extend(
            [
                f"{station}_NSE",
                f"{station}_RMSE",
                f"{station}_NRMSE",
                f"{station}_Bias",
                f"{station}_NAbsBias",
                f"{station}_MAE",
                f"{station}_KGE",
                f"{station}_Corr",
                f"{station}_ObsScale",
                f"{station}_n",
            ]
        )

    header.extend(gene_keys)
    with open(TSV_LOG, "w", encoding="utf-8") as file:
        file.write("\t".join(header) + "\n")


def write_tsv_row(
    generation: int,
    phase: str,
    individual: int,
    vector: np.ndarray,
    gene_keys: Sequence[str],
    row: dict,
) -> None:
    aggregate = row.get("aggregate_objectives", {})
    station_metrics = row.get("station_metrics", {})

    values: List[object] = [
        CALIBRATION_MODE,
        generation,
        phase,
        individual,
        row.get("evaluation_source", "RRI_RUN"),
        format_number(aggregate.get("MeanNSE", -1e9)),
        format_number(aggregate.get("MeanNRMSE", 1e9)),
        format_number(aggregate.get("WorstNAbsBias", 1e9)),
        int(bool(aggregate.get("valid", False))),
        ",".join(aggregate.get("invalid_stations", [])),
    ]

    for station in STATION_IDS:
        metrics = station_metrics.get(station, {})
        values.extend(
            [
                format_number(metrics.get("NSE")),
                format_number(metrics.get("RMSE")),
                format_number(metrics.get("NRMSE")),
                format_number(metrics.get("Bias")),
                format_number(metrics.get("NAbsBias")),
                format_number(metrics.get("MAE")),
                format_number(metrics.get("KGE")),
                format_number(metrics.get("Corr")),
                format_number(metrics.get("ObsScale")),
                metrics.get("n_pairs", 0),
            ]
        )

    values.extend(format_number(value) for value in vector)

    with open(TSV_LOG, "a", encoding="utf-8") as file:
        file.write("\t".join(map(str, values)) + "\n")



# =============================================================================
# DUPLICATE-EVALUATION CACHE
# =============================================================================


FitnessTuple = Tuple[float, float, float]
CacheKey = Tuple[str, ...]
FitnessCache = Dict[CacheKey, Tuple[FitnessTuple, dict]]


def effective_parameter_values(
    vector: np.ndarray,
    base_names: Sequence[str],
    spans: Sequence[Tuple[int, int]],
) -> List[float]:
    del base_names, spans
    return effective_geometry_values(
        np.asarray(vector, dtype=float)
    )


def effective_parameter_key(
    vector: np.ndarray,
    base_names: Sequence[str],
    spans: Sequence[Tuple[int, int]],
) -> CacheKey:
    """
    Duplicate key for the effective geometry genes.

    Genes are clipped to their configured bounds and represented using the
    same six-significant-digit convention used by the previous calibration
    cache. Equivalent candidates therefore reuse the deterministic RRI result.
    """
    values = effective_parameter_values(
        vector,
        base_names,
        spans,
    )
    return tuple(f"{value:.6g}" for value in values)


def make_reuse_row(
    source_row: dict,
    generation: int,
    individual: int,
    evaluation_source: str,
    cache_key: CacheKey,
) -> dict:
    """Clone a prior result for logging without pretending RRI ran again."""
    row = copy.deepcopy(source_row)
    source_generation = source_row.get("generation")
    source_individual = source_row.get("individual")
    source_timestamp = source_row.get("timestamp")

    row["generation"] = int(generation)
    row["individual"] = int(individual)
    row["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    row["evaluation_source"] = evaluation_source
    row["cache_key"] = list(cache_key)
    row["cached_from"] = {
        "generation": source_generation,
        "individual": source_individual,
        "timestamp": source_timestamp,
        "evaluation_source": source_row.get(
            "evaluation_source",
            "RRI_RUN",
        ),
    }
    return row


def annotate_real_run_row(
    row: dict,
    cache_key: CacheKey,
) -> dict:
    """Mark a worker result as a real RRI evaluation."""
    row["evaluation_source"] = "RRI_RUN"
    row["cache_key"] = list(cache_key)
    return row


def evaluate_population_with_cache(
    executor: ProcessPoolExecutor,
    vectors: Sequence[np.ndarray],
    phase: str,
    generation: int,
    display_offset: int,
    base_names: Sequence[str],
    spans: Sequence[Tuple[int, int]],
    gene_keys: Sequence[str],
    fitness_cache: FitnessCache,
    carried_fitness: Optional[
        Sequence[Tuple[float, float, float]]
    ] = None,
) -> List[FitnessTuple]:
    """
    Evaluate a population while preserving NSGA-II multiplicity.

    - If carried_fitness is supplied, parents are not submitted to RRI again.
      Their full prior result is recovered from the cache and logged as a
      carried-parent reuse.
    - Otherwise, cached vectors are reused immediately.
    - Duplicate uncached vectors within the same batch are grouped so only one
      RRI job is submitted; all equivalent individuals receive the same result.
    """
    results: List[Optional[FitnessTuple]] = [None] * len(vectors)
    keys: List[CacheKey] = [
        effective_parameter_key(vector, base_names, spans)
        for vector in vectors
    ]

    # Fast path for selected parents carried from the preceding generation.
    if carried_fitness is not None:
        if len(carried_fitness) != len(vectors):
            raise RuntimeError(
                "Carried parent fitness length does not match population size"
            )

        for zero_index, (individual, key, fitness) in enumerate(
            zip(vectors, keys, carried_fitness)
        ):
            display_index = display_offset + zero_index + 1
            fitness_tuple = tuple(float(value) for value in fitness)
            results[zero_index] = fitness_tuple

            cached = fitness_cache.get(key)
            if cached is None:
                # This should occur only when resuming an old-format checkpoint.
                # Use the supplied carried fitness for selection, but do not
                # fabricate station metrics.
                row = {
                    "calibration_mode": CALIBRATION_MODE,
                    "generation": generation,
                    "individual": display_index,
                    "timestamp": datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    ),
                    "evaluation_source": "CARRIED_PARENT_FITNESS",
                    "cache_key": list(key),
                    "aggregate_objectives": {
                        "MeanNSE": fitness_tuple[0],
                        "MeanNRMSE": fitness_tuple[1],
                        "WorstNAbsBias": fitness_tuple[2],
                        "valid": True,
                        "invalid_stations": [],
                    },
                    "station_metrics": {},
                    "params": {
                        gene: float(value)
                        for gene, value in zip(
                            gene_keys,
                            effective_parameter_values(
                                individual,
                                base_names,
                                spans,
                            ),
                        )
                    },
                }
            else:
                _cached_fitness, source_row = cached
                row = make_reuse_row(
                    source_row,
                    generation,
                    display_index,
                    "CARRIED_PARENT_FITNESS",
                    key,
                )

            append_jsonl(row)
            write_tsv_row(
                generation,
                phase,
                display_index,
                individual,
                gene_keys,
                row,
            )

        return [
            fitness for fitness in results
            if fitness is not None
        ]

    # Group all currently uncached candidates by effective key. This catches
    # duplicates created inside the same offspring batch before any worker has
    # finished.
    waiting_by_key: Dict[CacheKey, List[int]] = {}
    for zero_index, (individual, key) in enumerate(zip(vectors, keys)):
        display_index = display_offset + zero_index + 1
        cached = fitness_cache.get(key)

        if cached is not None:
            fitness, source_row = cached
            results[zero_index] = fitness
            row = make_reuse_row(
                source_row,
                generation,
                display_index,
                "CACHE_REUSE",
                key,
            )
            append_jsonl(row)
            write_tsv_row(
                generation,
                phase,
                display_index,
                individual,
                gene_keys,
                row,
            )
        else:
            waiting_by_key.setdefault(key, []).append(zero_index)

    futures = {}
    for key, zero_indices in waiting_by_key.items():
        representative_index = zero_indices[0]
        individual = vectors[representative_index]
        display_index = display_offset + representative_index + 1

        future = executor.submit(
            evaluate_worker,
            individual.copy(),
            list(base_names),
            list(spans),
            list(gene_keys),
            generation,
            display_index,
        )
        futures[future] = (
            key,
            representative_index,
            display_index,
            individual,
            zero_indices,
        )

    for future in as_completed(futures):
        (
            key,
            representative_index,
            display_index,
            representative,
            zero_indices,
        ) = futures[future]

        mean_nse, mean_nrmse, worst_bias, row = future.result()
        fitness: FitnessTuple = (
            float(mean_nse),
            float(mean_nrmse),
            float(worst_bias),
        )

        row = annotate_real_run_row(row, key)
        fitness_cache[key] = (fitness, copy.deepcopy(row))
        results[representative_index] = fitness

        append_jsonl(row)
        write_tsv_row(
            generation,
            phase,
            display_index,
            representative,
            gene_keys,
            row,
        )

        # Any additional individuals with the same effective parameter vector
        # remain present in NSGA-II, but reuse the representative evaluation.
        for duplicate_index in zero_indices[1:]:
            duplicate_display = display_offset + duplicate_index + 1
            duplicate_vector = vectors[duplicate_index]
            results[duplicate_index] = fitness

            duplicate_row = make_reuse_row(
                row,
                generation,
                duplicate_display,
                "SAME_BATCH_DUPLICATE",
                key,
            )
            append_jsonl(duplicate_row)
            write_tsv_row(
                generation,
                phase,
                duplicate_display,
                duplicate_vector,
                gene_keys,
                duplicate_row,
            )

    if any(fitness is None for fitness in results):
        raise RuntimeError(
            f"At least one {phase} evaluation did not return a result"
        )

    return [
        fitness for fitness in results
        if fitness is not None
    ]


# =============================================================================
# MAIN
# =============================================================================


def write_final_pareto_and_depth(
    population: Sequence[np.ndarray],
    population_fitness: Sequence[FitnessTuple],
    gene_keys: Sequence[str],
) -> None:
    if not population or not population_fitness:
        return

    objectives = [
        (-mean_nse, mean_nrmse, worst_bias)
        for mean_nse, mean_nrmse, worst_bias
        in population_fitness
    ]
    fronts = fast_nondominated_sort(objectives)
    if not fronts or not fronts[0]:
        return

    first_front = fronts[0]
    elite_index = select_balanced_compromise(
        first_front,
        objectives,
    )

    context = build_geometry_context(PARTICLE_DIR)
    elite_vector = np.asarray(
        population[elite_index],
        dtype=float,
    )
    used_values, diagnostics = generate_depth_candidate(
        context,
        elite_vector,
        FINAL_BEST_DEPTH,
    )

    rows = []
    for index in first_front:
        fitness = population_fitness[index]
        effective = effective_geometry_values(
            np.asarray(population[index], dtype=float)
        )
        row = {
            "PopulationIndex": int(index),
            "MeanNSE": float(fitness[0]),
            "MeanNRMSE": float(fitness[1]),
            "WorstNAbsBias": float(fitness[2]),
            "BalancedCompromise": int(index == elite_index),
        }
        row.update(
            {
                key: float(value)
                for key, value in zip(
                    gene_keys,
                    effective,
                )
            }
        )
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        FINAL_PARETO_TSV,
        sep="\t",
        index=False,
    )

    summary_path = ARCHIVE_DIR / "Final_Best_DepthProfile.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_depth_file": str(FINAL_BEST_DEPTH),
                "pareto_table": str(FINAL_PARETO_TSV),
                "fitness": {
                    "MeanNSE": float(
                        population_fitness[elite_index][0]
                    ),
                    "MeanNRMSE": float(
                        population_fitness[elite_index][1]
                    ),
                    "WorstNAbsBias": float(
                        population_fitness[elite_index][2]
                    ),
                },
                "genes": {
                    key: float(value)
                    for key, value in zip(
                        gene_keys,
                        used_values,
                    )
                },
                "geometry": diagnostics,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print(f"[FINAL] Best depth: {FINAL_BEST_DEPTH}")
    print(f"[FINAL] Pareto table: {FINAL_PARETO_TSV}")
    print(f"[FINAL] Summary: {summary_path}")


def main() -> None:
    # location.txt controls the order of columns in hydro_hr.txt.
    prepare_three_station_location_file()
    ensure_paths()
    np.random.seed(RNG_SEED)

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    (
        _template,
        base_names,
        initial_vector,
        spans,
        gene_keys,
    ) = get_state()

    # Preflight the actual D50 seed and topology on the base model before
    # any worker copies or RRI simulations are started.
    base_context = build_geometry_context(PARTICLE_DIR)

    print("Three-station RRI NSGA-II depth-profile calibration")
    print(f"Calibration mode: {CALIBRATION_MODE}")
    print(f"Seed depth: {PARTICLE_DIR / SEED_DEPTH_REL}")
    print(f"Stations: {', '.join(STATION_IDS)}")
    print(f"Location file: {LOCATION_FILE}")
    print(f"Observation layout: {OBS_INPUT_MODE}")
    print(f"Simulation-output layout: {SIM_OUTPUT_MODE}")
    print(f"Matching mode: {MATCH_MODE}")
    print(
        "Objectives: maximize mean NSE; minimize mean normalized RMSE; "
        "minimize worst normalized absolute bias"
    )
    print(f"Population: {POP_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Parallel workers: {PARALLEL_WORKERS}")
    print(f"Genes: {', '.join(gene_keys)}")
    print(
        "Gene bounds: "
        + ", ".join(
            f"{key}={BOUNDS[key]}"
            for key in gene_keys
        )
    )
    print(
        "Seed vector: "
        + ", ".join(
            f"{key}={value:g}"
            for key, value in zip(
                gene_keys,
                initial_vector,
            )
        )
    )
    print(
        f"Available downstream length below NAN003: "
        f"{base_context['available_downstream_km']:.3f} km"
    )
    print(
        f"Downstream core/taper end: "
        f"{base_context['core_end_km']:.3f} / "
        f"{base_context['taper_end_km']:.3f} km"
    )
    print(
        f"Downstream terminal: "
        f"{base_context['terminal_info']}"
    )
    print(
        f"Detected main-stem confluences with tributary branches: "
        f"{len(base_context['tributary_map'])}"
    )
    print(
        "[DEDUP] Enabled: carried parent fitness + exact "
        "effective-geometry cache"
    )

    # Geometry-only preflight: exact seed must regenerate without running RRI.
    if PREFLIGHT_ONLY:
        preflight_path = (
            PARTICLE_DIR
            / "riv"
            / "_depth_NSGA2_preflight_seed.txt"
        )
        _used, diagnostics = generate_depth_candidate(
            base_context,
            initial_vector,
            preflight_path,
        )
        seed_bytes = (
            PARTICLE_DIR / SEED_DEPTH_REL
        ).read_bytes()
        preflight_bytes = preflight_path.read_bytes()
        exact_match = seed_bytes == preflight_bytes
        print(
            f"[PREFLIGHT] Seed regeneration byte-identical: "
            f"{exact_match}"
        )
        print(
            "[PREFLIGHT] Geometry diagnostics:\n"
            + json.dumps(
                diagnostics,
                indent=2,
                allow_nan=True,
            )
        )
        try:
            preflight_path.unlink()
        except Exception:
            pass
        if not exact_match:
            raise RuntimeError(
                "Preflight seed [0,0,1] did not reproduce the "
                "supplied D50 depth file exactly."
            )
        print("[PREFLIGHT] Complete. RRI was NOT run.")
        return

    population = init_population(
        POP_SIZE,
        base_names,
        initial_vector=initial_vector,
    )
    population_fitness: Optional[List[FitnessTuple]] = None
    fitness_cache: FitnessCache = {}
    generation_start = 1

    checkpoint = load_checkpoint()
    if checkpoint is not None:
        checkpoint_mode = checkpoint.get("calibration_mode")
        if checkpoint_mode not in (None, CALIBRATION_MODE):
            raise RuntimeError(
                "Checkpoint calibration mode does not match this run: "
                f"checkpoint={checkpoint_mode!r}, "
                f"current={CALIBRATION_MODE!r}"
            )

        checkpoint_keys = list(
            checkpoint.get("gene_keys", [])
        )
        checkpoint_population = checkpoint.get(
            "population",
            [],
        )
        checkpoint_population_fitness = checkpoint.get(
            "population_fitness"
        )

        if checkpoint_keys == gene_keys:
            population = [
                np.asarray(individual, dtype=float)
                for individual in checkpoint_population
            ]
        else:
            print(
                "[CKPT] Gene order changed; realigning by unique gene key."
            )
            old_index = {
                key: index
                for index, key in enumerate(checkpoint_keys)
            }
            missing = [
                key for key in gene_keys
                if key not in old_index
            ]
            if missing:
                raise RuntimeError(
                    "Checkpoint cannot be realigned because these genes "
                    "are missing: "
                    + ", ".join(missing)
                )
            population = [
                np.asarray(
                    [
                        individual[old_index[key]]
                        for key in gene_keys
                    ],
                    dtype=float,
                )
                for individual in checkpoint_population
            ]

        if len(population) != POP_SIZE:
            raise RuntimeError(
                "Checkpoint population size does not match GA_POP_SIZE: "
                f"checkpoint={len(population)}, current={POP_SIZE}. "
                "Use the original population size when resuming."
            )

        if checkpoint_population_fitness is not None:
            population_fitness = [
                tuple(float(value) for value in fitness)
                for fitness in checkpoint_population_fitness
            ]
            if len(population_fitness) != len(population):
                print(
                    "[CKPT] Stored population fitness length mismatch; "
                    "parents will be evaluated once."
                )
                population_fitness = None

        loaded_cache = checkpoint.get("fitness_cache")
        if isinstance(loaded_cache, dict):
            fitness_cache = loaded_cache
            print(
                f"[DEDUP] Restored {len(fitness_cache)} cached unique "
                "evaluations from checkpoint."
            )

        generation_start = int(
            checkpoint.get("generation_next", 1)
        )
        rng_state = checkpoint.get("numpy_rng_state")
        if rng_state is not None:
            np.random.set_state(rng_state)
        print(
            f"[RESUME] Continuing from generation {generation_start}"
        )

    write_tsv_header(gene_keys)

    deadline: Optional[float] = None
    if GA_MAX_SECONDS > 0:
        deadline = time.time() + max(
            0,
            GA_MAX_SECONDS - GRACE_SECONDS,
        )
        print(
            f"[BUDGET] Runtime budget active; reserving "
            f"{GRACE_SECONDS} seconds for checkpointing."
        )

    total_rri_runs_before = len(fitness_cache)

    with ProcessPoolExecutor(
        max_workers=PARALLEL_WORKERS,
        initializer=_init_worker,
        initargs=(
            str(PARTICLE_DIR),
            str(WORKERS_ROOT),
        ),
    ) as executor:
        for generation in range(
            generation_start,
            GENERATIONS + 1,
        ):
            generation_cache_before = len(fitness_cache)

            parent_fitness_final = evaluate_population_with_cache(
                executor=executor,
                vectors=population,
                phase="parent",
                generation=generation,
                display_offset=0,
                base_names=base_names,
                spans=spans,
                gene_keys=gene_keys,
                fitness_cache=fitness_cache,
                carried_fitness=population_fitness,
            )

            parent_objectives = [
                (-mean_nse, mean_nrmse, worst_bias)
                for mean_nse, mean_nrmse, worst_bias
                in parent_fitness_final
            ]
            parent_fronts = fast_nondominated_sort(
                parent_objectives
            )
            parent_crowding: Dict[int, float] = {}
            for front in parent_fronts:
                parent_crowding.update(
                    crowding_distance(
                        front,
                        parent_objectives,
                    )
                )

            offspring: List[np.ndarray] = []
            while len(offspring) < len(population):
                index_a = crowded_tournament(
                    population,
                    parent_fronts,
                    parent_crowding,
                    TOURNAMENT_SIZE,
                )
                index_b = crowded_tournament(
                    population,
                    parent_fronts,
                    parent_crowding,
                    TOURNAMENT_SIZE,
                )

                parent_a = population[index_a]
                parent_b = population[index_b]
                child = parent_a.copy()
                if (
                    np.random.rand() < CX_PROB
                    and index_a != index_b
                ):
                    child = uniform_crossover(
                        parent_a,
                        parent_b,
                        base_names,
                        swap_probability=0.5,
                    )
                child = gaussian_mutation(
                    child,
                    base_names,
                )
                offspring.append(child)

            offspring_fitness_final = evaluate_population_with_cache(
                executor=executor,
                vectors=offspring,
                phase="offspring",
                generation=generation,
                display_offset=POP_SIZE,
                base_names=base_names,
                spans=spans,
                gene_keys=gene_keys,
                fitness_cache=fitness_cache,
                carried_fitness=None,
            )

            merged_population = population + offspring
            merged_fitness = (
                parent_fitness_final
                + offspring_fitness_final
            )
            merged_objectives = [
                (-mean_nse, mean_nrmse, worst_bias)
                for mean_nse, mean_nrmse, worst_bias
                in merged_fitness
            ]
            merged_fronts = fast_nondominated_sort(
                merged_objectives
            )

            selected_indices: List[int] = []
            for front in merged_fronts:
                if (
                    len(selected_indices) + len(front)
                    <= POP_SIZE
                ):
                    selected_indices.extend(front)
                else:
                    distances = crowding_distance(
                        front,
                        merged_objectives,
                    )
                    ordered = sorted(
                        front,
                        key=lambda index: distances[index],
                        reverse=True,
                    )
                    remaining = (
                        POP_SIZE - len(selected_indices)
                    )
                    selected_indices.extend(
                        ordered[:remaining]
                    )
                    break

            population = [
                merged_population[index].copy()
                for index in selected_indices
            ]
            population_fitness = [
                merged_fitness[index]
                for index in selected_indices
            ]

            first_front = merged_fronts[0]
            elite_index = select_balanced_compromise(
                first_front,
                merged_objectives,
            )
            elite_fitness = merged_fitness[elite_index]
            elite_vector = merged_population[elite_index]

            generation_new_runs = (
                len(fitness_cache)
                - generation_cache_before
            )
            nominal_evaluations = (
                len(population)
                + len(offspring)
            )
            avoided = max(
                0,
                nominal_evaluations - generation_new_runs,
            )

            print(
                f"[Generation {generation:03d}] "
                f"MeanNSE={elite_fitness[0]:.4f}  "
                f"MeanNRMSE={elite_fitness[1]:.4f}  "
                f"WorstNAbsBias={elite_fitness[2]:.4f}"
            )
            print(
                "[Genes] "
                + ", ".join(
                    f"{key}={value:.5f}"
                    for key, value in zip(
                        gene_keys,
                        effective_geometry_values(
                            elite_vector
                        ),
                    )
                )
            )
            print(
                f"[DEDUP Gen {generation:03d}] "
                f"new RRI runs={generation_new_runs}; "
                f"nominal evaluations={nominal_evaluations}; "
                f"RRI runs avoided={avoided}; "
                f"unique cache size={len(fitness_cache)}"
            )

            elite_entry = {
                "calibration_mode": CALIBRATION_MODE,
                "generation": generation,
                "selection_rule": (
                    "normalized distance to ideal point within "
                    "first Pareto front"
                ),
                "aggregate_metrics": {
                    "MeanNSE": float(elite_fitness[0]),
                    "MeanNRMSE": float(elite_fitness[1]),
                    "WorstNAbsBias": float(elite_fitness[2]),
                },
                "params": {
                    key: float(value)
                    for key, value in zip(
                        gene_keys,
                        effective_geometry_values(
                            elite_vector
                        ),
                    )
                },
                "dedup": {
                    "new_rri_runs_this_generation": int(
                        generation_new_runs
                    ),
                    "nominal_evaluations": int(
                        nominal_evaluations
                    ),
                    "rri_runs_avoided": int(avoided),
                    "unique_cache_size": int(
                        len(fitness_cache)
                    ),
                },
            }

            save_checkpoint(
                generation_next=generation + 1,
                gene_keys=gene_keys,
                population=population,
                population_fitness=population_fitness,
                fitness_cache=fitness_cache,
                rng_state=np.random.get_state(),
                elite_entry=elite_entry,
            )

            if STOP_REQUESTED:
                print(
                    "[STOP] Stop requested; exiting after checkpoint."
                )
                return

            if (
                deadline is not None
                and time.time() > deadline
            ):
                print(
                    "[BUDGET] Runtime budget reached; "
                    "exiting after checkpoint."
                )
                return

    if population_fitness is not None:
        write_final_pareto_and_depth(
            population,
            population_fitness,
            gene_keys,
        )

    if (
        CHECKPOINT_FILE.exists()
        and DELETE_CHECKPOINT_ON_SUCCESS
    ):
        try:
            CHECKPOINT_FILE.unlink()
            print(
                "[CKPT] Completed run; removed checkpoint file."
            )
        except Exception:
            pass

    total_unique_runs = (
        len(fitness_cache) - total_rri_runs_before
    )
    print(
        f"\n[DEDUP] New unique RRI evaluations this session: "
        f"{total_unique_runs}"
    )
    print(
        f"[DEDUP] Final unique cache size: "
        f"{len(fitness_cache)}"
    )
    print(f"\nJSONL log: {JSONL_LOG}")
    print(f"TSV log:   {TSV_LOG}")


if __name__ == "__main__":
    main()
