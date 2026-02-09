#!/usr/bin/env python3
"""Compute offline statistics and greedy baselines for GreenFL scheduling.

This script performs two tasks:

1. Compute offline CAFE (https://arxiv.org/abs/2311.03615) and FedZero (https://arxiv.org/abs/2305.15092)
   statistics using a probe subset of each client's data.
2. Solve three scheduling baselines under a shared carbon budget and forced
   fine-tuning window using lightweight greedy heuristics:

   - **Problem 1 (alpha-fairness)**: maximise ``sum_c (sum_t (g_max - g_c^t) a_c^t)^alpha``.
   - **Problem 2 (CAFE-like)**: maximise ``(1/H) sum_t (b - sum_j min_{i in K_t(A)} d_{i,j})``.
   - **Problem 3 (FedZero-like)**: maximise ``sum_c sigma_c * sum_t a_c^t``.

Shared constraints:
    - Carbon budget: ``sum_{c,t} g_c^t a_c^t <= k``.
    - Fine-tuning window ``F(s) = {T+s-t_ft+1, ..., T+s}`` forces availability ``a_c^t = 1``.
    - No availability after ``T+s``.

Outputs
    - CSV and NPY availability matrices for each problem.
    - Optional plot comparing availability matrices.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import random
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Torch is required for offline stats.
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Robust imports: add repo root to sys.path (so "fl_training" and "building_availability_matrices" resolve).
# ---------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_REPO_ROOT: Optional[Path] = None
for root in [
    _THIS.parents[0],
    _THIS.parents[1],
    _THIS.parents[2],
    _THIS.parents[3],
    Path.cwd(),
]:
    if (root / "fl_training").exists() and (
        root / "building_availability_matrices"
    ).exists():
        _REPO_ROOT = root
        break
if _REPO_ROOT is None:
    _REPO_ROOT = Path.cwd()
sys.path.insert(0, str(_REPO_ROOT))

from fl_training.utils.constants import LOADER_TYPE  # type: ignore
from fl_training.utils.offline_probing import compute_cafe_and_fedzero_stats
from fl_training.utils.utils import get_data_dir, get_loaders, get_learner  # type: ignore

from building_availability_matrices.av_mat_generation.CI_based.window import Window  # type: ignore

_META = {
    "mnist": {
        "3ft_cb": [
            7.980065999999999,
            6.1960109999999995,
            4.235079,
            1.983414,
            1.773765,
            0.9428550000000001,
            0.7447260000000001,
        ],
        "3ft_n_rounds": [40, 30, 20, 10, 9, 5, 4],
        "1ft_cb": [
            7.980065999999999,
            6.1960109999999995,
            4.235079,
            1.983414,
            0.9428550000000001,
            0.7447260000000001,
            0.5575350000000001,
        ],
        "1ft_n_rounds": [40, 30, 20, 10, 5, 4, 3],
    },
    "cifar10": {
        "3ft_cb": [
            20.780553,
            15.540347999999996,
            11.732774999999997,
            8.838905999999998,
            6.1960109999999995,
            3.032562,
            0.9428550000000001,
        ],
        "3ft_n_rounds": [90, 75, 60, 45, 30, 15, 5],
        "1ft_cb": [
            20.780553,
            15.540347999999996,
            11.732774999999997,
            8.838905999999998,
            6.1960109999999995,
            3.032562,
            0.9428550000000001,
            0.5575350000000001,
        ],
        "1ft_n_rounds": [90, 75, 60, 45, 30, 15, 5, 3],
    },
}
_3FT_CB = [
    7.980065999999999,
    6.1960109999999995,
    4.235079,
    1.983414,
    1.773765,
    0.9428550000000001,
    0.7447260000000001,
]
_3FT_N_ROUNDS = [40, 30, 20, 10, 9, 5, 4]
_1FT_CB = [
    7.980065999999999,
    6.1960109999999995,
    4.235079,
    1.983414,
    0.9428550000000001,
    0.7447260000000001,
    0.5575350000000001,
]
_1FT_N_ROUNDS = [40, 30, 20, 10, 5, 4, 3]


@dataclass
class OfflineProbeClient:
    """Minimal client representation used during offline probing."""

    id: int
    train_iterator: DataLoader


@dataclass
class SolveResult:
    A_full: np.ndarray  # (K, H) binary
    s_best: int
    objective_value: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute offline stats (CAFE/FedZero) and solve 3 availability-matrix problems (greedy)."
    )

    # ---- Carbon scheduling parameters
    p.add_argument(
        "--T", type=int, required=True, help="Base training horizon (rounds)."
    )
    p.add_argument("--t-sl", type=int, required=True, help="Slack time (rounds).")
    p.add_argument(
        "--t-ft", type=int, required=True, help="Fine-tuning window length (rounds)."
    )
    p.add_argument(
        "--alpha", type=float, default=0.1, help="Alpha for Problem 1 objective."
    )
    p.add_argument(
        "--budget", type=int, required=True, help="Index of Carbon budget k."
    )
    p.add_argument(
        "--countries",
        type=str,
        required=True,
        help="Comma-separated list (must match CI dataset keys).",
    )

    # ---- Window / CI data options
    p.add_argument(
        "--window-out-folder",
        type=str,
        default=str(_REPO_ROOT / "building_availability_matrices"),
        help="Path that contains historical_data/ (passed to Window(..., out_folder=...)).",
    )
    p.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="ISO start time. If omitted, uses random_start.",
    )
    p.add_argument(
        "--random-start", action=argparse.BooleanOptionalAction, default=True
    )

    # ---- Offline stats options
    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="mnist, cifar10, cifar100, femnist, etc.",
    )
    p.add_argument("--model-name", type=str, default="mnist_cnn")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--bz", type=int, default=64)
    p.add_argument("--probe-fraction", type=float, default=0.05)
    p.add_argument("--min-probe-samples", type=int, default=32)
    p.add_argument("--max-probe-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=12345)

    p.add_argument(
        "--offline-stats-path",
        type=str,
        default="",
        help="If set and exists, load stats from here instead of recomputing.",
    )
    p.add_argument(
        "--save-offline-stats",
        type=str,
        default="",
        help="If set, save computed stats to this path (.pt).",
    )

    # ---- Solver options
    p.add_argument(
        "--s-step",
        type=int,
        default=1,
        help="Evaluate s in {1, 1+s_step, ...} up to t_sl.",
    )

    # ---- Output & plotting
    p.add_argument("--out-dir", type=str, default="availability_matrices")
    p.add_argument("--name-prefix", type=str, default="")
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--plot-save",
        type=str,
        default="",
        help="If set, saves comparison plot to this path (png/pdf).",
    )

    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def parse_countries(countries_arg: str) -> List[str]:
    return [c.strip() for c in countries_arg.split(",") if c.strip()]


def parse_start_time(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    ss = s.strip().replace(" ", "T")
    try:
        return datetime.fromisoformat(ss)
    except Exception as e:
        raise ValueError(
            f"Could not parse --start-time '{s}'. Use ISO like '2022-01-01T00:00:00'."
        ) from e


def build_probe_clients(
    experiment: str, batch_size: int, verbose: int
) -> List[OfflineProbeClient]:
    """Construct per-client training loaders for offline probing."""

    if verbose:
        print("==> Building per-client DataLoaders from generated data..")

    data_root = get_data_dir(experiment)
    train_dir = os.path.join(data_root, "train")

    train_iterators, _, _ = get_loaders(
        type_=LOADER_TYPE[experiment],
        data_dir=train_dir,
        batch_size=batch_size,
        is_validation=False,
    )

    clients: List[OfflineProbeClient] = []
    for task_id, train_iterator in enumerate(train_iterators):
        if train_iterator is None:
            continue
        clients.append(OfflineProbeClient(id=task_id, train_iterator=train_iterator))

    if verbose:
        print(f"Built {len(clients)} clients with non-empty train iterators.")
    if not clients:
        raise RuntimeError("Clients datasets are empty!")
    return clients


def compute_or_load_offline_stats(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    if args.offline_stats_path and Path(args.offline_stats_path).exists():
        if args.verbose:
            print(f"==> Loading offline stats from: {args.offline_stats_path}")
        payload = torch.load(args.offline_stats_path, map_location="cpu")
        D = payload["distance_matrix"]
        sigma = payload["sigma"]
        return np.asarray(D, dtype=float), np.asarray(sigma, dtype=float)

    torch.manual_seed(args.seed)
    clients = build_probe_clients(args.experiment, args.bz, args.verbose)

    if args.verbose:
        print("==> Initializing reference learner (w^ref)..")
    _tmp_pth = _META[args.experiment][f"{args.t_ft}ft_n_rounds"][args.budget]
    reference_learner = get_learner(
        name=args.experiment,
        model_name=args.model_name,
        device=args.device,
        optimizer_name="sgd",
        scheduler_name="constant",
        initial_lr=0.01,
        mu=0.0,
        n_rounds=1,
        seed=args.seed,
        input_dimension=getattr(args, "input_dimension", None),
        hidden_dimension=getattr(args, "hidden_dimension", None),
        chkpts_path=f"../baseline_chkpt/{args.experiment}/global_{_tmp_pth}.pt",
        history_coefficient=1.0,
    )

    if args.verbose:
        print(
            "==> Computing CAFE (distance matrix) and FedZero (sigma) offline stats.."
        )

    D_cafe, sigma_fedzero, stats = compute_cafe_and_fedzero_stats(
        reference_learner=reference_learner,
        clients=clients,
        probe_fraction=args.probe_fraction,
        min_probe_samples=args.min_probe_samples,
        max_probe_samples=args.max_probe_samples,
        base_seed=args.seed,
    )

    D = np.asarray(D_cafe, dtype=float)
    sigma = np.asarray(sigma_fedzero, dtype=float)

    if args.save_offline_stats:
        out_path = Path(args.save_offline_stats)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "distance_matrix": D_cafe,
            "sigma": sigma_fedzero,
            "stats": stats,
            "experiment": args.experiment,
            "seed": args.seed,
            "probe_fraction": args.probe_fraction,
            "min_probe_samples": args.min_probe_samples,
            "max_probe_samples": args.max_probe_samples,
        }
        torch.save(payload, str(out_path))
        if args.verbose:
            print(f"Offline stats saved to: {out_path}")

    if args.verbose:
        print(f"  Distance matrix shape: {D.shape}")
        print(f"  Sigma shape:           {sigma.shape}")
        print(
            f"  Sigma stats: min={sigma.min():.4f}, max={sigma.max():.4f}, mean={sigma.mean():.4f}"
        )

    return D, sigma


def load_ghg_matrix(
    args: argparse.Namespace, countries: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Load the carbon-intensity matrix ``g_c^t`` and associated labels."""

    horizon = args.T + args.t_sl
    start_time = parse_start_time(args.start_time)
    if start_time is None and not args.random_start:
        if args.verbose:
            print(
                "[GreenFL] WARNING: --no-random-start with no --start-time can crash Window. Enabling random_start=True."
            )
        args.random_start = True

    window = Window(
        start_time=start_time,
        random_start=args.random_start,
        out_folder=args.window_out_folder,
        n_rounds=horizon,
        countries=countries,
    )

    ghg_dataframe = window.get_GHG_matrix()
    row_labels = (
        list(ghg_dataframe.index)
        if hasattr(ghg_dataframe, "index")
        else list(countries)
    )
    ghg_matrix = (
        ghg_dataframe.to_numpy(dtype=float)
        if hasattr(ghg_dataframe, "to_numpy")
        else np.asarray(ghg_dataframe, dtype=float)
    )

    if not np.isfinite(ghg_matrix).all():
        raise ValueError("GHG matrix contains NaNs/Infs. Check CI data parsing.")
    return ghg_matrix, row_labels


def _forced_window_indices(T: int, t_ft: int, s: int) -> Tuple[int, int]:
    ft_start = T + s - t_ft
    ft_end = T + s
    return ft_start, ft_end


def _initialize_A_base(K: int, H: int, ft_start: int, ft_end: int) -> np.ndarray:
    A = np.zeros((K, H), dtype=int)
    A[:, ft_start:ft_end] = 1
    if ft_end < H:
        A[:, ft_end:] = 0
    return A


def solve_problem1_alpha_fair_greedy(
    GHG_mat: np.ndarray,
    T: int,
    t_sl: int,
    t_ft: int,
    alpha: float,
    budget_k: float,
    s_step: int = 1,
) -> SolveResult:
    K, H = GHG_mat.shape
    g_max = float(np.max(GHG_mat))
    U = g_max - GHG_mat

    best_val = -1e300
    best_A = None
    best_s = None

    for s in range(1, t_sl + 1, max(1, int(s_step))):
        ft_start, ft_end = _forced_window_indices(T, t_ft, s)
        if ft_start < 0 or ft_end > H:
            continue

        forced_cost = float(GHG_mat[:, ft_start:ft_end].sum())
        if forced_cost > budget_k + 1e-12:
            continue

        A = _initialize_A_base(K, H, ft_start, ft_end)

        x = np.sum(U[:, ft_start:ft_end], axis=1).astype(float)
        remaining = float(budget_k - forced_cost)

        while True:
            best_ratio = -1.0
            best = None

            for c in range(K):
                xc = x[c]
                xc_a = xc**alpha
                for t in range(ft_start):
                    if A[c, t] == 1:
                        continue
                    cost = float(GHG_mat[c, t])
                    if cost > remaining + 1e-12:
                        continue
                    u = float(U[c, t])
                    if u <= 0:
                        continue
                    marg = (xc + u) ** alpha - xc_a
                    if marg <= 0:
                        continue
                    ratio = float("inf") if cost <= 0 else (marg / cost)
                    if ratio > best_ratio + 1e-15:
                        best_ratio = ratio
                        best = (c, t, cost, u)

            if best is None:
                break
            c, t, cost, u = best
            if cost > remaining + 1e-12:
                break
            A[c, t] = 1
            remaining -= cost
            x[c] += u
            if remaining <= 1e-12:
                break

        val = float(np.sum(np.power(x, alpha)))
        if val > best_val:
            best_val, best_A, best_s = val, A, s

    if best_A is None or best_s is None:
        raise RuntimeError(
            "Problem 1: no feasible solution found (forced FT likely exceeds budget)."
        )
    return SolveResult(A_full=best_A, s_best=best_s, objective_value=best_val)


def solve_problem2_cafe_greedy(
    GHG_mat: np.ndarray,
    D: np.ndarray,
    T: int,
    t_sl: int,
    t_ft: int,
    budget_k: float,
    s_step: int = 1,
) -> SolveResult:
    K, H = GHG_mat.shape
    if D.shape != (K, K):
        raise ValueError(f"D has shape {D.shape}, but expected ({K},{K}).")

    max_d = float(np.max(D))
    b = float(2 * K * max_d)
    m_all = np.min(D, axis=0)

    best_val = -1e300
    best_A = None
    best_s = None

    for s in range(1, t_sl + 1, max(1, int(s_step))):
        ft_start, ft_end = _forced_window_indices(T, t_ft, s)
        if ft_start < 0 or ft_end > H:
            continue

        forced_cost = float(GHG_mat[:, ft_start:ft_end].sum())
        if forced_cost > budget_k + 1e-12:
            continue

        A = _initialize_A_base(K, H, ft_start, ft_end)
        remaining = float(budget_k - forced_cost)

        m = np.full((H, K), max_d, dtype=float)
        m[ft_start:ft_end, :] = m_all

        sum_m = float(m.sum())
        current_val = b - (sum_m / H)

        while True:
            best_ratio = -1.0
            best = None  # (c, t, cost, new_mt)

            for t in range(ft_start):
                if np.all(A[:, t] == 1):
                    continue
                mt = m[t]
                for c in range(K):
                    if A[c, t] == 1:
                        continue
                    cost = float(GHG_mat[c, t])
                    if cost > remaining + 1e-12:
                        continue

                    dc = D[c]
                    new_mt = np.minimum(mt, dc)
                    improv = float((mt - new_mt).sum()) / H
                    if improv <= 1e-15:
                        continue
                    ratio = float("inf") if cost <= 0 else (improv / cost)
                    if ratio > best_ratio + 1e-15:
                        best_ratio = ratio
                        best = (c, t, cost, new_mt)

            if best is None:
                break

            c, t, cost, new_mt = best
            if cost > remaining + 1e-12:
                break

            A[c, t] = 1
            remaining -= cost

            sum_m += float(new_mt.sum() - m[t].sum())
            m[t] = new_mt
            current_val = b - (sum_m / H)

            if remaining <= 1e-12:
                break

        if current_val > best_val:
            best_val, best_A, best_s = current_val, A, s

    if best_A is None or best_s is None:
        raise RuntimeError(
            "Problem 2: no feasible solution found (forced FT likely exceeds budget)."
        )
    return SolveResult(A_full=best_A, s_best=best_s, objective_value=best_val)


def solve_problem3_fedzero_greedy(
    GHG_mat: np.ndarray,
    sigma: np.ndarray,
    T: int,
    t_sl: int,
    t_ft: int,
    budget_k: float,
    s_step: int = 1,
) -> SolveResult:
    K, H = GHG_mat.shape
    if sigma.shape not in [(K,), (K, 1)]:
        raise ValueError(f"sigma has shape {sigma.shape}, but expected ({K},)")
    sigma = sigma.reshape(-1).astype(float)

    best_val = -1e300
    best_A = None
    best_s = None

    for s in range(1, t_sl + 1, max(1, int(s_step))):
        ft_start, ft_end = _forced_window_indices(T, t_ft, s)
        if ft_start < 0 or ft_end > H:
            continue

        forced_cost = float(GHG_mat[:, ft_start:ft_end].sum())
        if forced_cost > budget_k + 1e-12:
            continue

        A = _initialize_A_base(K, H, ft_start, ft_end)
        remaining = float(budget_k - forced_cost)

        val = float(np.sum(sigma) * (ft_end - ft_start))

        candidates = []
        for t in range(ft_start):
            for c in range(K):
                cost = float(GHG_mat[c, t])
                ratio = float("inf") if cost <= 0 else float(sigma[c] / cost)
                candidates.append((ratio, c, t, cost))
        candidates.sort(reverse=True, key=lambda x: x[0])

        for ratio, c, t, cost in candidates:
            if A[c, t] == 1:
                continue
            if cost > remaining + 1e-12:
                continue
            A[c, t] = 1
            remaining -= cost
            val += float(sigma[c])
            if remaining <= 1e-12:
                break

        if val > best_val:
            best_val, best_A, best_s = val, A, s

    if best_A is None or best_s is None:
        raise RuntimeError(
            "Problem 3: no feasible solution found (forced FT likely exceeds budget)."
        )
    return SolveResult(A_full=best_A, s_best=best_s, objective_value=best_val)


def save_availability(
    A_full: np.ndarray,
    row_labels: List[str],
    out_dir: Path,
    name: str,
    T: int,
    s_best: int,
) -> Tuple[Path, Path]:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    end_round = T + s_best

    df = pd.DataFrame(
        A_full[:, :end_round], index=row_labels, columns=list(range(1, end_round + 1))
    )
    csv_path = out_dir / f"{name}_{end_round}Rounds.csv"
    df.to_csv(csv_path)

    npy_path = out_dir / f"{name}_{end_round}Rounds_full.npy"
    np.save(npy_path, A_full.astype(int))
    return csv_path, npy_path


def _setup_cmap():
    NO_COLOR = "red"
    YES_COLOR = "green"
    cmap = ListedColormap([NO_COLOR, YES_COLOR])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    return cmap, norm


def plot_three_matrices(
    results: List[Tuple[str, SolveResult]],
    row_labels: List[str],
    T: int,
    t_ft: int,
    t_sl: int,
    budget: int,
    plot_save: str = "",
    show: bool = True,
) -> None:
    if not results:
        return
    K = len(row_labels)
    H = results[0][1].A_full.shape[1]

    cmap, norm = _setup_cmap()

    fig_h = max(6.0, 2.2 * len(results))
    fig_w = max(12.0, H / 10)
    fig, axs = plt.subplots(len(results), 1, figsize=(fig_w, fig_h), sharex=True)
    if len(results) == 1:
        axs = [axs]

    max_ticks = 18
    step = max(1, H // max_ticks)
    base_ticks = set(range(0, H, step))
    base_ticks.add(H - 1)
    if 0 <= T - 1 < H:
        base_ticks.add(T - 1)

    for ax, (label, res) in zip(axs, results):
        A = np.asarray(res.A_full, dtype=int)
        ax.imshow(A, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

        ax.set_yticks(range(K))
        ax.set_yticklabels(row_labels)

        end_round = T + res.s_best
        ft_start = end_round - t_ft
        ft_end = end_round

        ticks = set(base_ticks)
        if 0 <= end_round - 1 < H:
            ticks.add(end_round - 1)
        if 0 <= ft_start < H:
            ticks.add(ft_start)
        if 0 <= ft_end - 1 < H:
            ticks.add(ft_end - 1)
        ticks = sorted(ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t + 1) for t in ticks])

        ax.axvline(T - 0.5, linewidth=1, linestyle="--")
        ax.axvline(ft_start - 0.5, linewidth=1)
        ax.axvline(ft_end - 0.5, linewidth=1)
        ax.axvline(end_round - 0.5, linewidth=1.2, linestyle="-")
        ax.axvspan(ft_start - 0.5, ft_end - 0.5, alpha=0.12)

        ax.set_ylabel("Client")
        ax.set_title(
            f"{label} | {t_sl}sl-{budget}cb-{t_ft}ft | best s={res.s_best}, obj={res.objective_value:.6f}"
        )

    axs[-1].set_xlabel("Round")
    fig.tight_layout()

    if plot_save:
        out = Path(plot_save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[GreenFL] Saved comparison plot: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _actual_budget = _META[args.experiment][f"{args.t_ft}ft_cb"][args.budget]

    if args.verbose:
        print("Arguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print(f"  Carbon budget: {_actual_budget}")

    countries = parse_countries(args.countries)
    if args.verbose:
        print(f"[GreenFL] Countries ({len(countries)}): {countries}")

    D, sigma = compute_or_load_offline_stats(args)
    GHG_mat, row_labels = load_ghg_matrix(args, countries)

    K, H = GHG_mat.shape
    if D.shape[0] != K or sigma.shape[0] != K:
        raise RuntimeError(
            f"Mismatch in K:\\n"
            f"  - CI data (GHG_mat) has K={K}\\n"
            f"  - offline distance matrix has shape {D.shape}\\n"
            f"  - sigma has shape {sigma.shape}\\n"
            f"Make sure your dataset generation created exactly {K} clients."
        )

    if args.verbose:
        print("==> Solving Problem 1 (alpha-fairness) with greedy..")
    res1 = solve_problem1_alpha_fair_greedy(
        GHG_mat, args.T, args.t_sl, args.t_ft, args.alpha, _actual_budget, args.s_step
    )

    if args.verbose:
        print("==> Solving Problem 2 (CAFE-like) with greedy..")
    res2 = solve_problem2_cafe_greedy(
        GHG_mat, D, args.T, args.t_sl, args.t_ft, _actual_budget, args.s_step
    )

    if args.verbose:
        print("==> Solving Problem 3 (FedZero-like) with greedy..")
    res3 = solve_problem3_fedzero_greedy(
        GHG_mat, sigma, args.T, args.t_sl, args.t_ft, _actual_budget, args.s_step
    )

    out_dir = Path(args.out_dir)
    prefix = (
        args.name_prefix.strip()
        or f"T-{args.T}_tsl-{args.t_sl}_tft-{args.t_ft}_k-{args.budget}"
    )

    csv1, npy1 = save_availability(
        res1.A_full,
        row_labels,
        out_dir,
        f"{prefix}_prob1_alphaFair_{args.budget+1}cb_{args.t_ft}ft",
        args.T,
        res1.s_best,
    )
    csv2, npy2 = save_availability(
        res2.A_full,
        row_labels,
        out_dir,
        f"{prefix}_prob2_CAFE_{args.budget+1}cb_{args.t_ft}ft",
        args.T,
        res2.s_best,
    )
    csv3, npy3 = save_availability(
        res3.A_full,
        row_labels,
        out_dir,
        f"{prefix}_prob3_FedZero_{args.budget+1}cb_{args.t_ft}ft",
        args.T,
        res3.s_best,
    )

    print("[GreenFL] Saved availability matrices:")
    print(f"  Problem 1 CSV: {csv1}")
    print(f"  Problem 1 NPY: {npy1}")
    print(f"  Problem 2 CSV: {csv2}")
    print(f"  Problem 2 NPY: {npy2}")
    print(f"  Problem 3 CSV: {csv3}")
    print(f"  Problem 3 NPY: {npy3}")

    if args.plot or args.plot_save:
        plot_three_matrices(
            results=[
                ("Problem 1: alpha-fair", res1),
                ("Problem 2: CAFE", res2),
                ("Problem 3: FedZero", res3),
            ],
            row_labels=row_labels,
            T=args.T,
            t_ft=args.t_ft,
            t_sl=args.t_sl,
            budget=_actual_budget,
            plot_save=args.plot_save,
            show=args.plot,
        )


if __name__ == "__main__":
    main()
