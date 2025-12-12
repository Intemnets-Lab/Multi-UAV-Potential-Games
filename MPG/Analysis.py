import os
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List
import math
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.ticker import PercentFormatter

# ============================================================
# Global configuration & style
# ============================================================

class Config:
    def __init__(self, data: dict):
        # ----- project paths -----
        proj = data.get("project", {})
        # Strings relative to project root (where Analysis.py lives)
        self.results_dir = proj.get("results_dir", "Results")
        self.irada_benchmarking_dir = proj.get("IRADA_benchmarking_dir", "Benchmarking/IRADA")
        self.visualization_dir = proj.get("visualization_dir", "Visualizations")

        # ----- simulation -----
        self.n_runs = data["simulation"]["n_runs"]

        # ----- grid -----
        self.grid_width = data["grid"]["width"]
        self.grid_height = data["grid"]["height"]
        self.grid_spacing = data["grid"]["spacing"]

        # ----- uav -----
        self.num_uavs = data["uav"]["num_uavs"]
        self.speed = data["uav"]["speed"]
        self.max_flight_time = data["uav"]["max_flight_time"]

        # ----- algorithms -----
        for algo, enabled in data.get("algorithms", {}).items():
            setattr(self, algo, enabled)

    @classmethod
    def from_yaml(cls, path="settings.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def override(self, overrides: dict):
        """Apply CLI overrides (flat keys)."""
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self):
        return (
            f"Config(num_uavs={self.num_uavs}, "
            f"grid=({self.grid_width}x{self.grid_height}), "
            f"speed={self.speed}, max_flight_time={self.max_flight_time})"
        )


# Master switches
GraphGeneration: bool = True
GifGeneration: bool = False

# Plot style (adjust these to your needs)
FONT_FAMILY = "Times New Roman"
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 24

# ============================================================
# OPTIONAL: per-mode manual overrides of date/simulation
# ============================================================
# NonOverlap overrides
NON_DATE: str | None = "2025-12-07"      # e.g. "2025-12-01"
NON_SIM:  str | None = "simulation_1"      # e.g. "simulation_1"

# Overlap overrides
OVER_DATE: str | None = "2025-12-07"
OVER_SIM:  str | None = "simulation_1"

# IRADA overrides
IRADA_DATE: str | None = "2025-12-07"
IRADA_SIM:  str | None = "simulation_2"


def set_plot_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE,
    })


# ============================================================
# Helpers: simulation discovery & labels
# ============================================================

def find_latest_simulation(root: Path) -> Path:
    """
    Pick newest date folder under root, then highest simulation_N under that date.
    Expected structure: root/YYYY-MM-DD/simulation_N/
    """
    dates = sorted(p for p in root.iterdir() if p.is_dir())
    if not dates:
        raise RuntimeError(f"No date folders under {root}")
    latest_date = dates[-1]

    sims = []
    for d in latest_date.iterdir():
        if d.is_dir() and d.name.startswith("simulation_"):
            try:
                idx = int(d.name.split("_", 1)[1])
                sims.append((idx, d))
            except ValueError:
                pass
    if not sims:
        raise RuntimeError(f"No simulation_*/ folders under {latest_date}")
    return sorted(sims)[-1][1]

def _short_algo_label(game_label: str, algo_label: str) -> str:
    """
    Map (game_label, algo_label) to short taxonomy codes:

      game_label: "NonOverlap" -> N, "Overlap" -> O, IRADA stays "IRADA"
      order:      Sequential -> S, Random -> R
      mode:       ModeGG, ModeGR, ModeRG, ModeRR -> GG, GR, RG, RR

    Examples:
      game_label="NonOverlap", algo_label="ModeGG_Sequential" -> "NSGG"
      game_label="Overlap",    algo_label="ModeGR_Random"     -> "ORGR"
    """
    # IRADA stays as-is
    if algo_label == "IRADA" or game_label == "IRADA":
        return "IRADA"

    # Game code
    if game_label == "NonOverlap":
        game_code = "N"
    elif game_label == "Overlap":
        game_code = "O"
    else:
        game_code = game_label[:1].upper() if game_label else "?"

    # Expect algo_label like "ModeGG_Sequential"
    parts = algo_label.split("_")
    mode_token = parts[0] if parts else ""
    order_token = parts[1] if len(parts) > 1 else ""

    # Mode suffix: GG / GR / RG / RR
    if mode_token.startswith("Mode"):
        mode_suffix = mode_token.replace("Mode", "")  # "ModeGG" -> "GG"
    else:
        mode_suffix = mode_token or "?"

    # Order code: S for Sequential, R for Random
    order_lower = order_token.lower()
    if order_lower.startswith("seq"):
        order_code = "S"
    elif order_lower.startswith("rand"):
        order_code = "R"
    else:
        order_code = "?"

    return f"{game_code}{order_code}{mode_suffix}"


def _label_from_revenue_file(f: Path) -> str:
    """
    Create a concise label for a revenue workbook path.

    - IRADA file (contains 'IRADA'): label 'IRADA'
    - New pattern: UAVsN_GRIDM_ModeXX_Algo  -> 'ModeXX_Algo'
    - Fallback: file stem
    """
    stem = f.stem
    if "IRADA" in stem:
        return "IRADA"
    parts = stem.split("_")
    # e.g. UAVs2_GRID5_ModeGR_Random
    if len(parts) >= 4 and parts[2].startswith("Mode"):
        return f"{parts[2]}_{parts[3]}"
    return stem

def _mode_from_path(p: Path) -> str:
    """Return 'NonOverlap', 'Overlap', 'IRADA', or 'Other' based on path components."""
    parts = p.parts
    if "NonOverlap" in parts:
        return "NonOverlap"
    if "Overlap" in parts:
        return "Overlap"
    if "IRADA" in parts or "Benchmarking" in parts:
        return "IRADA"
    return "Other"


def vis_plots_root_for_sim(vis_root: Path, mode: str, rev_sim_path: Path) -> Path:
    """
    Build Visualizations/<mode>/plots/YYYY-MM-DD/simulation_k.
    rev_sim_path is the revenue simulation directory: .../revenue/YYYY-MM-DD/simulation_k
    """
    date = rev_sim_path.parent.name      # YYYY-MM-DD
    sim_name = rev_sim_path.name         # simulation_k
    return vis_root / mode / "plots" / date / sim_name


def vis_gifs_root_for_sim(vis_root: Path, mode: str, seq_sim_path: Path) -> Path:
    """
    Build Visualizations/<mode>/gifs/YYYY-MM-DD/simulation_k.
    seq_sim_path is the sequences simulation dir: .../sequences/YYYY-MM-DD/simulation_k
    """
    date = seq_sim_path.parent.name
    sim_name = seq_sim_path.name
    return vis_root / mode / "gifs" / date / sim_name


def vis_comparisons_root(vis_root: Path, non_rev_sim: Path) -> Path:
    """
    Build Visualizations/Comparisons/YYYY-MM-DD/simulation_k using the NonOverlap
    revenue sim path as the reference for date & sim index.
    """
    date = non_rev_sim.parent.name
    sim_name = non_rev_sim.name
    return vis_root / "Comparisons" / date / sim_name


# ============================================================
# Per-mode revenue graphs
# ============================================================

def analyze_revenue_excels_graphs(excel_dir: str,
                                  out_root: str | Path | None = None):
    """
    For each revenue .xlsx in `excel_dir`, produce:
      1) One plot per UAV: mean±std shading per round.
      2) One plot for total revenue rate: mean±std per round.
      3) One consolidated plot: all UAV means + mean(total)/n_uavs.

    If out_root is given, plots are saved under:
        out_root/analysis_plots/<file-stem>/
    Otherwise they are saved under:
        excel_dir/analysis_plots/<file-stem>/
    """
    excel_dir = Path(excel_dir)
    if out_root is None:
        plots_root = excel_dir / "analysis_plots"
    else:
        plots_root = Path(out_root) / "analysis_plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    files = [f for f in excel_dir.iterdir()
             if f.suffix == ".xlsx" and not f.stem.endswith("_stats")]

    for fpath in files:
        base = fpath.stem

        sheets = pd.read_excel(fpath, sheet_name=None, index_col=0)
        dfs = list(sheets.values())
        if not dfs:
            continue

        max_rounds = max(df.shape[0] for df in dfs)
        padded = [df.reindex(range(max_rounds), method="ffill") for df in dfs]

        uav_cols = [c for c in padded[0].columns
                    if str(c).upper().startswith("UAV")]
        if not uav_cols:
            continue
        n_uavs = len(uav_cols)

        arr = np.stack([df[uav_cols].values for df in padded], axis=0)
        mean_uav = arr.mean(axis=0)
        std_uav = arr.std(axis=0, ddof=1)

        tot_arr = arr.sum(axis=2)
        mean_tot = tot_arr.mean(axis=0)
        std_tot = tot_arr.std(axis=0, ddof=1)
        rounds = np.arange(max_rounds)

        out_dir = plots_root / base
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Per-UAV plots
        for i, col in enumerate(uav_cols):
            plt.figure()
            plt.plot(rounds, mean_uav[:, i], label=f"mean {col}")
            plt.fill_between(
                rounds,
                mean_uav[:, i] - std_uav[:, i],
                mean_uav[:, i] + std_uav[:, i],
                alpha=0.2,
            )
            plt.xlabel("Negotiation round")
            plt.ylabel("Revenue rate")
            plt.title(f"{col} Mean±Std per Round")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{col}_mean_std.png", dpi=300)
            plt.close()

        # 2) Total revenue rate
        plt.figure()
        plt.plot(rounds, mean_tot, label="mean total", linestyle="-")
        plt.fill_between(rounds, mean_tot - std_tot, mean_tot + std_tot, alpha=0.2)
        plt.xlabel("Negotiation round")
        plt.ylabel("Total revenue rate")
        plt.title("Total Revenue Rate Mean±Std per Round")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "Total_mean_std.png", dpi=300)
        plt.close()

        # 3) Consolidated: UAV means + mean_tot / n_uavs

        # Dynamic figure width based on number of UAVs, so legend + labels fit
        base_w = 7.5
        extra_per_uav = 0.4
        fig_w = base_w + extra_per_uav * max(0, n_uavs - 3)
        fig_h = 4.5

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

        for i, col in enumerate(uav_cols):
            ax.plot(rounds, mean_uav[:, i], label=col)

        ax.plot(
            rounds,
            mean_tot / n_uavs,
            label="mean total\n/ n_uavs",
            linewidth=2,
            linestyle="--",
        )

        ax.set_xlabel("Negotiation round")
        ax.set_ylabel("Revenue rate")
        # ax.set_title("Consolidated: UAV Means + Total/n_uavs")

        # Legend outside on the right
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            frameon=True,
        )

        fig.savefig(out_dir / "Consolidated_mean.png", dpi=300)
        plt.close(fig)

def plot_consolidated_total_revenue(excel_dir: str, output_path: str | None = None):
    """
    Reads each revenue .xlsx in `excel_dir`, computes the mean total revenue
    per negotiation round across runs, and plots all algorithms together.

    Legend labels use the compact taxonomy:
      NonOverlap + ModeGG_Sequential -> NSGG
      Overlap   + ModeGR_Random     -> ORGR
      IRADA                         -> IRADA
    """
    excel_dir = Path(excel_dir)
    files = [
        f for f in excel_dir.iterdir()
        if f.suffix == ".xlsx" and not f.stem.endswith("_stats")
    ]

    consolidated: dict[str, np.ndarray] = {}
    max_rounds_overall = 0

    for fpath in files:
        algo_label = _label_from_revenue_file(fpath)
        game_label = _mode_from_path(fpath)
        if algo_label == "IRADA" or game_label == "IRADA":
            label = "IRADA"
        else:
            label = _short_algo_label(game_label, algo_label)

        sheets = pd.read_excel(fpath, sheet_name=None, index_col=0)
        dfs = list(sheets.values())
        if not dfs:
            continue

        max_rounds = max(df.shape[0] for df in dfs)
        max_rounds_overall = max(max_rounds_overall, max_rounds)
        padded = [df.reindex(range(max_rounds), method="ffill") for df in dfs]

        uav_cols = [c for c in padded[0].columns if str(c).upper().startswith("UAV")]
        if not uav_cols:
            continue

        # total per round per run
        arr = np.stack([df[uav_cols].sum(axis=1).values for df in padded], axis=0)
        mean_tot = arr.mean(axis=0)

        consolidated[label] = mean_tot

    if not consolidated:
        return

    rounds = np.arange(max_rounds_overall)

    # Dynamic figure width based on number of algorithms
    n_algos = len(consolidated)
    base_w = 7.5
    extra_per_algo = 0.6
    fig_w = base_w + extra_per_algo * max(0, n_algos - 3)
    fig_h = 4.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    for label, mean_tot in consolidated.items():
        if len(mean_tot) < len(rounds):
            mean_tot = np.pad(mean_tot, (0, len(rounds) - len(mean_tot)), "edge")
        ax.plot(rounds, mean_tot, label=label)

    ax.set_xlabel("Negotiation round")
    ax.set_ylabel("Mean total revenue rate")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=True,
    )

    if output_path is None:
        output_path = excel_dir / "combined_total_revenue_rate.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# ============================================================
# Boxplot #1: Final total revenue rate per run (split views)
# ============================================================

def boxplot_final_totals_with_irada(rev_dirs: List[str | Path],
                                    out_png: str | None):
    """
    Create two separate boxplots of final total revenue per run:

    1) NonOverlap vs IRADA (excluding Overlap):
       - Includes all revenue workbooks from NonOverlap simulations.
       - Includes all IRADA revenue workbooks.
       - Saves to: final_total_nonoverlap_vs_irada.png

    2) Overlap only:
       - Includes only revenue workbooks from Overlap simulations.
       - Saves to: final_total_overlap_only.png

    `rev_dirs` can contain a mixture of NonOverlap, Overlap and IRADA
    revenue simulation folders.
    """
    # Decide output folder
    if out_png is not None:
        out_dir = Path(out_png)
        # If a file path was passed, use its parent as the folder
        if out_dir.suffix.lower() == ".png":
            out_dir = out_dir.parent
    else:
        first = Path(rev_dirs[0])
        out_dir = first / "boxplots_final_total_revenue_rate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Buckets
    labels_non_irada, data_non_irada = [], []   # NonOverlap + IRADA
    labels_overlap, data_overlap = [], []       # Overlap only

    for d in rev_dirs:
        sim = Path(d)
        if not sim.exists():
            continue

        parts = sim.parts
        # Exact folder classification
        is_non = "NonOverlap" in parts
        is_over = "Overlap" in parts
        is_irada_sim = "IRADA" in parts or "Benchmarking" in parts

        for f in sorted(sim.glob("*.xlsx")):
            if f.stem.endswith("_stats"):
                continue

            try:
                sheets = pd.read_excel(f, sheet_name=None, index_col=0)
            except Exception:
                continue

            finals = []
            for df in sheets.values():
                uav_cols = [c for c in df.columns
                            if str(c).upper().startswith("UAV")]
                if not uav_cols:
                    continue
                finals.append(df[uav_cols].iloc[-1].sum())

            if not finals:
                continue

            algo_label = _label_from_revenue_file(f)

            # ---- Bucket 1: NonOverlap vs IRADA (no Overlap) ----
            if is_non or is_irada_sim or "IRADA" in f.stem:
                if is_irada_sim or "IRADA" in f.stem:
                    label = "IRADA"
                else:
                    label = _short_algo_label("NonOverlap", algo_label)  # e.g. NSGG
                labels_non_irada.append(label)
                data_non_irada.append(finals)

            # ---- Bucket 2: Overlap only (exclude IRADA) ----
            if is_over and "IRADA" not in f.stem:
                label = _short_algo_label("Overlap", algo_label)  # e.g. ORGR
                labels_overlap.append(label)
                data_overlap.append(finals)

    # ------------------------------------------------------------
    # Plot 1: NonOverlap vs IRADA
    # ------------------------------------------------------------
    if data_non_irada:
        # Ensure IRADA is plotted last
        pairs = list(zip(labels_non_irada, data_non_irada))
        def sort_key(pair):
            lbl, _ = pair
            return (1 if lbl == "IRADA" else 0, lbl)
        pairs.sort(key=sort_key)
        labels_non_irada, data_non_irada = zip(*pairs)

        fig, ax = plt.subplots(figsize=(1.2 * len(labels_non_irada) + 4, 6))
        bp = ax.boxplot(data_non_irada, tick_labels=labels_non_irada,
                        patch_artist=True)

        for box in bp["boxes"]:
            box.set_facecolor("C0")
            box.set_edgecolor("black")
        for median in bp["medians"]:
            median.set(color="orange", linewidth=2)

        # ax.set_title("Final Total Revenue Rate – NonOverlap vs IRADA")
        ax.set_ylabel("Final total revenue rate")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

        plt.tight_layout()
        out1 = out_dir / "final_total_nonoverlap_vs_irada.png"
        fig.savefig(out1, dpi=300)
        plt.close(fig)
        print(f"Saved NonOverlap-vs-IRADA final-total boxplot to {out1}")
    else:
        print("[INFO] No NonOverlap/IRADA data for final-total boxplot.")

    # ------------------------------------------------------------
    # Plot 2: Overlap only
    # ------------------------------------------------------------
    if data_overlap:
        fig, ax = plt.subplots(figsize=(1.2 * len(labels_overlap) + 4, 6))
        bp = ax.boxplot(data_overlap, tick_labels=labels_overlap,
                        patch_artist=True)

        for box in bp["boxes"]:
            box.set_facecolor("C0")
            box.set_edgecolor("black")
        for median in bp["medians"]:
            median.set(color="orange", linewidth=2)

        # ax.set_title("Final Total Revenue Rate – Overlap Only")
        ax.set_ylabel("Final total revenue rate")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

        plt.tight_layout()
        out2 = out_dir / "final_total_overlap_only.png"
        fig.savefig(out2, dpi=300)
        plt.close(fig)
        print(f"Saved Overlap-only final-total boxplot to {out2}")
    else:
        print("[INFO] No Overlap data for final-total boxplot.")

# ============================================================
# Boxplot #2: Flight time left per tour (cross-mode)
# ============================================================

def _algo_label_from_seq_file(seq_file: Path) -> str | None:
    """
    Infer algorithm label from sequence filename.

    Works for:
      - UAVs2_GRID5_100_3_ModeGG_Random_sequences.xlsx
      - UAVs2_GRID5_100_3_ModeGR_Sequential_sequences.xlsx
      - ... and IRADA variants.

    Returns "ModeGG_Random", "ModeGR_Sequential", or "IRADA".
    """
    stem = seq_file.stem.replace("_sequences", "")

    # IRADA case
    if "IRADA" in stem:
        return "IRADA"

    parts = stem.split("_")

    # Find the first token that starts with "Mode"
    for i, p in enumerate(parts):
        if p.startswith("Mode"):
            if i + 1 < len(parts):
                # e.g. ModeGG_Random
                return f"{p}_{parts[i + 1]}"
            else:
                # Just ModeXX
                return p

    # Could not find a Mode* token
    return None


def get_revenue_file_for_sequence(seq_file: Path, rev_dir: Path) -> Path | None:
    """
    Given a sequence workbook and its revenue directory, infer the matching
    revenue workbook filename.

    Examples:
      seq: UAVs2_GRID5_100_3_ModeGR_Random_sequences.xlsx
      →   UAVs2_GRID5_ModeGR_Random.xlsx

      seq: UAVs2_GRID5_100_3_IRADA_sequences.xlsx
      →   UAVs2_GRID5_IRADA.xlsx  (preferred)
      →   UAVs2_GRID5_100_3_IRADA.xlsx (fallback)
    """
    stem = seq_file.stem.replace("_sequences", "")
    parts = stem.split("_")

    # IRADA case
    if "IRADA" in parts:
        # Prefer short: UAVsN_GRIDM_IRADA.xlsx
        rev_stem = f"{parts[0]}_{parts[1]}_IRADA"
        cand = rev_dir / f"{rev_stem}.xlsx"
        if cand.exists():
            return cand

        # Fallback: full stem as-is (UAVsN_GRIDM_100_3_IRADA.xlsx)
        cand2 = rev_dir / f"{stem}.xlsx"
        if cand2.exists():
            return cand2

        return None

    # Non-IRADA: look for "ModeXXX"
    mode_idx = next((i for i, p in enumerate(parts) if p.startswith("Mode")), None)
    if mode_idx is not None and mode_idx + 1 < len(parts):
        # Build UAVsN_GRIDM_ModeXX_Algo
        rev_stem = f"{parts[0]}_{parts[1]}_{parts[mode_idx]}_{parts[mode_idx + 1]}"
        cand = rev_dir / f"{rev_stem}.xlsx"
        if cand.exists():
            return cand

    # Fallback: try full stem (just drop _sequences)
    cand2 = rev_dir / f"{stem}.xlsx"
    if cand2.exists():
        return cand2

    return None


def boxplot_flight_time_left(seq_roots: List[str | Path],
                             cfg: Config,
                             out_png: str,
                             nonoverlap_wp_sim: str | Path = None):
    """
    For each configuration (game + algorithm), compute the remaining flight time
    for the final tour of each UAV in each run, and plot a boxplot (seconds).

    For every mode (NonOverlap, Overlap, IRADA):
        * Use the last negotiation round in each sheet.
        * Parse the UAVk sequence string into waypoint IDs.
        * For IRADA: compute single depot-to-depot tour distance.
        * For NonOverlap/Overlap: use m_j from file for multi-loop tours.
        * Compute remaining flight time.

    For a configuration with n_uavs UAVs and n_runs sheets, there will be
    n_uavs * n_runs data points in that box.
    
    Args:
        seq_roots: List of sequence directories (NonOverlap, Overlap, IRADA)
        cfg: Configuration object
        out_png: Output path for the boxplot
        nonoverlap_wp_sim: Path to NonOverlap waypoints simulation directory
                          (required for IRADA sequences)
    """
    # Collect all *_sequences.xlsx files from the roots
    seq_files: list[Path] = []
    for root in seq_roots:
        root = Path(root)
        if root.exists():
            seq_files.extend(root.rglob("*_sequences.xlsx"))

    if not seq_files:
        print("[WARN] No sequence files found; flight-time-left boxplot not created.")
        return

    print(f"[INFO] boxplot_flight_time_left: {len(seq_files)} sequence workbooks found.")

    label_to_values: dict[str, list[float]] = defaultdict(list)

    for seq_file in seq_files:
        algo_label = _algo_label_from_seq_file(seq_file)
        if not algo_label:
            print(f"[SKIP] Could not infer algo from {seq_file.name}")
            continue

        seq_dir = seq_file.parent
        parts = seq_dir.parts

        # -------- Determine game/mode for labeling --------
        if "IRADA" in parts or "Benchmarking" in parts or "IRADA" in seq_file.stem:
            game_label = "IRADA"
        elif "NonOverlap" in parts:
            game_label = "NonOverlap"
        elif "Overlap" in parts:
            game_label = "Overlap"
        else:
            game_label = "Other"

        if game_label == "IRADA":
            label = "IRADA"
        else:
            label = _short_algo_label(game_label, algo_label)

        # --- Derive waypoints directory from sequences dir ---
        if "IRADA" in str(seq_file):
            # IRADA sequences; use provided NonOverlap waypoints path
            if nonoverlap_wp_sim is None:
                print(f"[WARN] IRADA sequences require nonoverlap_wp_sim parameter; skipping {seq_file.name}")
                continue
            wp_dir = Path(nonoverlap_wp_sim)
        else:
            # NonOverlap / Overlap: same tree, different leaf
            wp_dir = Path(str(seq_dir).replace(
                os.sep + "sequences" + os.sep,
                os.sep + "waypoints" + os.sep
            ))

        # --- Waypoints workbook: derive UAVsN and GRIDM from sequence filename ---
        # e.g. UAVs3_GRID13_1920_16_ModeGG_Random_sequences.xlsx
        stem = seq_file.stem.replace("_sequences", "")
        name_parts = stem.split("_")
        if (len(name_parts) < 2 or
                not name_parts[0].startswith("UAVs") or
                not name_parts[1].startswith("GRID")):
            print(f"[WARN] Could not infer UAVs/GRID from {seq_file.name}")
            continue

        uavs_token = name_parts[0]   # e.g. "UAVs3"
        grid_token = name_parts[1]   # e.g. "GRID13"
        wp_file = wp_dir / f"{uavs_token}_{grid_token}_waypoints.xlsx"
        if not wp_file.exists():
            print(f"[WARN] Missing waypoints file for {seq_file.name}: {wp_file}")
            continue

        # Load sequences (no revenue needed here)
        try:
            seq_sheets = pd.read_excel(seq_file, sheet_name=None, index_col=0)
        except Exception as e:
            print(f"[WARN] Could not read {seq_file}: {e}")
            continue

        try:
            # waypoints have one sheet per run
            wp_book = pd.read_excel(wp_file, sheet_name=None)
        except Exception as e:
            print(f"[WARN] Could not read waypoint workbook {wp_file}: {e}")
            continue

        per_file_values: list[float] = []

        for run_name, seq_df in seq_sheets.items():
            # Waypoint sheet may be keyed either by name or index
            if run_name in wp_book:
                df_wp = wp_book[run_name]
            else:
                # Fallback: first sheet if naming is inconsistent
                df_wp = next(iter(wp_book.values()))

            coords = {
                int(r.Waypoint): (float(r.X), float(r.Y))
                for _, r in df_wp.iterrows()
            }

            uav_cols = [c for c in seq_df.columns if str(c).upper().startswith("UAV")]

            # Use only the last negotiation round
            last_row = seq_df.iloc[-1]

            # Check if this is IRADA (no m_j columns) or regular (has m_j columns)
            m_cols = [c for c in seq_df.columns if str(c).lower().startswith("m_")]
            has_mj = len(m_cols) == len(uav_cols)

            if not has_mj and game_label != "IRADA":
                print(f"[WARN] {seq_file.name}: m_j columns missing for non-IRADA file")
                continue

            for uidx, ucol in enumerate(uav_cols):
                seq_str = str(last_row[ucol])
                if not seq_str or seq_str.lower() == "nan":
                    continue

                ids = [int(x) for x in seq_str.split("-") if x]
                if not ids:
                    continue

                # ✅ SKIP if only one waypoint (hovering scenario)
                if len(ids) == 1:
                    per_file_values.append(0.0)  # No flight time left, hovering until depleted
                    continue

                # Compute tour geometry
                depot = (0.0, 0.0)
                pts = [coords[i] for i in ids if i in coords]
                if not pts:
                    continue

                def euclidean(a, b):
                    return math.hypot(a[0] - b[0], a[1] - b[1])

                # ✅ IRADA: Full tour D→waypoints→D (already complete in cell)
                if game_label == "IRADA":
                    # IRADA stores complete tours: just sum all edges
                    total_dist = 0.0
                    # Depot to first waypoint
                    total_dist += euclidean(depot, pts[0])
                    # Waypoint to waypoint
                    for i in range(len(pts) - 1):
                        total_dist += euclidean(pts[i], pts[i+1])
                    # Last waypoint back to depot
                    total_dist += euclidean(pts[-1], depot)
                    
                    total_time = total_dist / float(cfg.speed)
                    slack = cfg.max_flight_time - total_time
                    left_time = max(slack, 0.0)
                else:
                    # ✅ NonOverlap/Overlap: Use m_j from file for multi-loop tours
                    m_j = int(last_row[f"m_{uidx}"])
                    
                    first = euclidean(depot, pts[0])
                    fwd = sum(euclidean(pts[i], pts[i+1]) for i in range(len(pts)-1))
                    ret = euclidean(pts[-1], depot)
                    jump = euclidean(pts[-1], pts[0]) if len(pts) > 1 else 0.0
                    
                    # Compute total tour distance with m_j loops
                    total_dist = first + m_j * fwd + (m_j - 1) * jump + ret
                    total_time = total_dist / float(cfg.speed)
                    slack = cfg.max_flight_time - total_time
                    left_time = max(slack, 0.0)

                per_file_values.append(left_time)

        if per_file_values:
            label_to_values[label].extend(per_file_values)

    if not label_to_values:
        print("[WARN] Nothing computed; flight-time-left boxplot will not be created.")
        return

    # -------- Sorting: NonOverlap*, Overlap*, IRADA last --------
    def sort_key(lbl: str):
        if lbl == "IRADA":
            return (2, "", "")
        if lbl.startswith("N"):  # NonOverlap short labels
            return (0, lbl, "NonOverlap")
        if lbl.startswith("O"):  # Overlap short labels
            return (1, lbl, "Overlap")
        return (0, lbl, "Other")

    labels = sorted(label_to_values.keys(), key=sort_key)
    data = [label_to_values[l] for l in labels]

    # Dynamic figure width based on label count
    fig_w = 4.0 + 0.25 * len(labels)
    fig_h = 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    for box in bp["boxes"]:
        box.set_facecolor("C0")
        box.set_edgecolor("black")
    for median in bp["medians"]:
        median.set(color="orange", linewidth=2)

    ax.set_ylabel("Flight time left (s)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved flight-time-left boxplot to {out_path}")


#Use this function for equal font size (labels are too big!)
# def boxplot_flight_time_left(seq_roots: List[str | Path],
#                              cfg: Config,
#                              out_png: str):
#     """
#     Build a boxplot of (max_flight_time - tour_time) per tour per UAV (as %),
#     grouped by *game + algorithm* across all provided sequence roots.

#     Labels:
#       - NonOverlap-ModeGG_Random, ..., NonOverlap-ModeRR_Sequential
#       - Overlap-ModeGG_Random, ..., Overlap-ModeRR_Sequential
#       - IRADA (all IRADA runs)
#     """
#     # Collect all *_sequences.xlsx files
#     seq_files: list[Path] = []
#     for root in seq_roots:
#         root = Path(root)
#         if root.exists():
#             seq_files.extend(root.rglob("*_sequences.xlsx"))

#     if not seq_files:
#         print("[WARN] No sequence files found; flight-time-left boxplot not created.")
#         return

#     label_to_values: dict[str, list[float]] = defaultdict(list)

#     for seq_file in seq_files:
#         algo_label = _algo_label_from_seq_file(seq_file)
#         if not algo_label:
#             print(f"[SKIP] Could not infer algo from {seq_file.name}")
#             continue

#         seq_dir = seq_file.parent
#         parts = seq_dir.parts

#         # -------- Determine game/mode for labeling --------
#         if "IRADA" in parts or "Benchmarking" in parts or "IRADA" in seq_file.stem:
#             game_label = "IRADA"
#         elif "NonOverlap" in parts:
#             game_label = "NonOverlap"
#         elif "Overlap" in parts:
#             game_label = "Overlap"
#         else:
#             game_label = "Other"

#         if game_label == "IRADA":
#             label = "IRADA"
#         elif game_label in ("NonOverlap", "Overlap"):
#             label = f"{game_label}-{algo_label}"
#         else:
#             label = algo_label  # fallback

#         # --- Derive revenue and waypoints dirs from sequences dir ---
#         if "IRADA" in str(seq_file):
#             # IRADA sequences/revenue under Benchmarking; waypoints from NonOverlap
#             seq_dir_str = str(seq_dir)
#             rev_dir_str = seq_dir_str.replace(
#                 f"Benchmarking{os.sep}IRADA{os.sep}sequences",
#                 f"Benchmarking{os.sep}IRADA{os.sep}revenue"
#             )
#             wp_dir_str = seq_dir_str.replace(
#                 f"Benchmarking{os.sep}IRADA{os.sep}sequences",
#                 f"Results{os.sep}NonOverlap{os.sep}waypoints"
#             )
#             rev_dir = Path(rev_dir_str)
#             wp_dir = Path(wp_dir_str)
#         else:
#             # NonOverlap / Overlap
#             rev_dir = Path(str(seq_dir).replace(
#                 os.sep + "sequences" + os.sep,
#                 os.sep + "revenue" + os.sep
#             ))
#             wp_dir = Path(str(seq_dir).replace(
#                 os.sep + "sequences" + os.sep,
#                 os.sep + "waypoints" + os.sep
#             ))

#         # --- Waypoints workbook: derive UAVsN and GRIDM from sequence filename ---
#         # e.g. UAVs3_GRID10_100_3_ModeGG_Random_sequences.xlsx
#         stem = seq_file.stem.replace("_sequences", "")
#         name_parts = stem.split("_")
#         if (len(name_parts) < 2 or
#                 not name_parts[0].startswith("UAVs") or
#                 not name_parts[1].startswith("GRID")):
#             print(f"[WARN] Could not infer UAVs/GRID from {seq_file.name}")
#             continue

#         uavs_token = name_parts[0]   # e.g. "UAVs3"
#         grid_token = name_parts[1]   # e.g. "GRID10"
#         wp_file = wp_dir / f"{uavs_token}_{grid_token}_waypoints.xlsx"
#         if not wp_file.exists():
#             print(f"[WARN] Missing waypoints file for {seq_file.name}: {wp_file}")
#             continue

#         # Revenue workbook inferred from sequence name
#         rev_file = get_revenue_file_for_sequence(seq_file, rev_dir)
#         if rev_file is None or not rev_file.exists():
#             print(f"[WARN] Missing revenue file for {seq_file.name} in {rev_dir}")
#             continue

#         # Load data
#         try:
#             seq_sheets = pd.read_excel(seq_file, sheet_name=None, index_col=0)
#             rev_sheets = pd.read_excel(rev_file, sheet_name=None, index_col=0)
#         except Exception as e:
#             print(f"[WARN] Could not read {seq_file} or {rev_file}: {e}")
#             continue

#         per_file_values: list[float] = []

#         for run_name, seq_df in seq_sheets.items():
#             rev_df = rev_sheets.get(run_name)
#             if rev_df is None:
#                 print(f"[WARN] {rev_file.name}: missing revenue sheet {run_name}")
#                 continue

#             try:
#                 df_wp = pd.read_excel(wp_file, sheet_name=run_name)
#             except Exception as e:
#                 print(f"[WARN] {wp_file.name}: missing waypoints sheet {run_name}: {e}")
#                 continue

#             coords = {int(r.Waypoint): (float(r.X), float(r.Y))
#                       for _, r in df_wp.iterrows()}
#             depot = (0.0, 0.0)

#             uav_cols = [c for c in seq_df.columns if str(c).upper().startswith("UAV")]
#             m_cols = [c for c in seq_df.columns if str(c).lower().startswith("m_")]
#             has_m = len(m_cols) == len(uav_cols)

#             for round_idx in range(len(seq_df)):
#                 for uidx, ucol in enumerate(uav_cols):
#                     seq_str = str(seq_df.iloc[round_idx][ucol])
#                     if not seq_str or seq_str.lower() == "nan":
#                         continue

#                     ids = [int(x) for x in seq_str.split("-") if x]
#                     if not ids:
#                         continue

#                     if has_m:
#                         m_val = int(seq_df.iloc[round_idx][f"m_{uidx}"])
#                         ids = ids * max(1, m_val)

#                     pts = [depot] + [coords[i] for i in ids if i in coords] + [depot]

#                     # Euclidean distance
#                     dist = sum(
#                         ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
#                         for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:])
#                     )

#                     used_time = dist / float(cfg.speed)
#                     left_time = float(cfg.max_flight_time) - used_time
#                     left_pct = max(0.0, (left_time / float(cfg.max_flight_time)) * 100.0)
#                     per_file_values.append(left_pct)

#         if per_file_values:
#             label_to_values[label].extend(per_file_values)

#     if not label_to_values:
#         print("[WARN] Nothing computed; flight-time-left boxplot will not be created.")
#         return

#     # -------- Sorting: NonOverlap*, Overlap*, IRADA last --------
#     def sort_key(lbl: str):
#         if lbl == "IRADA":
#             return (2, "", "")
#         if lbl.startswith("NonOverlap-"):
#             algo = lbl.split("NonOverlap-")[1]
#             return (0, algo, "NonOverlap")
#         if lbl.startswith("Overlap-"):
#             algo = lbl.split("Overlap-")[1]
#             return (1, algo, "Overlap")
#         return (0, lbl, "Other")

#     labels = sorted(label_to_values.keys(), key=sort_key)
#     data = [label_to_values[l] for l in labels]

#     # --- Dynamic figure size based on label count and tick font size ---
#     xtick_fs = plt.rcParams.get("xtick.labelsize", 10)
#     font_scale = xtick_fs / 10.0  # 1.0 at 10pt, 2.4 at 24pt, etc.

#     base_w = 4.0
#     per_label_w = 0.25 * font_scale  # more labels or bigger font => wider figure
#     fig_w = base_w + per_label_w * len(labels)
#     fig_h = 4.0 * font_scale**0.5   # mild growth in height with font size

#     fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
#     bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

#     for box in bp["boxes"]:
#         box.set_facecolor("C0")
#         box.set_edgecolor("black")
#     for median in bp["medians"]:
#         median.set(color="orange", linewidth=2)

#     ax.set_ylabel("Flight time left (%)")
#     ax.grid(True, axis="y", linestyle="--", alpha=0.5)
#     plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

#     out_path = Path(out_png)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)
#     print(f"Saved flight-time-left boxplot to {out_path}")

# ============================================================
# Boxplot #3: Per-UAV contribution (per configuration)
# ============================================================
def boxplot_uav_contribution_all(rev_dirs: List[str | Path],
                                 out_png: str | None):
    """
    For each revenue workbook (configuration) across all rev_dirs
    (expected to belong to the same game: NonOverlap OR Overlap OR IRADA):

      - For each sheet (run), find the round with maximum total revenue rate.
      - Compute per-UAV share = UAV_i / sum_j UAV_j at that round.
      - Aggregate shares across runs for that configuration.
      - Produce one boxplot per configuration with one box per UAV.

    Output:
      A PNG per configuration saved under the given output directory.
    """
    # Decide output directory from out_png
    if out_png:
        out_dir = Path(out_png)
        if out_dir.suffix.lower() == ".png":
            out_dir = out_dir.parent
    else:
        out_dir = Path(rev_dirs[0]) / "boxplots_uav_contribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    # config_label -> { "uav_cols": [...], "shares": {uav_col: [vals]} }
    by_config: dict[str, dict] = {}

    for rd in rev_dirs:
        rd = Path(rd)
        if not rd.exists():
            continue

        for f in sorted(rd.rglob("UAVs*.xlsx")):
            if f.stem.endswith("_stats"):
                continue

            try:
                sheets = pd.read_excel(f, sheet_name=None, index_col=0)
            except Exception as e:
                print(f"[WARN] Could not read {f}: {e}")
                continue

            # Configuration label from filename
            # e.g. NSGG, ORGR, IRADA (via _label_from_revenue_file)
            algo_label = _label_from_revenue_file(f)
            config_label = algo_label

            if config_label not in by_config:
                by_config[config_label] = {
                    "uav_cols": None,
                    "shares": defaultdict(list),
                }

            cfg_entry = by_config[config_label]

            for df in sheets.values():
                uav_cols = [c for c in df.columns if str(c).upper().startswith("UAV")]
                if not uav_cols:
                    continue

                # Lock UAV column order once
                if cfg_entry["uav_cols"] is None:
                    cfg_entry["uav_cols"] = uav_cols

                totals = df[uav_cols].sum(axis=1)
                if (totals <= 0).all():
                    continue

                # Round with maximum total revenue rate
                idx = totals.idxmax()
                row = df.loc[idx, uav_cols]
                tot = row.sum()
                if tot <= 0:
                    continue

                shares = (row / tot).values  # fractions in [0,1]
                for col, val in zip(uav_cols, shares):
                    cfg_entry["shares"][col].append(float(val))

    if not by_config:
        print("[WARN] No data for UAV contribution boxplots.")
        return

    # One figure per configuration
    for config_label, payload in sorted(by_config.items()):
        uav_cols = payload["uav_cols"]
        if not uav_cols:
            print(f"[WARN] Config {config_label} has no UAV columns; skipping.")
            continue

        data = [payload["shares"][c] for c in uav_cols]
        if not any(len(lst) for lst in data):
            print(f"[WARN] No data for config {config_label}; skipping.")
            continue

        fig, ax = plt.subplots(figsize=(1.0 * len(uav_cols) + 3, 6))
        bp = ax.boxplot(data, tick_labels=uav_cols, patch_artist=True)

        for box in bp["boxes"]:
            box.set_facecolor("C0")
            box.set_edgecolor("black")
        for median in bp["medians"]:
            median.set(color="orange", linewidth=2)

        ax.set_ylabel("Share of total revenue rate")

        # ---- Dynamic Y-scale based on maximum share ----
        all_vals = np.concatenate(
            [np.asarray(d, float) for d in data if len(d)]
        )
        if all_vals.size > 0:
            max_share = float(all_vals.max())     # e.g. 0.132
            if max_share <= 0:
                top_frac = 0.1
            else:
                # nearest 10% ceiling, capped at 100%
                top_frac = min(1.0, math.ceil(max_share * 10) / 10.0)
        else:
            top_frac = 1.0

        ax.set_ylim(0, top_frac)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

        # Tick every 5% (or 10% if very tall)
        step = 0.05 if top_frac <= 0.5 else 0.1
        ax.set_yticks(np.arange(0.0, top_frac + 1e-9, step))

        plt.xticks(rotation=0, ha="center")
        plt.tight_layout()

        # Safe filename for config_label
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", config_label)
        out_file = out_dir / f"uav_contribution_{safe_label}.png"
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f"Saved UAV contribution boxplot for {config_label} → {out_file}")

# ============================================================
# GIF generation (per mode)
# ============================================================
def generate_gifs_from_sequences(sim_sequences_dir: str,
                                 cfg: Config):
    """
    Generate GIFs for each *_sequences.xlsx in `sim_sequences_dir`, using matching
    revenue and waypoints files derived from the folder structure, and save them to:

      Visualizations/<Mode>/gifs/YYYY-MM-DD/simulation_k/<ConfigLabel>/*.gif
    """
    seq_dir = Path(sim_sequences_dir)

    # Determine mode from path
    mode = _mode_from_path(seq_dir)
    here = Path(__file__).parent
    vis_root = here / cfg.visualization_dir

    # Where to store GIFs for this simulation (per configuration subfolders)
    gifs_sim_root = vis_gifs_root_for_sim(vis_root, mode, seq_dir)

    # Default NonOverlap / Overlap mapping for data
    rev_dir_default = Path(str(seq_dir).replace(
        os.sep + "sequences" + os.sep,
        os.sep + "revenue" + os.sep
    ))
    wp_dir_default = Path(str(seq_dir).replace(
        os.sep + "sequences" + os.sep,
        os.sep + "waypoints" + os.sep
    ))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    zero_color = "#CCCCCC"
    nonzero_color = "#111111"

    for seq_file in seq_dir.glob("*_sequences.xlsx"):
        base_stem = seq_file.stem.replace("_sequences", "")
        parts = base_stem.split("_")

        # Determine config label (ModeXX_Algo or IRADA)
        algo_label = _algo_label_from_seq_file(seq_file) or "Unknown"

        # Derive rev_dir / wp_dir per file (IRADA vs others)
        if "IRADA" in parts:
            seq_dir_str = str(seq_dir)
            rev_dir_str = seq_dir_str.replace(
                f"Benchmarking{os.sep}IRADA{os.sep}sequences",
                f"Benchmarking{os.sep}IRADA{os.sep}revenue"
            )
            wp_dir_str = seq_dir_str.replace(
                f"Benchmarking{os.sep}IRADA{os.sep}sequences",
                f"Results{os.sep}NonOverlap{os.sep}waypoints"
            )
            rev_dir = Path(rev_dir_str)
            wp_dir = Path(wp_dir_str)
        else:
            rev_dir = rev_dir_default
            wp_dir = wp_dir_default

        rev_f = get_revenue_file_for_sequence(seq_file, rev_dir)
        wp_f = wp_dir / f"{parts[0]}_{parts[1]}_waypoints.xlsx"

        if rev_f is None or not rev_f.exists() or not wp_f.exists():
            print(f"Skipping {base_stem}: missing revenue or waypoints.")
            continue

        rev_sheets = pd.read_excel(rev_f, sheet_name=None, index_col=0)
        seq_sheets = pd.read_excel(seq_file, sheet_name=None, index_col=0)

        # Parse from filename (but will recompute from columns later)
        n_uavs = int(parts[0][4:])
        grid_dim = int(parts[1][4:])

        # Per-configuration GIF folder
        cfg_gifs_dir = gifs_sim_root / algo_label
        cfg_gifs_dir.mkdir(parents=True, exist_ok=True)

        for run_name, seq_df in seq_sheets.items():
            rev_df = rev_sheets.get(run_name)
            if rev_df is None:
                continue

            # -------- align revenue length to sequences (forward-fill) --------
            n_seq = len(seq_df)
            n_rev = len(rev_df)
            if n_rev < n_seq:
                last_row = rev_df.iloc[-1]
                extra = pd.DataFrame([last_row] * (n_seq - n_rev),
                                     columns=rev_df.columns)
                rev_df = pd.concat([rev_df, extra], ignore_index=True)
                print(f"[INFO] Extended revenue {base_stem}:{run_name} "
                      f"from {n_rev} to {n_seq} rows by repeating last row.")
            elif n_rev > n_seq:
                # Very unlikely, but keep them consistent
                rev_df = rev_df.iloc[:n_seq].copy()
                print(f"[WARN] Truncated revenue {base_stem}:{run_name} "
                      f"from {n_rev} to {n_seq} rows to match sequences.")

            # reset indices so iloc/round labels stay simple
            seq_df = seq_df.reset_index(drop=True)
            rev_df = rev_df.reset_index(drop=True)

            df_wp = pd.read_excel(wp_f, sheet_name=run_name)
            coords = {int(r.Waypoint): (r.X, r.Y, r.Revenue)
                      for _, r in df_wp.iterrows()}

            xs_sorted = [coords[i][0] for i in sorted(coords)]
            d = abs(xs_sorted[1] - xs_sorted[0]) if len(xs_sorted) >= 2 else 1.0

            xs_all = [c[0] for c in coords.values()]
            ys_all = [c[1] for c in coords.values()]

            x_span = max(xs_all) - min(xs_all)
            y_span = max(ys_all) - min(ys_all)
            ips = 0.5
            sidebar = 2.5
            fig_w = x_span * ips + sidebar
            fig_h = y_span * ips + 1.0

            fig = plt.figure(figsize=(fig_w, fig_h))

            header = (
                f"UAVs = {n_uavs}  Grid = {grid_dim}m × {grid_dim}m\n"
                f"Simulation Run = {run_name.replace('SimRun','')}"
            )
            fig.text(0.5, 0.98, header, ha="center", va="top",
                     fontsize=16, weight="bold")

            ax = fig.add_axes([0.05, 0.05, 0.70, 0.88])
            ax.set_xlabel(f"X (m), spacing={d:.1f}")
            ax.set_ylabel(f"Y (m), spacing={d:.1f}")
            ax.set_xlim(min(xs_all) - d, max(xs_all) + d)
            ax.set_ylim(min(ys_all) - d, max(ys_all) + d)

            xticks = list(range(int(min(xs_all)),
                                int(max(xs_all)) + 1, int(d)))
            yticks = list(range(int(min(ys_all)),
                                int(max(ys_all)) + 1, int(d)))
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            wp_xs, wp_ys, wp_cs = zip(*[
                (x, y, nonzero_color if rev > 0 else zero_color)
                for x, y, rev in coords.values()
            ])
            ax.scatter(wp_xs, wp_ys, c=wp_cs, s=80, zorder=1)
            ax.scatter([0], [0], marker="*", color="k", s=200, zorder=2)

            rounds = list(seq_df.index)
            n_rounds = len(rounds)
            uav_cols = [c for c in seq_df.columns
                        if str(c).upper().startswith("UAV")]
            n_uavs = len(uav_cols)

            dynamic_texts: List[plt.Text] = []

            def update(frame):
                # clear old lines, extra scatters, and texts
                for line in list(ax.lines):
                    line.remove()
                for coll in ax.collections[2:]:
                    coll.remove()
                for txt in dynamic_texts:
                    txt.remove()
                dynamic_texts.clear()

                # redraw waypoints and depot
                ax.scatter(wp_xs, wp_ys, c=wp_cs, s=80, zorder=1)
                ax.scatter([0], [0], marker="*", color="k", s=200, zorder=2)

                sidebar_x = 0.78
                sidebar_y = 0.88
                dy = 0.06

                for u in range(n_uavs):
                    seq_str = seq_df.iloc[frame][f"UAV{u}"]
                    idxs = [int(x) for x in str(seq_str).split("-") if x]
                    pts = [(0, 0)] + [
                        (coords[i][0], coords[i][1]) for i in idxs
                    ] + [(0, 0)]
                    px, py = zip(*pts)
                    ax.plot(px, py, color=colors[u], lw=2)

                    m_val = seq_df.iloc[frame].get(f"m_{u}", 1)
                    z_val = rev_df.iloc[frame][f"UAV{u}"]
                    txt = rf"$m_{{{u}}}$={m_val}, $Z_{{{u}}}$={z_val:.2f}"
                    t = fig.text(
                        sidebar_x, sidebar_y - u * dy,
                        txt,
                        va="top", ha="left",
                        color=colors[u],
                        fontsize=14,
                    )
                    dynamic_texts.append(t)

                    if m_val > 1 and len(idxs) >= 2:
                        x0, y0 = coords[idxs[0]][:2]
                        x1, y1 = coords[idxs[-1]][:2]
                        ax.plot([x0, x1], [y0, y1],
                                linestyle="--", color=colors[u], lw=1)

                Z_tot = rev_df[[c for c in rev_df.columns
                                if str(c).upper().startswith("UAV")]]\
                    .iloc[frame].sum()
                t_tot = fig.text(
                    sidebar_x, sidebar_y - n_uavs * dy,
                    rf"$Z$={Z_tot:.2f}",
                    va="top", ha="left",
                    color="black",
                    fontsize=14,
                )
                dynamic_texts.append(t_tot)

                t_round = fig.text(
                    0.5, 0.90,
                    f"Negotiation Round = {rounds[frame]}",
                    ha="center", va="top",
                    fontsize=14,
                )
                dynamic_texts.append(t_round)

            ani = animation.FuncAnimation(
                fig, update, frames=n_rounds, interval=800, blit=False
            )
            gif_path = cfg_gifs_dir / f"{base_stem}_{run_name}.gif"
            ani.save(str(gif_path), writer=PillowWriter(fps=1))
            plt.close(fig)
            print(f"Saved GIF: {gif_path}")

# ============================================================
# CLI + main orchestration
# ============================================================

def parse_cli_overrides():
    parser = argparse.ArgumentParser(description="Analysis overrides")
    parser.add_argument("--num_uavs", type=int)
    parser.add_argument("--grid_width", type=int)
    parser.add_argument("--grid_height", type=int)
    parser.add_argument("--grid_spacing", type=int)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--max_flight_time", type=float)
    parser.add_argument("--n_runs", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    HERE = Path(__file__).parent

    set_plot_style()

    # --------------------------------------------------------
    # Load config + CLI overrides
    # --------------------------------------------------------
    args = parse_cli_overrides()
    config = Config.from_yaml("settings.yaml")
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.override(overrides)
    print("[INFO] Final config:", config)

    # --------------------------------------------------------
    # Project roots
    # --------------------------------------------------------
    RESULTS = HERE / config.results_dir
    VIS_ROOT = HERE / config.visualization_dir
    IRADA_ROOT = HERE / config.irada_benchmarking_dir

    NON_ROOT = RESULTS / "NonOverlap"
    OVER_ROOT = RESULTS / "Overlap"

    IRADA_SEQ_ROOT = IRADA_ROOT / "sequences"
    IRADA_REV_ROOT = IRADA_ROOT / "revenue"

    # --------------------------------------------------------
    # Helper: pick simulation folder (manual or latest)
    # --------------------------------------------------------
    def pick_sim(root_seq: Path,
                 root_rev: Path,
                 manual_date: str | None,
                 manual_sim: str | None) -> tuple[Path, Path]:
        """
        Use manual date/sim if given, else the latest simulation under root_seq.
        Expected structure: root_seq/YYYY-MM-DD/simulation_k
        """
        if manual_date and manual_sim:
            seq_sim = root_seq / manual_date / manual_sim
            rev_sim = root_rev / manual_date / manual_sim
            print(f"[INFO] Using MANUAL sim: {seq_sim}")
            return seq_sim, rev_sim
        else:
            seq_sim = find_latest_simulation(root_seq)
            rev_sim = Path(
                str(seq_sim).replace(
                    f"{os.sep}sequences{os.sep}",
                    f"{os.sep}revenue{os.sep}",
                )
            )
            print(f"[INFO] Using LATEST sim: {seq_sim}")
            return seq_sim, rev_sim

    # --------------------------------------------------------
    # Find simulations per mode (using overrides if set)
    # --------------------------------------------------------
    # NonOverlap
    non_seq_sim, non_rev_sim = pick_sim(
        NON_ROOT / "sequences",
        NON_ROOT / "revenue",
        NON_DATE,
        NON_SIM,
    )
    print(f"[INFO] NonOverlap revenue sim: {non_rev_sim}")
    print(f"[INFO] NonOverlap sequences sim: {non_seq_sim}")

    # Overlap
    over_seq_sim, over_rev_sim = pick_sim(
        OVER_ROOT / "sequences",
        OVER_ROOT / "revenue",
        OVER_DATE,
        OVER_SIM,
    )
    print(f"[INFO] Overlap revenue sim:   {over_rev_sim}")
    print(f"[INFO] Overlap sequences sim: {over_seq_sim}")

    # IRADA (optional)
    irada_seq_sim: Path | None = None
    irada_rev_sim: Path | None = None
    if IRADA_SEQ_ROOT.exists() and IRADA_REV_ROOT.exists():
        irada_seq_sim, irada_rev_sim = pick_sim(
            IRADA_SEQ_ROOT,
            IRADA_REV_ROOT,
            IRADA_DATE,
            IRADA_SIM,
        )
        print(f"[INFO] IRADA sequences sim: {irada_seq_sim}")
        print(f"[INFO] IRADA revenue sim:   {irada_rev_sim}")
    else:
        print(f"[WARN] IRADA roots not found: {IRADA_SEQ_ROOT} / {IRADA_REV_ROOT}")

    # --------------------------------------------------------
    # Visualization base dirs
    # --------------------------------------------------------
    non_mode = "NonOverlap"
    over_mode = "Overlap"
    irada_mode = "IRADA"

    non_vis_plots = vis_plots_root_for_sim(VIS_ROOT, non_mode, non_rev_sim)
    over_vis_plots = vis_plots_root_for_sim(VIS_ROOT, over_mode, over_rev_sim)
    irada_vis_plots = (
        vis_plots_root_for_sim(VIS_ROOT, irada_mode, irada_rev_sim)
        if irada_rev_sim and irada_rev_sim.exists()
        else None
    )

    comp_vis_root = vis_comparisons_root(VIS_ROOT, non_rev_sim)

    # --------------------------------------------------------
    # Graph generation per mode (written under Visualizations)
    # --------------------------------------------------------
    if GraphGeneration:
        # NonOverlap
        print(f"[RUN] Generating NonOverlap graphs → {non_rev_sim}")
        analyze_revenue_excels_graphs(str(non_rev_sim), out_root=non_vis_plots)
        plot_consolidated_total_revenue(
            str(non_rev_sim),
            output_path=non_vis_plots / "combined_total_revenue_rate.png",
        )

        # Overlap
        print(f"[RUN] Generating Overlap graphs → {over_rev_sim}")
        analyze_revenue_excels_graphs(str(over_rev_sim), out_root=over_vis_plots)
        plot_consolidated_total_revenue(
            str(over_rev_sim),
            output_path=over_vis_plots / "combined_total_revenue_rate.png",
        )

        # IRADA
        if irada_rev_sim and irada_rev_sim.exists() and irada_vis_plots:
            print(f"[RUN] Generating IRADA graphs → {irada_rev_sim}")
            analyze_revenue_excels_graphs(str(irada_rev_sim), out_root=irada_vis_plots)
            plot_consolidated_total_revenue(
                str(irada_rev_sim),
                output_path=irada_vis_plots / "combined_total_revenue_rate.png",
            )
    else:
        print("[INFO] GraphGeneration disabled")

    # --------------------------------------------------------
    # Boxplot #1 – Final total revenue rate (comparisons)
    # --------------------------------------------------------
    rev_dirs: list[Path] = [non_rev_sim, over_rev_sim]
    if irada_rev_sim and irada_rev_sim.exists():
        rev_dirs.append(irada_rev_sim)

    comp_vis_root.mkdir(parents=True, exist_ok=True)
    boxplot_final_totals_with_irada(rev_dirs, out_png=str(comp_vis_root))

    # --------------------------------------------------------
    # Boxplot #2 – Flight time left (comparisons)
    # --------------------------------------------------------
    seq_roots: list[Path] = [non_seq_sim, over_seq_sim]
    if irada_seq_sim and irada_seq_sim.exists():
        seq_roots.append(irada_seq_sim)

    # Derive NonOverlap waypoints path for IRADA
    non_wp_sim = Path(str(non_seq_sim).replace(
        os.sep + "sequences" + os.sep,
        os.sep + "waypoints" + os.sep
    ))

    flight_png = comp_vis_root / "flight_time_left_all_algorithms.png"
    boxplot_flight_time_left(
        seq_roots, 
        config, 
        out_png=str(flight_png),
        nonoverlap_wp_sim=str(non_wp_sim)  # ← Pass NonOverlap waypoints path
    )

    # --------------------------------------------------------
    # Boxplot #3 – Per-UAV contribution (per configuration, per game)
    # --------------------------------------------------------
    # NonOverlap
    non_uav_vis_root = non_vis_plots / "boxplots_uav_contribution"
    boxplot_uav_contribution_all([non_rev_sim], out_png=str(non_uav_vis_root))

    # Overlap
    over_uav_vis_root = over_vis_plots / "boxplots_uav_contribution"
    boxplot_uav_contribution_all([over_rev_sim], out_png=str(over_uav_vis_root))

    # IRADA (if present)
    if irada_rev_sim and irada_rev_sim.exists() and irada_vis_plots:
        irada_uav_vis_root = irada_vis_plots / "boxplots_uav_contribution"
        boxplot_uav_contribution_all([irada_rev_sim], out_png=str(irada_uav_vis_root))

    # --------------------------------------------------------
    # GIF generation (per mode, optional)
    # --------------------------------------------------------
    if GifGeneration:
        # NonOverlap
        print(f"[RUN] Generating NonOverlap GIFs → {non_seq_sim}")
        generate_gifs_from_sequences(str(non_seq_sim), config)

        # Overlap
        print(f"[RUN] Generating Overlap GIFs → {over_seq_sim}")
        generate_gifs_from_sequences(str(over_seq_sim), config)

        # IRADA
        if irada_seq_sim and irada_seq_sim.exists():
            print(f"[RUN] Generating IRADA GIFs → {irada_seq_sim}")
            generate_gifs_from_sequences(str(irada_seq_sim), config)
    else:
        print("[INFO] GifGeneration disabled")
