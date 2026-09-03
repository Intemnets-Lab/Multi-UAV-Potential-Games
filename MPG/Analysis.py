"""
Analysis.py - Thesis analysis pipeline for UAV waypoint-negotiation results.

Produces all 7 required deliverables across UAV counts 3-10:
  1/2. Mean total revenue rate vs round, per strategy (random/k-means initial)
  3/4. Best-of-8-strategies vs baseline boxplot (Greedy / Cluster+GA)
  5.   Per-UAV revenue-share boxplots
  6/7. Flight-time-left boxplot, best-of-8 vs baseline

Organized into two classes:
  DataLayer - loading, trust model, and math (no plotting)
  Plots     - rendering, styled to match the images approved for the thesis

...plus module-level CONFIGURATION and orchestration (discover_* / run()) at
the bottom, since those are one-shot script logic rather than reusable utilities.

Trust model:
  - Revenue rate is ALWAYS read directly from the revenue Excel files, never
    recomputed here - computing it in analysis code was tried once during
    the thesis and turned out to be the source of an error, not a fix.
  - mj is ALWAYS trusted as given in a sequences file, never re-derived.
  - Flight-time-left is the one thing this pipeline computes, since it isn't
    a column in any file - derived from trusted mj + tour geometry.
"""
import math
import os
import re
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mticker


# ============================================================
# CONFIGURATION - edit these directly, no separate settings file
# ============================================================
HERE = Path(__file__).parent

SPEED = 16
MAX_FLIGHT_TIME = 1920
GRID_WIDTH = 13
N_RUNS = 100
UAV_COUNTS = list(range(3, 11))  # 3..10

VISUALIZATIONS_ROOT = HERE / "Visualizations"

OLD_SEQUENCES_ROOT = HERE / "Old-sequences" / "NonOverlap"
OLD_REVENUES_ROOT = HERE / "Old-revenues" / "NonOverlap"
GRIDS_ROOT = HERE / "Grids" / "NonOverlap"
RESULTS_ROOT = HERE / "Results"
GREEDY_SEQUENCES_ROOT = HERE / "Greedy_sequences"
GREEDY_REVENUES_ROOT = HERE / "Greedy_revenues"
CLUSTER_SEQUENCES_ROOT = HERE / "Cluster_sequences"
CLUSTER_REVENUES_ROOT = HERE / "Cluster_revenues"

# ---- Auxiliary outputs - the 7 required items above always run; these are
# extra, optional views not part of that required set, off by default
# since they're slower and not everyone needs them every run ----
RUN_PER_ROUND_MEAN_STD_PLOTS = False   # variance across SimRuns per round, per UAV+total, per strategy
RUN_GIF_GENERATION = False             # animated negotiation replay - slow, produces many large files

# ---- Parallelization - per-UAV-count data loading is independent across
# UAV counts, so it's split across worker processes. 1 = no parallelism. ----
MAX_WORKERS = os.cpu_count() or 1

# ---- Plot styling - change these for font/size/color preferences (e.g.
# for publication) without touching any plotting code below ----
FONT_FAMILY = "Times New Roman"
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 18
DPI = 300

BOX_COLOR = "C0"
BOX_EDGE_COLOR = "black"
MEDIAN_COLOR = "orange"
MEDIAN_LINE_WIDTH = 2
LINE_WIDTH = 1.8               # items 1/2's revenue-vs-round curves
GRID_LINESTYLE = "--"
GRID_ALPHA = 0.5

# ---- Adaptive layout - figure size and label rotation respond to how
# much content is actually being plotted, rather than being fixed, so a
# graph with 2 boxes and one with 17 boxes both render legibly ----
FIG_HEIGHT = 7                          # inches, all boxplots
FIG_WIDTH_MIN = 6                       # inches, floor for narrow plots
FIG_WIDTH_PER_BOX_COMPARISON = 0.8      # items 3,4,6,7: frameworks + baseline (mixed-length labels)
FIG_WIDTH_PADDING_COMPARISON = 2
FIG_WIDTH_PER_BOX_SHARE = 1.0           # item 5: per-UAV share (short, uniform "UAV0".."UAV9" labels)
FIG_WIDTH_PADDING_SHARE = 3
LINE_CHART_FIGSIZE = (9, 6)    # items 1/2's fixed-line-count charts
# ============================================================


class DataLayer:
    """Loading, trust model, and math. No plotting, no matplotlib import
    needed for anything in this class - kept independently testable."""

    @staticmethod
    def _progress(msg):
        """Single overwriting terminal line, so long loads don't look frozen.
        Padded to clear leftover characters from a longer previous message."""
        print(f"\r  {msg}".ljust(100), end="", flush=True)

    # ---- small pure-math helpers ----
    @staticmethod
    def euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def parse_sequence(raw):
        if raw is None or (isinstance(raw, float) and math.isnan(raw)) or raw == "":
            return []
        return [int(x) for x in str(raw).split("-") if x]

    @staticmethod
    def thesis_code(mode_prefix, mode_token, order_token):
        """mode_prefix: 'N' (NonOverlap) or 'O' (Overlap). mode_token: 'GG'/'GR'/'RG'/'RR'.
        order_token: 'Sequential' or 'Random'. Returns e.g. 'NSGG', 'NRGR'."""
        visit = "S" if order_token.lower().startswith("seq") else "R"
        return f"{mode_prefix}{visit}{mode_token}"

    @staticmethod
    def parse_strategy_from_filename(stem):
        """Extract (mode_token, order_token) from a Games.py-style filename stem
        containing 'Mode{XX}_{Order}', e.g. 'UAVs3_GRID13_ModeGG_Sequential' -> ('GG','Sequential')."""
        m = re.search(r"Mode([GR]{2})_(Sequential|Random)", stem, re.IGNORECASE)
        if not m:
            return None, None
        return m.group(1).upper(), m.group(2)

    @staticmethod
    def solve_tour(coords, mj, speed):
        """Given a FIXED order of coords and a TRUSTED mj, compute tour_time.
        Mirrors PathOptimizer.simulate_mj's distance formula exactly, just using
        mj as given rather than solving for it."""
        n = len(coords)
        if n == 0:
            return 0.0
        depot = (0.0, 0.0)
        first = DataLayer.euclidean(depot, coords[0])
        fwd = sum(DataLayer.euclidean(coords[i], coords[i + 1]) for i in range(n - 1))
        ret = DataLayer.euclidean(coords[-1], depot)
        jump = DataLayer.euclidean(coords[-1], coords[0]) if n > 1 else 0.0
        total_dist = first + mj * fwd + (mj - 1) * jump + ret
        return total_dist / speed

    # ---- file readers ----
    @staticmethod
    def load_waypoints_sheet(wp_xl, sim_run):
        """Returns {wp_id: (x, y, revenue)} for one SimRun sheet. wp_xl is an
        already-open pd.ExcelFile (not a path) - callers open it once per
        UAV count and reuse it across all 100 SimRuns, since re-opening a
        workbook per sheet read is dramatically slower (confirmed ~18x on
        even a small test file) for no benefit."""
        df = wp_xl.parse(f"SimRun{sim_run}")
        return {
            int(row.Waypoint): (float(row.X), float(row.Y), float(row.Revenue))
            for row in df.itertuples()
        }

    @staticmethod
    def load_revenue_curve(rev_f, n_runs=100):
        """Mean total revenue rate per negotiation round, averaged across all
        SimRun sheets in one revenue workbook. Reads revenue directly from the
        file (never computed) - returns a 1D array indexed by round, forward-
        filled to the longest SimRun's round count so short runs don't drag
        the tail down."""
        try:
            xl = pd.ExcelFile(rev_f)
        except Exception:
            return None
        sheets = []
        for sim_run in range(1, n_runs + 1):
            sheet_name = f"SimRun{sim_run}"
            if sheet_name not in xl.sheet_names:
                continue
            df = xl.parse(sheet_name)
            uav_cols = [c for c in df.columns if str(c).upper().startswith("UAV")]
            totals = df[uav_cols].sum(axis=1).values
            sheets.append(totals)
        if not sheets:
            return None
        max_rounds = max(len(s) for s in sheets)
        padded = []
        for s in sheets:
            if len(s) < max_rounds:
                s = list(s) + [s[-1]] * (max_rounds - len(s))
            padded.append(s)
        arr = pd.DataFrame(padded)
        return arr.mean(axis=0).values  # length max_rounds

    @staticmethod
    def revenue_file_for_strategy(rev_dir, uav_count, mode_token, order_token, grid_width=13):
        return Path(rev_dir) / f"UAVs{uav_count}_GRID{grid_width}_Mode{mode_token}_{order_token}.xlsx"

    @staticmethod
    def flight_time_left_for_row(seq_row, num_uavs, wp_coords, speed, max_flight_time,
                                  uav_prefix="UAV", mj_prefix="m_"):
        """Returns {uav_idx: seconds_left} for one row, trusting mj as given."""
        left = {}
        for u in range(num_uavs):
            ids = DataLayer.parse_sequence(seq_row.get(f"{uav_prefix}{u}"))
            if not ids:
                left[u] = float(max_flight_time)
                continue
            mj = int(seq_row[f"{mj_prefix}{u}"])
            coords = [wp_coords[i][:2] for i in ids if i in wp_coords]
            tour_time = DataLayer.solve_tour(coords, mj, speed)
            left[u] = max(max_flight_time - tour_time, 0.0)
        return left

    # ---- source-specific loaders. Each returns a per-strategy dict of
    # per-SimRun records: {code: [{"sim_run": k, "final_rates": {u: rate},
    # "final_left": {u: seconds}, "total": float}, ...]} ----

    @staticmethod
    def load_random_initial(old_seq_root, old_rev_root, grids_root, uav_count, sim_index,
                             speed, max_flight_time, n_runs=100, grid_width=13):
        """Random-initial NonOverlap data for one UAV count. old_seq_root/old_rev_root
        are the Old-sequences/Old-revenues NonOverlap folders; sim_index is which
        Simulation_N (1-8) holds this uav_count (found by filename, not assumed order).
        Revenue is trusted as-is (already repaired); sequences give us mj (trusted)
        for flight-time-left."""
        seq_dir = Path(old_seq_root) / f"Simulation_{sim_index}"
        rev_dir = Path(old_rev_root) / f"Simulation_{sim_index}"
        wp_path = Path(grids_root) / f"UAVs{uav_count}_GRID{grid_width}_waypoints.xlsx"
        wp_xl = pd.ExcelFile(wp_path)

        strategy_files = sorted(seq_dir.glob(f"UAVs{uav_count}_GRID{grid_width}_*_sequences.xlsx"))
        results = {}
        for si, seq_f in enumerate(strategy_files, start=1):
            stem = seq_f.stem.replace("_sequences", "")
            mode_token, order_token = DataLayer.parse_strategy_from_filename(stem)
            if mode_token is None:
                continue
            code = DataLayer.thesis_code("N", mode_token, order_token)
            rev_f = rev_dir / f"UAVs{uav_count}_GRID{grid_width}_Mode{mode_token}_{order_token}.xlsx"
            if not rev_f.exists():
                print(f"\n[WARN] missing revenue file for {seq_f.name}: {rev_f}")
                continue

            seq_xl, rev_xl = pd.ExcelFile(seq_f), pd.ExcelFile(rev_f)
            records = []
            for sim_run in range(1, n_runs + 1):
                DataLayer._progress(f"random-initial UAV={uav_count}: {code} ({si}/{len(strategy_files)}) SimRun {sim_run}/{n_runs}")
                sheet = f"SimRun{sim_run}"
                if sheet not in seq_xl.sheet_names or sheet not in rev_xl.sheet_names:
                    continue
                seq_df, rev_df = seq_xl.parse(sheet), rev_xl.parse(sheet)
                wp_coords = DataLayer.load_waypoints_sheet(wp_xl, sim_run)
                last_seq, last_rev = seq_df.iloc[-1], rev_df.iloc[-1]
                final_rates = {u: float(last_rev[f"UAV{u}"]) for u in range(uav_count)}
                final_left = DataLayer.flight_time_left_for_row(last_seq, uav_count, wp_coords, speed, max_flight_time)
                records.append({
                    "sim_run": sim_run, "final_rates": final_rates,
                    "final_left": final_left, "total": sum(final_rates.values()),
                })
            results[code] = records
        print()  # move off the progress line
        return results

    @staticmethod
    def load_kmeans_initial(results_root, uav_count, sim_index,
                             speed, max_flight_time, n_runs=100, grid_width=13):
        """K-means-initial NonOverlap data - straight Games.py output under
        Results/NonOverlap/simulation_N/{sequences,revenue,waypoints}/.
        Revenue is trusted as Games.py's own output (already verified correct)."""
        base = Path(results_root) / "NonOverlap" / f"simulation_{sim_index}"
        seq_dir, rev_dir, wp_dir = base / "sequences", base / "revenue", base / "waypoints"
        wp_path = next(wp_dir.glob(f"UAVs{uav_count}_GRID{grid_width}_waypoints.xlsx"), None)
        if wp_path is None:
            raise FileNotFoundError(f"No waypoints file for UAVs{uav_count} under {wp_dir}")
        wp_xl = pd.ExcelFile(wp_path)

        strategy_files = sorted(seq_dir.glob(f"UAVs{uav_count}_GRID{grid_width}_*_sequences.xlsx"))
        results = {}
        for si, seq_f in enumerate(strategy_files, start=1):
            stem = seq_f.stem.replace("_sequences", "")
            mode_token, order_token = DataLayer.parse_strategy_from_filename(stem)
            if mode_token is None:
                continue
            code = DataLayer.thesis_code("N", mode_token, order_token)
            rev_f = rev_dir / f"UAVs{uav_count}_GRID{grid_width}_Mode{mode_token}_{order_token}.xlsx"
            if not rev_f.exists():
                print(f"\n[WARN] missing revenue file for {seq_f.name}: {rev_f}")
                continue

            seq_xl, rev_xl = pd.ExcelFile(seq_f), pd.ExcelFile(rev_f)
            records = []
            for sim_run in range(1, n_runs + 1):
                DataLayer._progress(f"kmeans-initial UAV={uav_count}: {code} ({si}/{len(strategy_files)}) SimRun {sim_run}/{n_runs}")
                sheet = f"SimRun{sim_run}"
                if sheet not in seq_xl.sheet_names or sheet not in rev_xl.sheet_names:
                    continue
                seq_df, rev_df = seq_xl.parse(sheet), rev_xl.parse(sheet)
                wp_coords = DataLayer.load_waypoints_sheet(wp_xl, sim_run)
                last_seq, last_rev = seq_df.iloc[-1], rev_df.iloc[-1]
                final_rates = {u: float(last_rev[f"UAV{u}"]) for u in range(uav_count)}
                final_left = DataLayer.flight_time_left_for_row(last_seq, uav_count, wp_coords, speed, max_flight_time)
                records.append({
                    "sim_run": sim_run, "final_rates": final_rates,
                    "final_left": final_left, "total": sum(final_rates.values()),
                })
            results[code] = records
        print()  # move off the progress line
        return results

    @staticmethod
    def load_greedy(greedy_seq_root, greedy_rev_root, grids_root, uav_count,
                     speed, max_flight_time, n_runs=100, grid_width=13):
        """Greedy baseline (Thais's data): round 0 only. Revenue trusted as given."""
        seq_f = Path(greedy_seq_root) / f"UAVs{uav_count}_GRID{grid_width}_{int(max_flight_time)}_{int(speed)}_Greedy_sequences.xlsx"
        rev_f = Path(greedy_rev_root) / f"UAVs{uav_count}_GRID{grid_width}_Greedy.xlsx"
        wp_path = Path(grids_root) / f"UAVs{uav_count}_GRID{grid_width}_waypoints.xlsx"
        seq_xl, rev_xl, wp_xl = pd.ExcelFile(seq_f), pd.ExcelFile(rev_f), pd.ExcelFile(wp_path)

        records = []
        for sim_run in range(1, n_runs + 1):
            DataLayer._progress(f"Greedy UAV={uav_count}: SimRun {sim_run}/{n_runs}")
            sheet = f"SimRun{sim_run}"
            if sheet not in seq_xl.sheet_names or sheet not in rev_xl.sheet_names:
                continue
            seq_df, rev_df = seq_xl.parse(sheet), rev_xl.parse(sheet)
            wp_coords = DataLayer.load_waypoints_sheet(wp_xl, sim_run)
            row0_seq, row0_rev = seq_df.iloc[0], rev_df.iloc[0]
            final_rates = {u: float(row0_rev[f"UAV{u}"]) for u in range(uav_count)}
            final_left = DataLayer.flight_time_left_for_row(row0_seq, uav_count, wp_coords, speed, max_flight_time)
            records.append({
                "sim_run": sim_run, "final_rates": final_rates,
                "final_left": final_left, "total": sum(final_rates.values()),
            })
        print()
        return records

    @staticmethod
    def load_cluster_ga(cluster_seq_root, cluster_rev_root, grids_root, uav_count,
                         speed, max_flight_time, n_runs=100, grid_width=13):
        """Cluster+GA baseline (Thais's K-means data): round 0 only. Revenue
        trusted as given (Cluster_revenues), sequences+mj used for flight-time-left."""
        seq_f = Path(cluster_seq_root) / f"UAVs{uav_count}_GRID{grid_width}_{int(max_flight_time)}_{int(speed)}_cluster_ga_sequences.xlsx"
        rev_f = Path(cluster_rev_root) / f"UAVs{uav_count}_GRID{grid_width}_cluster_ga.xlsx"
        wp_path = Path(grids_root) / f"UAVs{uav_count}_GRID{grid_width}_waypoints.xlsx"
        seq_xl, rev_xl, wp_xl = pd.ExcelFile(seq_f), pd.ExcelFile(rev_f), pd.ExcelFile(wp_path)

        records = []
        for sim_run in range(1, n_runs + 1):
            DataLayer._progress(f"Cluster+GA UAV={uav_count}: SimRun {sim_run}/{n_runs}")
            sheet = f"SimRun{sim_run}"
            if sheet not in seq_xl.sheet_names or sheet not in rev_xl.sheet_names:
                continue
            seq_df, rev_df = seq_xl.parse(sheet), rev_xl.parse(sheet)
            wp_coords = DataLayer.load_waypoints_sheet(wp_xl, sim_run)
            row0_seq, row0_rev = seq_df.iloc[0], rev_df.iloc[0]
            final_rates = {u: float(row0_rev[f"UAV{u}"]) for u in range(uav_count)}
            final_left = DataLayer.flight_time_left_for_row(row0_seq, uav_count, wp_coords, speed, max_flight_time)
            records.append({
                "sim_run": sim_run, "final_rates": final_rates,
                "final_left": final_left, "total": sum(final_rates.values()),
            })
        print()
        return records

    # ---- tournament selection ----
    @staticmethod
    def tournament_best(strategy_records):
        """strategy_records: {code: [{"sim_run": k, "total": float, ...}, ...]}
        For each SimRun, whichever strategy has the highest total wins that
        SimRun. Returns (best_code, win_counts dict)."""
        by_sim_run = defaultdict(dict)
        for code, records in strategy_records.items():
            for rec in records:
                by_sim_run[rec["sim_run"]][code] = rec["total"]

        win_counts = defaultdict(int)
        for sim_run, totals in by_sim_run.items():
            if not totals:
                continue
            winner = max(totals, key=totals.get)
            win_counts[winner] += 1

        if not win_counts:
            return None, {}
        best_code = max(win_counts, key=win_counts.get)
        return best_code, dict(win_counts)


class Plots:
    """The 7 required plot types, styled to match the images approved for
    the thesis presentation. All styling (font, sizes, colors) and layout
    (figure sizing, label rotation) constants live in the CONFIGURATION
    block at the top of the file, not here - this class only reads them."""

    @staticmethod
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

    @staticmethod
    def _style_boxplot(ax, bp):
        for box in bp["boxes"]:
            box.set_facecolor(BOX_COLOR)
            box.set_edgecolor(BOX_EDGE_COLOR)
        for median in bp["medians"]:
            median.set(color=MEDIAN_COLOR, linewidth=MEDIAN_LINE_WIDTH)

    @staticmethod
    def _box_figsize(n_boxes, width_per_box, padding):
        """Figure width grows with how many boxes are actually being shown,
        so a 2-box and a 17-box comparison both get legible spacing instead
        of a fixed size that's cramped for one and wasteful for the other."""
        width = max(FIG_WIDTH_MIN, width_per_box * n_boxes + padding)
        return (width, FIG_HEIGHT)

    @staticmethod
    def _auto_rotate_xticklabels(fig, ax):
        """Rotates x-tick labels to 45 degrees only if they'd actually
        overlap at 0 degrees - measured from the real rendered text extents
        (so it automatically adapts to whatever font size is configured,
        rather than a guessed label-count threshold that could be wrong
        for a different font size or label length mix)."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tick_labels = ax.get_xticklabels()
        if len(tick_labels) < 2:
            return
        boxes = [t.get_window_extent(renderer=renderer) for t in tick_labels]
        overlap = any(boxes[i].x1 > boxes[i + 1].x0 for i in range(len(boxes) - 1))
        if overlap:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        else:
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # ---- Items 1 & 2: mean total revenue rate vs round, one graph per UAV
    # count, one line per framework - matches image 1's actual shape ----
    @staticmethod
    def mean_revenue_by_framework(curves_by_code, uav_count, out_path):
        """curves_by_code: {framework_code: 1D array of mean total revenue per round}.
        Different frameworks can converge (stop needing further rounds) at
        different round counts - each shorter curve is extended flat
        (repeating its OWN last value, i.e. what it stayed at once converged)
        out to the longest framework's round count for this UAV fleet size,
        so every line spans the same x-axis and can be visually compared
        all the way across, matching how a converged negotiation would
        simply stay at its final value if more rounds were run."""
        max_len = max(len(c) for c in curves_by_code.values())
        fig, ax = plt.subplots(figsize=LINE_CHART_FIGSIZE, layout="constrained")
        for code in sorted(curves_by_code):
            curve = curves_by_code[code]
            if len(curve) < max_len:
                curve = np.concatenate([curve, np.full(max_len - len(curve), curve[-1])])
            ax.plot(np.arange(len(curve)), curve, label=code, linewidth=LINE_WIDTH)
        ax.set_xlabel("Negotiation round")
        ax.set_ylabel("Mean total revenue rate")
        ax.set_title(f"{uav_count} UAVs")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # ---- Items 3, 4, 6, 7: all 8 frameworks + baseline boxplot, one per UAV count ----
    @staticmethod
    def multi_box_comparison(values_by_label, ylabel, out_path):
        """values_by_label: ordered list of (label, values) pairs - all 8
        frameworks followed by the baseline, so the baseline reads as the
        reference point on the right, matching image 2's layout."""
        labels = [label for label, _ in values_by_label]
        data = [vals for _, vals in values_by_label]
        fig, ax = plt.subplots(
            figsize=Plots._box_figsize(len(labels), FIG_WIDTH_PER_BOX_COMPARISON, FIG_WIDTH_PADDING_COMPARISON),
            layout="constrained",
        )
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        Plots._style_boxplot(ax, bp)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
        Plots._auto_rotate_xticklabels(fig, ax)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # ---- Item 5: per-UAV % share boxplot, one entity at one UAV count ----
    @staticmethod
    def per_uav_share(shares_by_uav, title, out_path):
        """shares_by_uav: {uav_idx: [share_run1, share_run2, ...]} (fractions 0-1).
        Y-axis is sized to the actual data range with a small margin, rather
        than always anchoring at 0% - per-UAV shares typically cluster in a
        narrow band around 1/num_uavs, so a fixed 0%-to-rounded-max axis
        wastes most of the figure on empty space above and below the real
        data, which shrinks the actual content when printed/published."""
        uav_indices = sorted(shares_by_uav)
        data = [shares_by_uav[u] for u in uav_indices]
        labels = [f"UAV{u}" for u in uav_indices]

        fig, ax = plt.subplots(
            figsize=Plots._box_figsize(len(labels), FIG_WIDTH_PER_BOX_SHARE, FIG_WIDTH_PADDING_SHARE),
            layout="constrained",
        )
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        Plots._style_boxplot(ax, bp)
        ax.set_ylabel("Share of total revenue rate")
        ax.set_title(title)

        all_vals = (np.concatenate([np.asarray(d, float) for d in data if len(d)])
                    if any(len(d) for d in data) else np.array([0.0, 0.1]))
        data_min, data_max = float(all_vals.min()), float(all_vals.max())
        data_range = max(data_max - data_min, 1e-6)
        pad = max(data_range * 0.12, 0.01)  # at least 1 percentage point, so points aren't flush to the edge
        y_lo = max(0.0, data_min - pad)
        y_hi = min(1.0, data_max + pad)
        ax.set_ylim(y_lo, y_hi)
        decimals = 1 if (y_hi - y_lo) <= 0.03 else 0  # avoid duplicate rounded labels on a very tight axis
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=decimals))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
        Plots._auto_rotate_xticklabels(fig, ax)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Orchestration - one-shot script logic, not reusable utilities,
# so kept as plain functions rather than a third class.
# ============================================================

# =====================================================================
# AUXILIARY OUTPUTS (optional, toggled off by default - see
# RUN_PER_ROUND_MEAN_STD_PLOTS / RUN_GIF_GENERATION in CONFIGURATION)
# =====================================================================
# These are extra views beyond the 7 required items, not a replacement
# for anything above. Rewritten from the original Analysis.py's
# equivalents: rewired to the CURRENT flat Results/{mode}/simulation_N/
# layout (no date folders), and using DataLayer's existing filename/sheet
# parsing instead of re-implementing it.
#
# Four of the original script's functions are NOT here because items
# 1-7 above already cover the same ground with corrected logic (proper
# cross-UAV-count aggregation, all-frameworks-vs-baseline comparisons,
# tournament-selected "best", last-round consistency) - keeping both
# would just be redundant, verbose duplication:
#   plot_consolidated_total_revenue   -> superseded by items 1/2
#   boxplot_final_totals_with_irada   -> superseded by items 3/4
#   boxplot_flight_time_left          -> superseded by items 6/7 (and had
#                                         a real bug: hardcoded 0.0 flight-
#                                         time-left for any single-waypoint
#                                         tour, regardless of its actual mj)
#   boxplot_uav_contribution_all      -> superseded by item 5 (and used
#                                         "round of max total" instead of
#                                         "last round", an inconsistency
#                                         item 5 deliberately avoids)
# =====================================================================

def per_round_mean_std_plots(rev_dir, out_dir, mode_prefix, n_runs=100):
    """For every strategy revenue file in rev_dir, plots mean+/-std
    shading per round: one plot per UAV, one for the total, one
    consolidated view with all UAVs plus mean(total)/n_uavs. Shows
    variance ACROSS SimRuns at each round for one strategy at a time -
    complements items 1/2, which compare strategies against each other
    but don't show any one strategy's own spread."""
    rev_dir = Path(rev_dir)
    for fpath in sorted(rev_dir.glob("UAVs*.xlsx")):
        mode_token, order_token = DataLayer.parse_strategy_from_filename(fpath.stem)
        if mode_token is None:
            continue
        code = DataLayer.thesis_code(mode_prefix, mode_token, order_token)

        xl = pd.ExcelFile(fpath)
        dfs = [xl.parse(s) for s in xl.sheet_names[:n_runs]]
        if not dfs:
            continue
        max_rounds = max(len(df) for df in dfs)
        padded = [df.reindex(range(max_rounds)).ffill() for df in dfs]
        uav_cols = [c for c in padded[0].columns if str(c).upper().startswith("UAV")]
        if not uav_cols:
            continue
        n_uavs = len(uav_cols)

        arr = np.stack([df[uav_cols].values for df in padded], axis=0)
        mean_uav, std_uav = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        tot_arr = arr.sum(axis=2)
        mean_tot, std_tot = tot_arr.mean(axis=0), tot_arr.std(axis=0, ddof=1)
        rounds = np.arange(max_rounds)

        code_dir = Path(out_dir) / code
        code_dir.mkdir(parents=True, exist_ok=True)

        for i, col in enumerate(uav_cols):
            fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
            ax.plot(rounds, mean_uav[:, i], label=f"mean {col}")
            ax.fill_between(rounds, mean_uav[:, i] - std_uav[:, i], mean_uav[:, i] + std_uav[:, i], alpha=0.2)
            ax.set_xlabel("Negotiation round")
            ax.set_ylabel("Revenue rate")
            ax.legend()
            fig.savefig(code_dir / f"{col}_mean_std.png", dpi=DPI)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
        ax.plot(rounds, mean_tot, label="mean total")
        ax.fill_between(rounds, mean_tot - std_tot, mean_tot + std_tot, alpha=0.2)
        ax.set_xlabel("Negotiation round")
        ax.set_ylabel("Total revenue rate")
        ax.legend()
        fig.savefig(code_dir / "Total_mean_std.png", dpi=DPI)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5 + 0.4 * max(0, n_uavs - 3), 4.5), layout="constrained")
        for i, col in enumerate(uav_cols):
            ax.plot(rounds, mean_uav[:, i], label=col)
        ax.plot(rounds, mean_tot / n_uavs, label="mean total / n_uavs", linewidth=2, linestyle="--")
        ax.set_xlabel("Negotiation round")
        ax.set_ylabel("Revenue rate")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)
        fig.savefig(code_dir / "Consolidated_mean.png", dpi=DPI)
        plt.close(fig)


def generate_gifs_from_sequences(seq_dir, rev_dir, wp_dir, out_dir, mode_prefix, n_runs=100):
    """Animated replay of the negotiation for every strategy/SimRun in
    seq_dir: each UAV's tour drawn frame-by-frame across rounds, with a
    sidebar showing mj/revenue per UAV and the running total. Genuinely
    unique output, not covered by items 1-7."""
    seq_dir, rev_dir, wp_dir = Path(seq_dir), Path(rev_dir), Path(wp_dir)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    zero_color, nonzero_color = "#CCCCCC", "#111111"

    for seq_file in sorted(seq_dir.glob("UAVs*_sequences.xlsx")):
        stem = seq_file.stem.replace("_sequences", "")
        mode_token, order_token = DataLayer.parse_strategy_from_filename(stem)
        if mode_token is None:
            continue
        code = DataLayer.thesis_code(mode_prefix, mode_token, order_token)
        uavs_token, grid_token = stem.split("_")[0], stem.split("_")[1]
        rev_f = rev_dir / f"{uavs_token}_{grid_token}_Mode{mode_token}_{order_token}.xlsx"
        wp_f = next(wp_dir.glob(f"{uavs_token}_{grid_token}_waypoints.xlsx"), None)
        if not rev_f.exists() or wp_f is None:
            print(f"[SKIP] {stem}: missing matching revenue or waypoints file")
            continue

        seq_xl, rev_xl, wp_xl = pd.ExcelFile(seq_file), pd.ExcelFile(rev_f), pd.ExcelFile(wp_f)
        code_dir = Path(out_dir) / code
        code_dir.mkdir(parents=True, exist_ok=True)

        for sim_run in range(1, n_runs + 1):
            sheet = f"SimRun{sim_run}"
            if sheet not in seq_xl.sheet_names or sheet not in rev_xl.sheet_names:
                continue
            seq_df, rev_df = seq_xl.parse(sheet), rev_xl.parse(sheet)
            n_seq, n_rev = len(seq_df), len(rev_df)
            if n_rev < n_seq:
                rev_df = pd.concat([rev_df, pd.DataFrame([rev_df.iloc[-1]] * (n_seq - n_rev))], ignore_index=True)
            elif n_rev > n_seq:
                rev_df = rev_df.iloc[:n_seq].copy()
            seq_df, rev_df = seq_df.reset_index(drop=True), rev_df.reset_index(drop=True)

            wp_coords = DataLayer.load_waypoints_sheet(wp_xl, sim_run)
            coords = {i: (x, y, rev) for i, (x, y, rev) in wp_coords.items()}
            xs_all = [c[0] for c in coords.values()]
            ys_all = [c[1] for c in coords.values()]
            spacing = abs(sorted(set(xs_all))[1] - sorted(set(xs_all))[0]) if len(set(xs_all)) > 1 else 1.0

            uav_cols = [c for c in seq_df.columns if str(c).upper().startswith("UAV")]
            n_uavs = len(uav_cols)
            x_span, y_span = max(xs_all) - min(xs_all), max(ys_all) - min(ys_all)
            fig = plt.figure(figsize=(x_span * 0.5 / 100 + 2.5, y_span * 0.5 / 100 + 1.0))
            fig.text(0.5, 0.98, f"{code}  SimRun {sim_run}", ha="center", va="top", fontsize=16, weight="bold")
            ax = fig.add_axes([0.05, 0.05, 0.70, 0.88])
            ax.set_xlim(min(xs_all) - spacing, max(xs_all) + spacing)
            ax.set_ylim(min(ys_all) - spacing, max(ys_all) + spacing)
            wp_xs, wp_ys, wp_cs = zip(*[(x, y, nonzero_color if rev > 0 else zero_color) for x, y, rev in coords.values()])

            dynamic_texts = []

            def update(frame):
                for line in list(ax.lines):
                    line.remove()
                for coll in ax.collections[2:]:
                    coll.remove()
                for txt in dynamic_texts:
                    txt.remove()
                dynamic_texts.clear()
                ax.scatter(wp_xs, wp_ys, c=wp_cs, s=80, zorder=1)
                ax.scatter([0], [0], marker="*", color="k", s=200, zorder=2)

                for u in range(n_uavs):
                    ids = DataLayer.parse_sequence(seq_df.iloc[frame][f"UAV{u}"])
                    pts = [(0, 0)] + [(coords[i][0], coords[i][1]) for i in ids if i in coords] + [(0, 0)]
                    px, py = zip(*pts)
                    ax.plot(px, py, color=colors[u % len(colors)], lw=2)
                    m_val = seq_df.iloc[frame].get(f"m_{u}", 1)
                    z_val = rev_df.iloc[frame][f"UAV{u}"]
                    t = fig.text(0.78, 0.88 - u * 0.06, rf"$m_{{{u}}}$={m_val}, $Z_{{{u}}}$={z_val:.2f}",
                                 va="top", ha="left", color=colors[u % len(colors)], fontsize=12)
                    dynamic_texts.append(t)

                z_tot = rev_df[uav_cols].iloc[frame].sum()
                dynamic_texts.append(fig.text(0.78, 0.88 - n_uavs * 0.06, rf"$Z$={z_tot:.2f}",
                                               va="top", ha="left", fontsize=12))
                dynamic_texts.append(fig.text(0.5, 0.93, f"Round {frame}", ha="center", va="top", fontsize=12))

            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter
            ani = animation.FuncAnimation(fig, update, frames=len(seq_df), interval=800)
            gif_path = code_dir / f"{stem}_SimRun{sim_run}.gif"
            ani.save(str(gif_path), writer=PillowWriter(fps=1))
            plt.close(fig)
        print(f"[GIF] {code}: saved to {code_dir}")



def discover_simulation_index(root, expected_uav_count, folder_prefix="Simulation_"):
    """Scan {root}/{folder_prefix}{1..8} for one containing files whose
    UAVs{N} token matches expected_uav_count. Returns the index, or None."""
    root = Path(root)
    for i in range(1, 9):
        candidate = root / f"{folder_prefix}{i}"
        if not candidate.exists():
            continue
        for f in candidate.glob(f"UAVs{expected_uav_count}_GRID*_*"):
            return i
    return None


def discover_results_sim_index(results_root, expected_uav_count):
    """Same idea for Results/NonOverlap/simulation_N/ - scan each for a
    waypoints file matching the UAV count."""
    base = Path(results_root) / "NonOverlap"
    if not base.exists():
        return None
    for sim_dir in sorted(base.glob("simulation_*")):
        wp_dir = sim_dir / "waypoints"
        if not wp_dir.exists():
            continue
        if any(wp_dir.glob(f"UAVs{expected_uav_count}_GRID*_waypoints.xlsx")):
            m = re.search(r"simulation_(\d+)", sim_dir.name)
            if m:
                return int(m.group(1))
    return None


def _load_one_uav_count(args):
    """Loads all 4 data sources + tournament selection for one UAV count.
    Module-level (not a method/closure) so it's picklable and can run in
    a separate worker process - each UAV count's data is completely
    independent of every other's, so this is the natural place to
    parallelize: the slow part is hundreds of small Excel reads per UAV
    count, not the (fast, in-memory) plotting that happens afterward.
    Returns (uav_count, result_dict_or_None, skip_reasons)."""
    (uav_count, old_seq_root, old_rev_root, grids_root, results_root,
     greedy_seq_root, greedy_rev_root, cluster_seq_root, cluster_rev_root,
     speed, max_ft, n_runs, grid_width) = args

    skip_reasons = []
    random_sim_idx = discover_simulation_index(old_seq_root, uav_count)
    kmeans_sim_idx = discover_results_sim_index(results_root, uav_count)
    if random_sim_idx is None:
        skip_reasons.append((uav_count, "random-initial", "no matching Simulation_N folder found"))
    if kmeans_sim_idx is None:
        skip_reasons.append((uav_count, "kmeans-initial", "no matching Results/NonOverlap/simulation_N found"))
    if random_sim_idx is None or kmeans_sim_idx is None:
        return uav_count, None, skip_reasons

    random_data = DataLayer.load_random_initial(old_seq_root, old_rev_root, grids_root,
                                                  uav_count, random_sim_idx, speed, max_ft, n_runs, grid_width)
    kmeans_data = DataLayer.load_kmeans_initial(results_root, uav_count, kmeans_sim_idx,
                                                  speed, max_ft, n_runs, grid_width)
    greedy_data = DataLayer.load_greedy(greedy_seq_root, greedy_rev_root, grids_root,
                                         uav_count, speed, max_ft, n_runs, grid_width)
    cluster_data = DataLayer.load_cluster_ga(cluster_seq_root, cluster_rev_root, grids_root,
                                              uav_count, speed, max_ft, n_runs, grid_width)
    best_random, random_wins = DataLayer.tournament_best(random_data)
    best_kmeans, kmeans_wins = DataLayer.tournament_best(kmeans_data)

    result = dict(
        random_data=random_data, kmeans_data=kmeans_data,
        greedy_data=greedy_data, cluster_data=cluster_data,
        best_random=best_random, best_kmeans=best_kmeans,
        random_wins=random_wins, kmeans_wins=kmeans_wins,
        random_sim_idx=random_sim_idx, kmeans_sim_idx=kmeans_sim_idx,
    )
    return uav_count, result, skip_reasons


def run():
    out_root = VISUALIZATIONS_ROOT
    speed, max_ft, n_runs = SPEED, MAX_FLIGHT_TIME, N_RUNS

    old_seq_root = OLD_SEQUENCES_ROOT
    old_rev_root = OLD_REVENUES_ROOT
    grids_root = GRIDS_ROOT
    results_root = RESULTS_ROOT
    greedy_seq_root = GREEDY_SEQUENCES_ROOT
    greedy_rev_root = GREEDY_REVENUES_ROOT
    cluster_seq_root = CLUSTER_SEQUENCES_ROOT
    cluster_rev_root = CLUSTER_REVENUES_ROOT

    Plots.set_plot_style()

    # ---- Per-UAV-count data, cached for reuse across items 1,2,3,4,5,6,7.
    # Each UAV count's loading is independent of every other's, so this
    # is dispatched across worker processes (MAX_WORKERS in CONFIGURATION)
    # rather than run one at a time. ----
    per_uav = {}
    skipped = []

    load_args = [
        (uav_count, old_seq_root, old_rev_root, grids_root, results_root,
         greedy_seq_root, greedy_rev_root, cluster_seq_root, cluster_rev_root,
         speed, max_ft, n_runs, GRID_WIDTH)
        for uav_count in UAV_COUNTS
    ]

    def _report(uav_count, result):
        print(f"[UAV={uav_count}] best random-initial: {result['best_random']} {result['random_wins']}")
        print(f"[UAV={uav_count}] best kmeans-initial:  {result['best_kmeans']} {result['kmeans_wins']}")

    if MAX_WORKERS > 1 and len(UAV_COUNTS) > 1:
        print(f"\nLoading data for {len(UAV_COUNTS)} UAV count(s) across up to {MAX_WORKERS} worker process(es)...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_load_one_uav_count, args): args[0] for args in load_args}
            for future in as_completed(futures):
                uav_count, result, skip_reasons = future.result()
                skipped.extend(skip_reasons)
                if result is not None:
                    per_uav[uav_count] = result
                    _report(uav_count, result)
    else:
        for args in load_args:
            uav_count = args[0]
            print(f"\n=== UAV count {uav_count} ({UAV_COUNTS.index(uav_count) + 1}/{len(UAV_COUNTS)}) ===")
            uav_count, result, skip_reasons = _load_one_uav_count(args)
            skipped.extend(skip_reasons)
            if result is not None:
                per_uav[uav_count] = result
                _report(uav_count, result)

    if skipped:
        print("\n[WARN] Skipped UAV counts (missing data):")
        for uav_count, scenario, reason in skipped:
            print(f"  UAV={uav_count} ({scenario}): {reason}")

    strategy_codes = [DataLayer.thesis_code("N", m, o) for m in ("GG", "GR", "RG", "RR")
                       for o in ("Sequential", "Random")]

    # ================================================================
    # Items 1 & 2: mean total revenue rate vs round, one graph per UAV
    # count, one line per framework (matches the approved image 1 shape:
    # a fixed UAV count with all 8 strategies compared as lines)
    # ================================================================
    for scenario, rev_root_fn, item_dir in [
        ("random", lambda uc: old_rev_root / f"Simulation_{per_uav[uc]['random_sim_idx']}", "mean_revenue_vs_round_random_initial"),
        ("kmeans", lambda uc: results_root / "NonOverlap" / f"simulation_{per_uav[uc]['kmeans_sim_idx']}" / "revenue", "mean_revenue_vs_round_kmeans_initial"),
    ]:
        print(f"\nBuilding {item_dir}...")
        for ui, uav_count in enumerate(per_uav, start=1):
            DataLayer._progress(f"{item_dir}: UAV={uav_count} ({ui}/{len(per_uav)})")
            curves = {}
            rev_dir = rev_root_fn(uav_count)
            for code in strategy_codes:
                m = re.match(r"N([SR])([GR]{2})", code)
                visit_letter, mode_token = m.group(1), m.group(2)
                order_token = "Sequential" if visit_letter == "S" else "Random"
                rev_f = DataLayer.revenue_file_for_strategy(rev_dir, uav_count, mode_token, order_token, GRID_WIDTH)
                if not rev_f.exists():
                    continue
                curve = DataLayer.load_revenue_curve(rev_f, n_runs=n_runs)
                if curve is not None:
                    curves[code] = curve
            if curves:
                out_path = out_root / item_dir / f"UAV{uav_count}.png"
                Plots.mean_revenue_by_framework(curves, uav_count, out_path)
        print(f"\n[DONE] {item_dir}")

    # ================================================================
    # Items 3 & 4: best-of-8 vs baseline, final total revenue rate,
    # one graph per UAV count
    # ================================================================
    print("\nBuilding final-revenue comparisons...")
    for uav_count, data in per_uav.items():
        random_series = [(code, [r["total"] for r in data["random_data"][code]])
                          for code in strategy_codes if code in data["random_data"]]
        greedy_vals = [r["total"] for r in data["greedy_data"]]
        Plots.multi_box_comparison(
            random_series + [("Greedy", greedy_vals)],
            "Final total revenue rate",
            out_root / "final_revenue_vs_greedy_random_initial" / f"UAV{uav_count}.png",
        )

        kmeans_series = [(code, [r["total"] for r in data["kmeans_data"][code]])
                          for code in strategy_codes if code in data["kmeans_data"]]
        cluster_vals = [r["total"] for r in data["cluster_data"]]
        Plots.multi_box_comparison(
            kmeans_series + [("Cluster+GA", cluster_vals)],
            "Final total revenue rate",
            out_root / "final_revenue_vs_clusterga_kmeans_initial" / f"UAV{uav_count}.png",
        )
    print("[DONE] final-revenue comparisons")

    # ================================================================
    # Item 5: per-UAV % share, best-of-8 / Greedy / best-of-8 / Cluster+GA,
    # one graph per (UAV count, entity)
    # ================================================================
    def shares_from_records(records, uav_count):
        shares = defaultdict(list)
        for rec in records:
            total = rec["total"]
            if total <= 0:
                continue
            for u in range(uav_count):
                shares[u].append(rec["final_rates"][u] / total)
        return shares

    print("\nBuilding per-UAV revenue share...")
    for uav_count, data in per_uav.items():
        best_random_records = data["random_data"][data["best_random"]]
        Plots.per_uav_share(
            shares_from_records(best_random_records, uav_count),
            f"Per-UAV Share - {data['best_random']} (best) - {uav_count} UAVs",
            out_root / "per_uav_revenue_share" / "random_initial_best" / f"UAV{uav_count}.png",
        )
        Plots.per_uav_share(
            shares_from_records(data["greedy_data"], uav_count),
            f"Per-UAV Share - Greedy - {uav_count} UAVs",
            out_root / "per_uav_revenue_share" / "greedy" / f"UAV{uav_count}.png",
        )
        best_kmeans_records = data["kmeans_data"][data["best_kmeans"]]
        Plots.per_uav_share(
            shares_from_records(best_kmeans_records, uav_count),
            f"Per-UAV Share - {data['best_kmeans']} (best) - {uav_count} UAVs",
            out_root / "per_uav_revenue_share" / "kmeans_initial_best" / f"UAV{uav_count}.png",
        )
        Plots.per_uav_share(
            shares_from_records(data["cluster_data"], uav_count),
            f"Per-UAV Share - Cluster+GA - {uav_count} UAVs",
            out_root / "per_uav_revenue_share" / "cluster_ga" / f"UAV{uav_count}.png",
        )
    print("[DONE] per-UAV revenue share")

    # ================================================================
    # Items 6 & 7: flight-time-left, best-of-8 vs baseline, one graph
    # per UAV count
    # ================================================================
    print("\nBuilding flight-time-left comparisons...")
    for uav_count, data in per_uav.items():
        random_left_series = [
            (code, [v for r in data["random_data"][code] for v in r["final_left"].values()])
            for code in strategy_codes if code in data["random_data"]
        ]
        greedy_left = [v for r in data["greedy_data"] for v in r["final_left"].values()]
        Plots.multi_box_comparison(
            random_left_series + [("Greedy", greedy_left)],
            "Flight time left (s)",
            out_root / "flighttime_left_vs_greedy_random_initial" / f"UAV{uav_count}.png",
        )

        kmeans_left_series = [
            (code, [v for r in data["kmeans_data"][code] for v in r["final_left"].values()])
            for code in strategy_codes if code in data["kmeans_data"]
        ]
        cluster_left = [v for r in data["cluster_data"] for v in r["final_left"].values()]
        Plots.multi_box_comparison(
            kmeans_left_series + [("Cluster+GA", cluster_left)],
            "Flight time left (s)",
            out_root / "flighttime_left_vs_clusterga_kmeans_initial" / f"UAV{uav_count}.png",
        )
    print("[DONE] flight-time-left comparisons")

    # ================================================================
    # Auxiliary outputs (optional, off by default - see
    # RUN_PER_ROUND_MEAN_STD_PLOTS / RUN_GIF_GENERATION in CONFIGURATION).
    # Organized per UAV count under Visualizations/, since each UAV
    # count's random-initial and kmeans-initial data live in different
    # simulation_N folders - UAV count is the stable, meaningful key.
    # ================================================================
    if RUN_PER_ROUND_MEAN_STD_PLOTS:
        print("\nBuilding per-round mean+/-std plots...")
        for uav_count, data in per_uav.items():
            random_rev_dir = old_rev_root / f"Simulation_{data['random_sim_idx']}"
            per_round_mean_std_plots(
                random_rev_dir, out_root / f"UAV{uav_count}_random_initial" / "per_round_plots", "N", n_runs,
            )
            kmeans_rev_dir = results_root / "NonOverlap" / f"simulation_{data['kmeans_sim_idx']}" / "revenue"
            per_round_mean_std_plots(
                kmeans_rev_dir, out_root / f"UAV{uav_count}_kmeans_initial" / "per_round_plots", "N", n_runs,
            )
        print("[DONE] per-round mean+/-std plots")

    if RUN_GIF_GENERATION:
        print("\nBuilding negotiation GIFs...")
        for uav_count, data in per_uav.items():
            random_seq_dir = old_seq_root / f"Simulation_{data['random_sim_idx']}"
            random_rev_dir = old_rev_root / f"Simulation_{data['random_sim_idx']}"
            generate_gifs_from_sequences(
                random_seq_dir, random_rev_dir, grids_root,
                out_root / f"UAV{uav_count}_random_initial" / "gifs", "N", n_runs,
            )
            kmeans_base = results_root / "NonOverlap" / f"simulation_{data['kmeans_sim_idx']}"
            generate_gifs_from_sequences(
                kmeans_base / "sequences", kmeans_base / "revenue", kmeans_base / "waypoints",
                out_root / f"UAV{uav_count}_kmeans_initial" / "gifs", "N", n_runs,
            )
        print("[DONE] negotiation GIFs")

    print(f"\nAll outputs written under: {out_root}")


if __name__ == "__main__":
    run()
