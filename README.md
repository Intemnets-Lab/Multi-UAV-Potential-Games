# Multi-UAV Potential Game Framework

> **A game-theoretic approach to decentralized multi-UAV persistent monitoring with convergence guarantees**

This repository implements the algorithms and simulation framework described in the thesis: **"Path Optimization for UAV Waypoint Navigation Using Potential Game Theory"** (Loyola Marymount University, 2025)

### 🎯 What does it do?

Coordinates **3–10 UAVs** to monitor a grid of waypoints (e.g., wildfire perimeters) by:

- Modeling coordination as an **exact potential game** with guaranteed Nash equilibrium convergence
- Linking revisit frequency to **Nyquist sampling requirements** for temporal coverage guarantees
- Supporting **controlled overlap** at high-priority locations for redundancy
- Benchmarking against external allocation strategies (K-means, K-means+GA, greedy) and, optionally, IRADA

### 🚀 Why use this framework?

Unlike heuristic or centralized approaches, this provides:

- ✅ **Convergence guarantees** via potential game theory
- ✅ **Decentralized negotiation** (no single point of failure)
- ✅ **Tunable redundancy** (overlap mode for safety-critical regions)
- ✅ **Reproducible benchmarking** (open-source outputs, configs, plots)

---

## 📦 Quick Start (2 minutes)

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

## Installation & First Run

Before running the simulation framework, ensure you have the following installed:

**Required Software:**

- **Python 3.10+** (tested on 3.10.18-3.11)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

**Recommended Tools:**

- **VS Code** or **PyCharm** (for code editing)
- **Terminal/Command Prompt** (for running scripts)

### Requirements

Create a `requirements.txt` file with:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
pyyaml>=5.4.0
openpyxl>=3.0.0
Pillow>=9.0.0
```

`Pillow` is what actually writes the animated negotiation GIFs (via matplotlib's `PillowWriter`) - it's required if you ever turn `RUN_GIF_GENERATION` on in `Analysis.py`, not optional.

### Step 1: Clone the Repository

```
# Clone the repository
git clone https://github.com/Intemnets-Lab/Multi-UAV-Potential-Games.git

# Navigate into the created directory
cd Multi-UAV-Potential-Games

# Verify you're in the right place
ls
# You should see files like: Games.py, Analysis.py, settings.yaml, make_generic_assignment_template.py, etc.
```

### Step 2: Install Dependencies

Create a virtual environment (recommended) and install required packages:

```
# Create virtual environment
python -m venv PotentialDrones

# Activate virtual environment
# On Windows:
PotentialDrones\Scripts\activate
# On macOS/Linux:
source PotentialDrones/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Core Dependencies** (if `requirements.txt` is missing):

```
pip install numpy pandas matplotlib pyyaml openpyxl Pillow
```

### Step 3: Verify `settings.yaml` Configuration

Open `settings.yaml` and verify/modify the basic parameters:

```yaml
project:
  results_dir: Results
  grids_dir: Grids
  benchmark_dir: Cluster_sequences
  custom_benchmark_dir: Generic_initial_assignment

simulation:
  seed: null                             # or an integer, for reproducible runs
  n_runs: 5                              # number of independent experiments
  enable_logging: true                   # write detailed negotiation logs to disk
  run_modes: ["NonOverlap", "Overlap"]   # run one mode, or both
  initial_assignment_source: random      # random | cluster_ga | kmeans_centroids | generic

grid:
  width: 5                               # grid width (number of columns)
  height: 5                              # grid height (number of rows)
  spacing: 1000                          # spacing between waypoints (meters)
  zero_prob: 0.3                         # probability of zero-revenue waypoints

uav:
  num_uavs: 2                            # number of UAVs
  speed: 20                              # UAV speed (m/s)
  max_flight_time: 1800                  # max flight time (seconds, e.g. 30 min)

revenue:
  random: true                           # draw random revenue (true) or use a fixed value (false)
  fixed_value: 50                        # used only if random: false
  min: 10                                # min random revenue
  max: 100                               # max random revenue

overlap:
  clone_threshold: 300                   # revenue above which a waypoint gets cloned in Overlap mode
  clone_assignment: "random"             # "same" | "random" | "balanced"

algorithms:
  # Any of these can be set to false to skip that strategy for this run.
  # Any toggle you omit defaults to true (enabled).
  sequential_GG: true
  sequential_GR: true
  sequential_RG: true
  sequential_RR: true
  random_GG:     true
  random_GR:     true
  random_RG:     true
  random_RR:     true
```

**Key Parameters for First Run:**

- **`num_uavs: 2`** - Start small to verify the setup works
- **`grid.width: 5`, `grid.height: 5`** - Generates a 5×5 grid (24 waypoints + 1 depot)
- **`n_runs: 5`** - Run 5 simulations for statistical analysis
- **`algorithms`** - Set everything but `sequential_GG` (say) to `false` for faster first-run testing

### Step 4: Run Your First Simulation

#### **Option A: Run Non-Overlap & Overlap Games**

```
python Games.py
```

**What Happens:**

1. Creates (or reuses) `simulation_N` folders under `Results/NonOverlap/` and `Results/Overlap/` - numbered per session, not by calendar date
2. Runs every enabled strategy (from `settings.yaml`), across both modes if `run_modes` includes both
3. Generates Excel files for:
  - **Revenue rates** (`Results/{mode}/simulation_N/revenue/*.xlsx`)
  - **Waypoint sequences** (`Results/{mode}/simulation_N/sequences/*_sequences.xlsx`)
  - **Waypoint grids** (`Results/{mode}/simulation_N/waypoints/*.xlsx`)
4. Writes a per-mode negotiation log if `enable_logging: true`, and a session-level execution trace + progress chart under `Results/Summary/simulation_N/`

**Expected Output:**

```
[INFO] Final config: <Config {...}>
=== Simulation Started === (8 CPU core(s), no GPU detected, running up to 8 combo(s) in parallel)
Preflight: 24 waypoints, tour = 3841.2m, 192.1s of 1800.0s budget — OK
Preflight: 24 waypoints, tour = 3841.2m, 192.1s of 1800.0s budget — OK
  ✅ wrote NonOverlap/ModeGG_Sequential (SimRun 1)
  ✅ wrote Overlap/ModeGG_Sequential (SimRun 1)
  [██░░░░░░░░░░░░░░░░░░░░░░]  2/16 (13%)  elapsed 0m 3s  ETA 0m 18s
  ✅ wrote NonOverlap/ModeGG_Sequential (SimRun 2)
  ...
  [████████████████████████] 16/16 (100%)  elapsed 0m 21s  ETA 0s
  Execution trace written to Results/Summary/simulation_1/execution_trace.xlsx
  Framework progress chart written to Results/Summary/simulation_1/framework_progress.png
=== Simulation Complete ===
```

Preflight runs exactly once per enabled mode per session (not once per SimRun) - Overlap always has at least as many waypoints as NonOverlap (clones add entries), so each mode gets its own feasibility check rather than reusing the other's result.

#### **Option B: Run IRADA Benchmark**

```
python IRADA.py
```

*(Not verified against the current codebase as part of this documentation pass - see the note at the top of the IRADA.py section below.)*

#### **Option C: Run Analysis & Generate Plots**

```
python Analysis.py
```

**What Happens:**

1. Reads paths and settings directly from the `CONFIGURATION` block at the top of `Analysis.py` (there's no separate settings file for this script, and no automatic "find the latest simulation" scanning - point the constants at the folders you want analyzed)
2. Loads data for every UAV fleet size in `UAV_COUNTS`, in parallel across `MAX_WORKERS` processes
3. Runs a tournament across all 8 strategies per fleet size to determine the "best" one, for each of the random-initial and K-means-initial scenarios
4. Generates the 7 required comparison graphs (revenue over rounds, final-total boxplots, per-UAV share, flight-time-left boxplots) for every fleet size
5. Optionally generates per-round mean±std plots and negotiation GIFs, if `RUN_PER_ROUND_MEAN_STD_PLOTS` / `RUN_GIF_GENERATION` are set to `True`
6. Saves everything under `Visualizations/`

**Expected Output:**

```
Loading data for 8 UAV count(s) across up to 8 worker process(es)...
[UAV=3] best random-initial: NRRG {'NRRG': 62, 'NSGG': 38}
[UAV=3] best kmeans-initial:  NSGG {'NSGG': 71, 'NRGG': 29}
...

Building mean_revenue_vs_round_random_initial...
[DONE] mean_revenue_vs_round_random_initial
Building mean_revenue_vs_round_kmeans_initial...
[DONE] mean_revenue_vs_round_kmeans_initial

Building final-revenue comparisons...
[DONE] final-revenue comparisons

Building per-UAV revenue share...
[DONE] per-UAV revenue share

Building flight-time-left comparisons...
[DONE] flight-time-left comparisons

All outputs written under: Visualizations
```

### Step 5: Verify Outputs

```
Results/
├── NonOverlap/
│   └── simulation_1/
│       ├── revenue/
│       │   ├── UAVs2_GRID5_ModeGG_Sequential.xlsx
│       │   └── UAVs2_GRID5_ModeGR_Sequential.xlsx
│       ├── sequences/
│       │   └── UAVs2_GRID5_1800_20_ModeGG_Sequential_sequences.xlsx
│       └── waypoints/
│           └── UAVs2_GRID5_waypoints.xlsx
├── Overlap/
│   └── simulation_1/
│       └── (same structure)
└── Summary/
    └── simulation_1/
        ├── execution_trace.xlsx
        └── framework_progress.png

Visualizations/
├── mean_revenue_vs_round_random_initial/
│   └── UAV3.png ... UAV10.png
├── mean_revenue_vs_round_kmeans_initial/
│   └── UAV3.png ... UAV10.png
├── final_revenue_vs_greedy_random_initial/
│   └── UAV3.png ... UAV10.png
├── final_revenue_vs_clusterga_kmeans_initial/
│   └── UAV3.png ... UAV10.png
├── per_uav_revenue_share/
│   ├── random_initial_best/UAV3.png ... UAV10.png
│   ├── greedy/UAV3.png ... UAV10.png
│   ├── kmeans_initial_best/UAV3.png ... UAV10.png
│   └── cluster_ga/UAV3.png ... UAV10.png
├── flighttime_left_vs_greedy_random_initial/
│   └── UAV3.png ... UAV10.png
├── flighttime_left_vs_clusterga_kmeans_initial/
│   └── UAV3.png ... UAV10.png
└── UAV{n}_{random_initial|kmeans_initial}/
    ├── per_round_plots/{code}/...   (only if RUN_PER_ROUND_MEAN_STD_PLOTS)
    └── gifs/{code}/...              (only if RUN_GIF_GENERATION)
```

No date subfolders appear anywhere in this tree - `simulation_N` numbers per session, and every other path is either fixed by the file's purpose or by the UAV fleet size / strategy code it belongs to.

### Step 6: Inspect Key Outputs

#### **Revenue Workbook** (`UAVs2_GRID5_ModeGG_Sequential.xlsx`)

- **Sheets**: `SimRun1`, `SimRun2`, ..., `SimRun5`
- **Columns**: `negotiation_round`, `UAV0`, `UAV1`, ...
- **Values**: Revenue rate per UAV per negotiation round

#### **Sequences Workbook** (`*_sequences.xlsx`)

- **Columns**: `negotiation_round`, `UAV0`, `m_0`, `UAV1`, `m_1`, ...
- **`UAVk`**: Waypoint tour (hyphen-joined, e.g. `"3-7-12"`)
- **`m_k`**: Feasible loop count for that tour (how many times the tour can repeat within `max_flight_time`)

#### **Waypoints Workbook** (`UAVs2_GRID5_waypoints.xlsx`)

- **Columns**: `Waypoint`, `Revenue`, `X`, `Y`, `Cluster`
- **Rows**: One per waypoint (grid positions, revenues, and which UAV it was initially assigned to)

#### **Plots** (in `Visualizations/`)

- **`mean_revenue_vs_round_*`**: all 8 strategies as lines, one graph per UAV fleet size
- **`final_revenue_vs_*` / `flighttime_left_vs_*`**: boxplots comparing all 8 strategies plus the relevant external baseline
- **`per_uav_revenue_share/`**: per-UAV workload distribution for the tournament-selected best strategy and each baseline

### Common Issues & Fixes

#### **Issue 1: `FileNotFoundError: settings.yaml`**

**Fix:** Ensure `settings.yaml` is in the same directory as the Python scripts.

#### **Issue 2: `ModuleNotFoundError: No module named 'yaml'`**

**Fix:** Install missing dependencies:

```
pip install pyyaml openpyxl
```

#### **Issue 3: Preflight Check Fails**

**Symptom:** `Preflight failed; skipping {mode} for SimRun {N}.`
**Fix:** Increase `max_flight_time` or decrease `grid.width`/`grid.height` in `settings.yaml`:

```
uav:
  max_flight_time: 3600  # Increase to 60 minutes
```

#### **Issue 4: IRADA Can't Find Waypoints**

*(Not verified against the current codebase - left as originally documented.)*

**Symptom:** `FileNotFoundError: Expected NonOverlap waypoint folder does NOT exist`
**Fix:** Run `Games.py` first to generate Non-Overlap waypoint files, then run `IRADA.py`.

#### **Issue 5: No Plots Generated**

**Fix:** Check the `CONFIGURATION` block at the top of `Analysis.py` - the 7 required comparison graphs always generate; the two auxiliary outputs need their own toggles set to `True`:

```python
RUN_PER_ROUND_MEAN_STD_PLOTS = True   # per-round mean+/-std shading
RUN_GIF_GENERATION = True             # animated negotiation replay
```

### Quick Test Run (30 seconds)

For a minimal test to verify everything works:

```yaml
# In settings.yaml, set:
uav:
  num_uavs: 2
grid:
  width: 3
  height: 3
simulation:
  n_runs: 2
algorithms:
  sequential_GG: true
  sequential_GR: false
  sequential_RG: false
  sequential_RR: false
  random_GG:     false
  random_GR:     false
  random_RG:     false
  random_RR:     false
```

Then run:

```
python Games.py
```

You should see a handful of runs complete in a few seconds, with Excel files under `Results/`.

### Next Steps

After verifying the basic setup:

1. **Scale Up**: Increase `num_uavs` to 3-5, `grid.width`/`grid.height` to 5-10
2. **Enable More Algorithms**: Turn on the Random-order strategies in `settings.yaml`
3. **Run Batch Simulations**: Use `Simulate.sh` (see next section)
4. **Explore Parameter Sensitivity**: Vary `speed`, `max_flight_time`, `zero_prob`

### Batch Execution with `Simulate.sh`

For running multiple parameter sweeps:

```
#!/bin/bash
# Example: Test different UAV counts and grid sizes

for uavs in 2 3 5; do
  for grid in 5 10 15; do
    python Games.py --num_uavs $uavs --grid_width $grid --grid_height $grid --n_runs 10
  done
done
```

**Usage:**

```
chmod +x Simulate.sh
./Simulate.sh
```

`Games.py` accepts `--num_uavs`, `--grid_width`, `--grid_height`, `--grid_spacing`, `--speed`, `--max_flight_time`, and `--n_runs` as CLI overrides, applied on top of whatever `settings.yaml` already has. `Analysis.py` currently takes no CLI arguments at all - its settings are the `CONFIGURATION` block at the top of the file, so a sweep script would need to edit that block directly (or run it once per sweep configuration by hand) rather than passing flags to it.

---

## 2. Configuration (`settings.yaml`)

| Field | Type | Default | Description |
|---|---|---|---|
| `project.results_dir` | `str` | `Results` | Base directory for Games.py's own output |
| `project.grids_dir` | `str` | `Grids` | Pregenerated waypoint grids (for `cluster_ga`/`kmeans_centroids`/`generic` initial-assignment sources) |
| `project.benchmark_dir` | `str` | `Cluster_sequences` | K-means and K-means+GA baseline files |
| `project.custom_benchmark_dir` | `str` | `Generic_initial_assignment` | Your own bring-your-own-baseline files (see `make_generic_assignment_template.py`) |
| `simulation.seed` | `int` or `null` | `null` | Random seed; `null` for unseeded runs |
| `simulation.n_runs` | `int` | 100 | Number of independent SimRuns |
| `simulation.enable_logging` | `bool` | `false` | Write per-mode negotiation logs to disk |
| `simulation.run_modes` | `list[str]` | `["NonOverlap", "Overlap"]` | Which mode(s) run this session |
| `simulation.initial_assignment_source` | `str` | `random` | `random` \| `cluster_ga` \| `kmeans_centroids` \| `generic` |
| `grid.width`, `grid.height` | `int` | 13 | Grid is `width × height` (excluding the depot) |
| `grid.spacing` | `float` | 92.608 | Physical distance (m) between adjacent grid points |
| `grid.zero_prob` | `float [0-1]` | 0.2 | Probability of a waypoint's revenue being zero |
| `uav.num_uavs` | `int` | 5 | Number of UAV agents |
| `uav.speed` | `float` | 16 | UAV speed (units consistent with spacing/time) |
| `uav.max_flight_time` | `float` | 1920 | Flight-time budget used by the 2-opt solver and preflight check |
| `revenue.random` | `bool` | `true` | Draw random revenue, vs. using a fixed value for every non-zero waypoint |
| `revenue.fixed_value` | `float` | 30 | Used only when `revenue.random: false` |
| `revenue.min`, `revenue.max` | `float` | 60, 600 | Uniform draw bounds when `revenue.random: true` |
| `overlap.clone_threshold` | `float` | 300 | Revenue above which a waypoint gets cloned in Overlap mode |
| `overlap.clone_assignment` | `str` | `random` | `same` \| `random` \| `balanced` |
| `algorithms.*` | `bool` | `true` | Eight toggles (`sequential_GG` ... `random_RR`) - any toggle omitted defaults to enabled |

Overrides can be passed via CLI (Games.py only - see the Batch Execution section above for exactly which flags are supported):

```
--num_uavs 12 --grid_width 15 --max_flight_time 2000
```

### Algorithm Naming Convention

Each algorithm is identified by three components:

| **Component** | **Options** | **Meaning** |
|---|---|---|
| **Mode** | GG, GR, RG, RR | Drop-Pick strategy |
| | GG = Greedy Drop, Greedy Pick | UAVs drop lowest-revenue waypoint, pick highest-revenue waypoint |
| | GR = Greedy Drop, Random Pick | UAVs drop lowest-revenue waypoint, pick random waypoint |
| | RG = Random Drop, Greedy Pick | UAVs drop random waypoint, pick highest-revenue waypoint |
| | RR = Random Drop, Random Pick | UAVs drop random waypoint, pick random waypoint |
| **Order** | Sequential, Random | Agent turn order |
| | Sequential | UAVs negotiate in fixed order (UAV0 → UAV1 → ...) |
| | Random | UAVs negotiate in shuffled order each round |
| **Game** | NonOverlap, Overlap | Waypoint ownership model |
| | NonOverlap | Each waypoint assigned to exactly one UAV |
| | Overlap | High-value waypoints can be "cloned" for multiple UAVs |

**Short Labels (used in plots):** `[N/O][S/R][G/R][G/R]` - e.g. `NSGG` = NonOverlap, Sequential, Greedy-Greedy; `ORGR` = Overlap, Random, Greedy-Random. If IRADA benchmarking is active in your workflow, its results are typically labeled `IRADA` rather than fit into this 4-letter scheme.

---

## **Project Structure**

```
├── Games.py                          # Main simulation (Non-Overlap & Overlap games)
├── IRADA.py                          # IRADA benchmark allocator (not covered by this documentation pass)
├── Analysis.py                       # Post-processing and visualization
├── make_generic_assignment_template.py  # Generates blank bring-your-own-baseline templates
├── settings.yaml                     # Games.py configuration
├── Simulate.sh                       # Batch execution script
├── Results/                          # Games.py output
│   ├── NonOverlap/simulation_N/{revenue,sequences,waypoints}/
│   ├── Overlap/simulation_N/{revenue,sequences,waypoints}/
│   └── Summary/simulation_N/{execution_trace.xlsx,framework_progress.png}
├── Grids/NonOverlap/                 # Pregenerated waypoint grids
├── Cluster_sequences/                # K-means / K-means+GA baseline files (or wherever benchmark_dir points)
│   └── K-means/
├── Generic_initial_assignment/       # Your own baseline files (or wherever custom_benchmark_dir points)
├── Old-sequences/, Old-revenues/     # Random-initial baseline data, read by Analysis.py
├── Greedy_sequences/, Greedy_revenues/     # External Greedy baseline, read by Analysis.py
├── Cluster_revenues/                 # External K-means+GA baseline revenue, read by Analysis.py
├── BenchmarkingIRADA/                # IRADA outputs (not covered by this documentation pass)
└── Visualizations/                   # Analysis.py output
```

---

## File 1: Games.py

### Purpose

Runs the Non-Overlapping and Overlapping potential-game simulations: UAVs negotiate over waypoints across multiple rounds (drop/pick/reassign) until revenue rate stabilizes, and every round of every negotiation is written to Excel.

### Configuration (`settings.yaml`)

All simulation parameters live in `settings.yaml`, loaded once at startup:

```yaml
project:
  results_dir: Results
  grids_dir: Grids                       # pregenerated waypoint grids (see below)
  benchmark_dir: Cluster_sequences       # K-means/K-means+GA baseline files
  custom_benchmark_dir: Generic_initial_assignment  # your own baseline files

simulation:
  seed: null                             # or an integer, for reproducible runs
  n_runs: 100
  enable_logging: false
  run_modes: ["NonOverlap", "Overlap"]   # run one mode, or both
  initial_assignment_source: random      # random | cluster_ga | kmeans_centroids | generic

grid:
  width: 13
  height: 13
  spacing: 92.608
  zero_prob: 0.2

uav:
  num_uavs: 3
  speed: 16
  max_flight_time: 1920

revenue:
  random: true
  fixed_value: 30
  min: 60
  max: 600

overlap:
  clone_threshold: 300
  clone_assignment: "random"             # "same" | "random" | "balanced"

algorithms:
  # Which of the 8 market strategies actually run this session. Any
  # toggle you omit defaults to enabled.
  sequential_GG: true
  sequential_GR: true
  sequential_RG: true
  sequential_RR: true
  random_GG:     true
  random_GR:     true
  random_RG:     true
  random_RR:     true
```

Every simulation parameter lives in this one file - there's no separate CLI-flags-only configuration path to keep in sync.

### Initial assignment sources

Every SimRun's negotiation starts from an initial UAV split. Four sources are available, chosen via `simulation.initial_assignment_source`:

| Source | Needs external files? | What it is |
|---|---|---|
| `random` | No | A fresh grid is generated and split via uniform random round-robin - the default, self-contained mode. |
| `cluster_ga` | Yes, `benchmark_dir` | Reads a K-means+GA-optimized starting split from `{benchmark_dir}/UAVs{N}_GRID{W}_{max_flight_time}_{speed}_cluster_ga_sequences.xlsx` (round 0 of that file only). |
| `kmeans_centroids` | Yes, `benchmark_dir` | Reads a raw (not GA-optimized) K-means cluster membership from `{benchmark_dir}/K-means/centroids_UAV_{N}.xlsx` - columns `Cluster_k`, `Centroid_X`, `Centroid_Y`, `Assigned_Waypoints` (a Python-list-formatted string). `Cluster_k` maps directly to UAV index. |
| `generic` | Yes, `custom_benchmark_dir` | The simplest format to produce yourself: `{custom_benchmark_dir}/UAVs{N}_GRID{W}_initial_assignment.xlsx`, one sheet per SimRun, columns `UAV0`...`UAV{N-1}` with hyphen-joined waypoint IDs (e.g. `"3-17-42-91"`). |

For all three external sources, Games.py only ever needs to know *which waypoints each UAV starts with* - never `m_j`, revenue, or tour order. Its own 2-opt solver establishes the tour from scratch and derives `m_j`/revenue during negotiation, regardless of source. This means bringing your own comparison baseline (a different clustering method, a greedy heuristic, or anything else) only requires producing the `generic` format above - run `python make_generic_assignment_template.py` with no arguments to generate a blank, correctly-shaped template for every UAV count, with a placeholder in every cell showing exactly what to fill in.

Sheet names inside these files are matched by the exact string `SimRun{n}` first; if that's not found, the *n*-th sheet in the workbook is used instead. This means files from different contributors work regardless of their exact sheet-naming convention or casing - the only requirement is that sheets are kept in SimRun order.

### Running both modes, or just one

`simulation.run_modes` controls whether NonOverlap, Overlap, or both run in a given session. Preflight feasibility (does the full waypoint tour fit within `max_flight_time`) is checked once per enabled mode, not once per SimRun and not shared between modes - Overlap always has at least as many waypoints as NonOverlap (clones add entries on top of the base grid), so its feasibility genuinely can differ and gets its own check.

### Parallelization

Every (SimRun × mode × strategy) combination is an independent unit of work. Rather than working through them one at a time, or in small batches, the entire session's combos - across every SimRun, both modes, and all enabled strategies - are built up front and submitted to a single worker pool in one batch, so every CPU core stays continuously busy from start to finish instead of idling between smaller batches. The number of workers defaults to the machine's detected CPU core count.

While running, a live terminal dashboard shows overall progress, an ETA (based on each combo's own measured duration, not just elapsed wall-clock time since the batch started), and which specific combos are currently in flight. Results are buffered in memory per combo and flushed to their Excel files periodically, rather than reopening and rewriting a growing workbook after every single completed combo - this is what keeps write overhead flat rather than growing as a session progresses.

### Output structure

```
Results/
  NonOverlap/
    simulation_1/
      revenue/UAVs{N}_GRID{W}_Mode{XX}_{Order}.xlsx
      sequences/UAVs{N}_GRID{W}_{max_flight_time}_{speed}_Mode{XX}_{Order}_sequences.xlsx
      waypoints/UAVs{N}_GRID{W}_waypoints.xlsx
  Overlap/
    simulation_1/
      (same structure)
  Summary/
    simulation_1/
      execution_trace.xlsx      # per-combo timing/round-count detail
      framework_progress.png    # per-framework completion/speed chart
```

`simulation_N` numbers independently per session (not per calendar date) - each full run of `Games.py` gets the next available number, found by counting existing `simulation_*` folders under whichever mode(s) are actually enabled that session.

- **Revenue workbook**: sheets `SimRun1`...`SimRunN`, columns `negotiation_round`, `UAV0`, `UAV1`, ... - revenue rate achieved per UAV per round.
- **Sequences workbook**: same sheet structure, columns `negotiation_round`, `UAV0`, `m_0`, `UAV1`, `m_1`, ... - `UAVk` is a hyphen-joined waypoint tour (e.g. `"3-17-42"`), `m_k` is the feasible loop count for that tour.
- **Waypoints workbook**: columns `Waypoint`, `Revenue`, `X`, `Y`, `Cluster` (which UAV each waypoint was initially assigned to, for whichever `initial_assignment_source` was active that session).

### Algorithm naming convention

Every strategy is a 4-letter code: `[N/O][S/R][G/R][G/R]`.

- **N/O** - NonOverlap or Overlap (waypoint ownership model: exclusive vs. clonable at high-value locations)
- **S/R** - Sequential or Random agent turn order within a negotiation round
- **G/R G/R** - drop mode, then pick mode (Greedy = lowest-revenue drop / highest-revenue pick; Random = uniformly random choice)

E.g. `NSGG` = NonOverlap, Sequential turn order, Greedy drop, Greedy pick. `ORRG` = Overlap, Random turn order, Random drop, Greedy pick.

### Core classes

- **`Config`** - loads and validates `settings.yaml`, applies any CLI overrides.
- **`WaypointManager`** - grid generation (or loading a pregenerated one), revenue draws, Overlap-mode clone creation.
- **`PathOptimizer`** - 2-opt tour solver, and the `m_j` (feasible loop count) math shared by every part of the codebase that needs it.
- **`UAVAgent`** - one UAV's current tour and its drop/pick negotiation moves.
- **`InitialAssigner`** - uniform random round-robin starting split, used for the `random` initial-assignment source; the other three sources load an externally-provided split instead (see above), never computing one themselves.
- **`TaskAllocator` / `NegotiationAllocator`** - the multi-round negotiation loop: drop phase, pick phase, pool reassignment, rollback-on-regression, convergence detection.
- **`PreflightChecker`** - one feasibility check per enabled mode per session: does the full waypoint tour fit within `max_flight_time`.
- **`SimulationRunner`** - orchestrates the whole session: builds every SimRun's base grid up front, dispatches all combos across the worker pool, and writes output incrementally as combos complete.

---

## **File 2: IRADA.py**

*(Not verified against the current codebase as part of this documentation pass - left as originally documented, and worth re-auditing against the actual current `IRADA.py` before trusting it.)*

### **Purpose**

Implements the **IRADA** (Iterative Resource Allocation with Dynamic Adjustment) benchmark allocator using chronological event-driven scheduling.

### **Core Functions**

#### **1. IRADAAllocator Class**

```
class IRADAAllocator(TaskAllocator):
    def __init__(self, manager, config, log, max_rounds=1000)
    def allocate(self, initial_paths) -> (List[List[float]], List[List[List[int]]])
```

- **`__init__`**: Initializes with max rounds and `κ` (kappa) coefficient.
- **`allocate`**: **Event-driven simulation**:
  1. Initializes each UAV with an "ownership" set (initial waypoints).
  2. Computes first waypoint picks using restricted pool (ownership).
  3. Uses a priority queue (heap) to process events chronologically: `(arrival_time, uav_id, waypoint)`.
  4. When a UAV arrives at a waypoint, it:
    * Records the trip segment.
    * Selects the next waypoint (or depot) using `select_next_target_IRADA`.
    * Schedules the next arrival event.
  5. When a UAV returns to depot, it closes a "round" (depot→trip→depot) and computes revenue rate.
  6. Stops when all UAVs complete `max_rounds` depot returns.
  7. Logs communication timestamps (`last_comm`) between UAVs.

#### **2. IRADA Score Functions**

```
def compute_phi(agent, poi_idx, t, all_agents) -> float
def compute_epsilon(agent, poi_idx, t) -> float
def compute_eta(agent, poi_idx, t, all_agents) -> float
def select_next_target_IRADA(agent, t, all_agents, include_depot=True, restrict_pool=None) -> int
```

- **`compute_phi (φ)`**: **Information value coefficient**: `φᵢ(t) = Î(i,t)` (estimated revenue/information at waypoint `i` at time `t`).
- **`compute_epsilon (ε)`**: **Feasibility coefficient**: `εᵢ,ᵥ(t) = exp(-γ · min(0, Rᵢ,ᵥ(t)))`, where `Rᵢ,ᵥ(t) = C_remain(t) - dist(qᵥ, pᵢ) - dist(pᵢ, depot) - C_margin`. Penalizes waypoints that violate flight time constraints.
- **`compute_eta (η)`**: **Communication coefficient**: `ηᵢ,ᵥ(t) = Π_{u≠v, i∈ownership_u} [1 - exp(-λ(t - t_comm(v,u)))] · exp(-||pᵢ - c_u(t)||² / ||c_u(t) - c_v(t)||²)`. Encourages coordination: penalizes selecting waypoints owned by recently communicated UAVs and far from the agent's weighted center.
- **`select_next_target_IRADA`**: Computes `score = φ · ε · η` for all waypoints (or depot) and selects the highest. If `restrict_pool` is provided (first pick), only considers those waypoints.

#### **3. ChronoSimulationRunner Class**

```
class ChronoSimulationRunner:
    def __init__(self, cfg, log)
    def run(self)
    def prepare_output_dirs(self) -> (str, str)
    def dump_excel_data(self, rev_data, path_data, rev_dir, path_dir)
```

- **`__init__`**: Initializes IRADA-specific runner.
- **`run`**: **Main IRADA execution**:
  1. Loads Non-Overlap waypoint file (using `find_latest_waypoints_results_root`) to ensure IRADA uses the same grid as Non-Overlap.
  2. Runs `IRADAAllocator.allocate()` for `n_runs` times.
  3. Collects per-UAV revenue rates and trip sequences.
  4. Writes outputs to `BenchmarkingIRADA/revenue/` and `BenchmarkingIRADA/sequences/`.
- **`prepare_output_dirs`**: Creates `simulation_N` folders under `BenchmarkingIRADA/`.

---

## File 3: Analysis.py

### Purpose

Post-processes Games.py's Excel output into the comparison graphs a thesis (or any benchmarking write-up) needs: revenue rate progression, final-total comparisons against external baselines, per-UAV workload distribution, and flight-time-budget utilization - across every UAV fleet size in one run.

### Configuration

Every setting - paths, simulation parameters, plot styling, feature toggles, worker count - lives as plain variables in a single `CONFIGURATION` block at the top of the file, not in a separate settings file. This is deliberate: everything a future user might want to change (which folders to read from, font size for a publication figure, whether to enable the slower auxiliary outputs) is visible and editable in one place, without hunting through the rest of the file.

```python
SPEED = 16
MAX_FLIGHT_TIME = 1920
GRID_WIDTH = 13
UAV_COUNTS = list(range(3, 11))    # 3..10

VISUALIZATIONS_ROOT = HERE / "Visualizations"
OLD_SEQUENCES_ROOT = HERE / "Old-sequences" / "NonOverlap"    # random-initial baseline
OLD_REVENUES_ROOT = HERE / "Old-revenues" / "NonOverlap"
GRIDS_ROOT = HERE / "Grids" / "NonOverlap"
RESULTS_ROOT = HERE / "Results"                                # Games.py's own output
GREEDY_SEQUENCES_ROOT = HERE / "Greedy_sequences"              # external baselines
GREEDY_REVENUES_ROOT = HERE / "Greedy_revenues"
CLUSTER_SEQUENCES_ROOT = HERE / "Cluster_sequences"
CLUSTER_REVENUES_ROOT = HERE / "Cluster_revenues"

FONT_FAMILY = "Times New Roman"
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 18
DPI = 300
# ... plus box/line color, grid style, and figure-sizing constants

RUN_PER_ROUND_MEAN_STD_PLOTS = False   # off by default - see Auxiliary outputs below
RUN_GIF_GENERATION = False

MAX_WORKERS = os.cpu_count()           # see Parallelization below
```

Run it with no arguments: `python Analysis.py`.

### Data sources it compares

For each UAV fleet size, four sources get loaded and compared against each other:

1. **Random-initial** - this framework's own negotiation results, starting from a uniform random split.
2. **K-means-initial** - this framework's own negotiation results, starting from an externally-provided K-means split (Games.py output under `Results/`).
3. **Greedy** - an external baseline algorithm's result.
4. **Cluster+GA** - an external K-means+GA baseline's result.

Revenue rate is always read directly from whichever Excel file reports it, never recomputed - the one exception is flight-time-left, which isn't a column in any file and gets derived from each source's own reported tour and `m_j` (also always trusted as given, never re-solved).

### The required outputs

Eight graphs each, one per UAV fleet size (3 through 10), written under `Visualizations/`:

| Folder | Content |
|---|---|
| `mean_revenue_vs_round_random_initial/` | All 8 market strategies compared as lines, mean total revenue rate per negotiation round |
| `mean_revenue_vs_round_kmeans_initial/` | Same, for K-means-initial |
| `final_revenue_vs_greedy_random_initial/` | Boxplot: all 8 strategies + Greedy, final-round total revenue rate |
| `final_revenue_vs_clusterga_kmeans_initial/` | Boxplot: all 8 strategies + Cluster+GA |
| `per_uav_revenue_share/` | Per-UAV % share of the total, for the best-performing strategy (by tournament, see below) plus each external baseline |
| `flighttime_left_vs_greedy_random_initial/` | Boxplot: all 8 strategies + Greedy, flight-time remaining at final round |
| `flighttime_left_vs_clusterga_kmeans_initial/` | Boxplot: all 8 strategies + Cluster+GA |

"Best-performing strategy" is decided by tournament: for each SimRun independently, whichever strategy has the highest total revenue rate wins that SimRun; whichever strategy accumulates the most wins across all SimRuns is "best" for that fleet size and scenario.

### Auxiliary outputs

Two additional views, off by default (`RUN_PER_ROUND_MEAN_STD_PLOTS`, `RUN_GIF_GENERATION`), since they're slower and not needed for every run:

- **Per-round mean±std plots** - variance across SimRuns at each negotiation round, per UAV and per total, for one strategy at a time. Written to `Visualizations/UAV{n}_{random_initial|kmeans_initial}/per_round_plots/{code}/`.
- **Negotiation GIFs** - animated replay of each UAV's tour changing round by round, with a sidebar showing `m_j`/revenue per UAV and the running total. Written to `Visualizations/UAV{n}_{random_initial|kmeans_initial}/gifs/{code}/`.

### Parallelization

Loading data for one UAV fleet size means reading hundreds of small Excel files (every strategy, every SimRun, across all four sources) - independent work from every other fleet size's loading. This is dispatched across `MAX_WORKERS` worker processes (default: all detected CPU cores), one fleet size per worker at a time, while the plotting itself - fast and entirely in-memory - stays on the main process afterward. Set `MAX_WORKERS = 1` to run everything sequentially instead.

### Output structure

```
Visualizations/
  mean_revenue_vs_round_random_initial/UAV3.png ... UAV10.png
  mean_revenue_vs_round_kmeans_initial/UAV3.png ... UAV10.png
  final_revenue_vs_greedy_random_initial/UAV3.png ... UAV10.png
  final_revenue_vs_clusterga_kmeans_initial/UAV3.png ... UAV10.png
  per_uav_revenue_share/
    random_initial_best/UAV3.png ... UAV10.png
    greedy/UAV3.png ... UAV10.png
    kmeans_initial_best/UAV3.png ... UAV10.png
    cluster_ga/UAV3.png ... UAV10.png
  flighttime_left_vs_greedy_random_initial/UAV3.png ... UAV10.png
  flighttime_left_vs_clusterga_kmeans_initial/UAV3.png ... UAV10.png
  UAV{n}_random_initial/{per_round_plots,gifs}/{code}/...    # only if enabled
  UAV{n}_kmeans_initial/{per_round_plots,gifs}/{code}/...    # only if enabled
```

### Core structure

- **`DataLayer`** - all loading and math, no plotting dependency. Every source-specific loader (`load_random_initial`, `load_kmeans_initial`, `load_greedy`, `load_cluster_ga`) returns the same shape: per-strategy, per-SimRun records of final revenue rates, per-UAV flight-time-left, and the SimRun's total.
- **`Plots`** - the three chart types (line, multi-box comparison, per-UAV share), reading all styling from the `CONFIGURATION` block. Figure width and axis limits scale with how much content is actually being plotted, and x-axis labels rotate automatically only when they'd genuinely overlap at the configured font size (measured from the actual rendered text, not a guessed threshold) - a 2-box comparison and a 9-box one, or a small font and a large one, both render legibly without separate handling.
- **`run()`** - orchestrates: loads all four sources per fleet size in parallel, runs the tournament, then builds every graph.

---

## 📂 Outputs

After each run:

```
Results/
  NonOverlap/simulation_N/{revenue,sequences,waypoints}/
  Overlap/simulation_N/{revenue,sequences,waypoints}/
  Summary/simulation_N/{execution_trace.xlsx,framework_progress.png}
BenchmarkingIRADA/
  revenue/simulation_N/
  sequences/simulation_N/
Visualizations/
  (see the Analysis.py section above for the full breakdown)
```

- **Revenue Excel** → per-algorithm totals (per round).
- **Sequences Excel** → UAV tours and feasible loop counts.
- **Waypoints Excel** → grid coordinates, revenues, and initial UAV assignment.
- **Plots** → revenue-over-rounds, final-total boxplots, per-UAV share, flight-time-left, and (optionally) per-round variance plots and GIFs.

## 📈 Interpreting Results

- **Revenue plots**: compare convergence of total revenue across algorithms.
- **Boxplots**: distribution of results across runs (total revenue, UAV contributions, flight-time left).
- **External baselines**: Greedy and K-means+GA are the current comparison points for the NonOverlap benchmarking workflow; IRADA benchmarking, if used, sits alongside these separately.
- **GIFs**: optional animated UAV routes (`RUN_GIF_GENERATION = True` in `Analysis.py`'s `CONFIGURATION` block).

---

## 🔧 Extending the Framework

- Add new allocators (e.g., CNP, CBBA, TS-DTA) by subclassing `TaskAllocator`.
- Toggle new market strategies in `settings.yaml`'s `algorithms:` section, and add the corresponding entry to `SimulationRunner._define_strategies()`.
- New Excel outputs following the existing column conventions are auto-picked up by `Analysis.py`'s `generic` initial-assignment source, or by extending `DataLayer` with a new loader for a dedicated source.

*Work in progress - future improvements:* advanced multi-objective metrics, dynamic deadlines, real-world maps.

### Performance Tuning Tips

#### **For Large Grids (>10×10):**

- Increase `max_flight_time` to avoid preflight failures
- Enable only 2-3 algorithms initially (disable Random-order modes)
- Reduce `n_runs` to 5 for faster iteration

#### **For Many UAVs (>5):**

- Expect longer negotiation times (more rounds to reach convergence)
- Use `enable_logging: false` to speed up execution
- Watch for repeated rollback messages in the log (indicates cyclic states)

#### **For Statistical Significance:**

- Use `n_runs ≥ 30` for publication-ready results
- Set a fixed `seed` for reproducibility across experiments

#### **For Faster Analysis.py Runs:**

- `MAX_WORKERS` in the `CONFIGURATION` block controls how many UAV fleet sizes load in parallel - defaults to all detected CPU cores; lower it if running alongside other heavy processes, or set it to `1` to disable parallelism entirely for easier debugging
- Leave `RUN_PER_ROUND_MEAN_STD_PLOTS` / `RUN_GIF_GENERATION` off unless you specifically need them - both are slower and produce many extra files

---

## **Frequently Asked Questions (FAQ)**

**Q1: Why does Overlap sometimes perform worse than NonOverlap?**
A: Clones add waypoints but don't increase total revenue. If `clone_threshold` is too low, UAVs waste time revisiting the same high-value locations instead of covering more area.

**Q2: Can I run only IRADA without MPG?**
*(Not verified against the current codebase.)* A: No. IRADA requires a NonOverlap waypoint file to ensure fair comparison on the same grid. Run `Games.py` first, then `IRADA.py`.

**Q3: What's the difference between `sequences.xlsx` and `revenue.xlsx`?**
A:
- `sequences.xlsx`: Lists which waypoints each UAV visits per round, plus the feasible loop count `m_k`
- `revenue.xlsx`: Shows the revenue *rate* (revenue/time) achieved per round

**Q4: How do I reproduce results exactly?**
A: Use the same `seed`, `grid` parameters, `n_runs`, and `initial_assignment_source` (with matching baseline files, if not `random`). Seed ensures identical random revenue draws and, for `random` initial assignment, identical starting splits.

**Q5: Can I visualize UAV paths on a map?**
A: Not built-in for real-world maps. Export waypoint coordinates from `waypoints.xlsx` and plot using `matplotlib.pyplot.scatter()` or GIS tools - or enable `RUN_GIF_GENERATION` in `Analysis.py` for an animated replay of the grid-space negotiation itself.

**Q6: Where do I get baseline files for `cluster_ga`, `kmeans_centroids`, or an external comparison algorithm?**
A: `cluster_ga` and `kmeans_centroids` expect files matching the format described in the Games.py section above, under `benchmark_dir`. For your own algorithm, run `python make_generic_assignment_template.py` with no arguments - it generates a blank, correctly-shaped template for every UAV fleet size, with a placeholder in every cell showing exactly what to fill in.

---

### Troubleshooting Convergence Issues

#### **Symptom: Negotiation doesn't converge, or takes many rounds**

**Causes:**
- Grid too large relative to `max_flight_time`
- Too many zero-revenue waypoints (`zero_prob` too high)
- Rollback stasis (cyclic state repetition)

**Fix:** `max_rounds`, `patience`, and the rollback-repeat limit are not currently exposed in `settings.yaml` - they're fixed defaults in `NegotiationAllocator`'s constructor (`max_rounds=100, patience=5`). Adjusting them requires editing that constructor call directly, or passing different values if you're instantiating `NegotiationAllocator` yourself. The most effective yaml-level levers are `max_flight_time`, `grid.width`/`grid.height`, and `zero_prob`.

---

## **Known Limitations**

1. **2-opt TSP Heuristic**: Not guaranteed to find global optimum (used for speed over exactness)
2. **Static Revenue Model**: Waypoint values don't decay over time (future work: temporal dynamics)
3. **Homogeneous UAVs**: All UAVs have identical `speed` and `max_flight_time`
4. **Euclidean Distance**: Assumes flat terrain (no elevation or no-fly zones)
5. **Clone Threshold**: Fixed per simulation (future: adaptive cloning based on demand)

---

## License

MIT License

Copyright (c) 2025 Intemnets-Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
