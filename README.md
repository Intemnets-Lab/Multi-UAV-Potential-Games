# Multi-UAV-Potential-Games
A repository on autonomous systems and multi-agent coordination, applying potential game theory to wildfire monitoring. 

IN-PROGRESS

# Benchmarking of Multi‑UAV Flight sequence Calculation

## 📂 Code Structure

```
├─ MPG_algorithms.py   # Simulation runner with algorithms, Excel, log files
├─ IRADA_algorithm.py   # IRADA with algorithms, Excel, log files
├─ Analysis.py        # Post‑processing: plots, boxplots, animations from Excel
├─ output5x5_benchmark/
│   ├─ revenue/...
│   ├─ sequences/...
│   ├─ waypoints/...
│   ├─ gifs/, plots/, excels/
└─ README.md          # (this file)
```


# `MPG_algorithms.py` — Full Reference & Walkthrough

> **Scope:**
> This script drives a family of multi‐UAV “drop/pick” negotiations on a fixed grid of waypoints, treating the process as an Exact Potential Game over revenue rates. It generates per‐run Excel outputs for revenue and sequences.

---

## 1. High-Level Architecture

```
MPG_algorithms.py
├── imports & global helpers
├── class Config             # all simulation parameters & toggles
├── class Logger             # thin wrapper over Python’s logging
├── class WaypointManager    # grid creation & random revenue/deadline draws
├── class sequenceOptimizer      # static STSPSolver 2-opt routine
├── class PreflightChecker   # verifies feasibility of initial waypoints
├── class InitialAssigner    # uniform or custom seeding of initial tours
├── class TaskAllocator      # base for negotiation & market allocators
│   └── class NegotiationAllocator  # “ModeXY” drop/pick negotiation
│   └── (other allocators: CNPAllocator, CBBAAllocator, TSDTAAllocator…)
├── class SimulationRunner   # orchestrates preflight, experiments, outputs
└── if __name__ == '__main__' → instantiate + runner.run()
```

---

## 2. Configuration (`Settings.yaml`)

All parameters load from `settings.yaml` (or can be overridden via CLI flags in `Simulate.sh`):

| Attribute             | Type           | Default     | Description                                                          |
| --------------------- | -------------- | ----------- | -------------------------------------------------------------------- |
| `grid_width`          | `int`          | 13          | number of cells along each axis (grid is `grid_width×grid_height`)   |
| `grid_height`         | `int`          | 13          | number of cells along each axis (grid is `grid_width×grid_height`)   |
| `grid_spacing`        | `float`        | 92.608      | physical distance (m) between adjacent grid points                   |
| `zero_prob`           | `float [0–1]`  | 0.2         | probability of each waypoint’s revenue being zero                    |
| `random_revenue`      | `bool`         | False       | re-draw random revenue ∈\[`revenue_min`,`revenue_max`] every run?    |
| `fixed_revenue`       | `float`        | 50          | if `random_revenue=False`, all non-zero waypoints use this           |
| `revenue_min`, `_max` | `float`        | 60, 600     | when `random_revenue=True`, uniform draw bounds                      |
| `num_uavs`            | `int`          | 5           | number of UAV agents                                                 |
| `speed`               | `float`        | 16          | UAV speed (units consistent with spacing (m)/time (s))               |
| `max_flight_time`     | `float`        | 1920        | 2-opt solver’s maximum allowable tour time                           |
| `n_runs`              | `int`          | 10          | number of independent experiments (only when `run_experiments=True`) |
| **Per‐Mode toggles**  | `bool`         | see below   | eight toggles to pick exactly which modes run:                       |
|   `sequential_GG`     | bool           | True        | run ModeGG in sequential pass?                                       |
|   `sequential_GR`     | bool           | False       | run ModeGR in sequential pass?                                       |
|   `sequential_RG`     | bool           | False       | run ModeRG in sequential pass?                                       |
|   `sequential_RR`     | bool           | False       | run ModeRR in sequential pass?                                       |
|   `random_GG`         | bool           | False       | run ModeGG in random pass?                                           |
|   `random_GR`         | bool           | True        | run ModeGR in random pass?                                           |
|   `random_RG`         | bool           | False       | run ModeRG in random pass?                                           |
|   `random_RR`         | bool           | True        | run ModeRR in random pass?                                           |
| `enable_logging`      | `bool`         | True        | whether to write negotiation logs to disk                            |
| `results_dir`         | `str` / `sequence` | `"output/"` | base directory for all MPG log file, plots, excels, gifs         |
| `IRADA_benchmarking_dir`| `str` / `sequence`| `"output/"`| base directory for all IRADA log file, plots, excels, gifs       |

Overrides can be passed via CLI:

```bash
--num_uavs 12 --grid_width 15 --max_flight_time 2000 --sequential_GG true --random_RR false
````

---

## 3. Core Classes

### 3.1 WaypointManager

```python
class WaypointManager:
    def __init__(self, config: Config):
        # 1) builds a cemented grid of (x,y) coords once (class‐static)
        # 2) draws self.values & self.deadlines per config zero_prob & revenue settings
```

* **`shared_pool()`** returns all waypoint indices with revenue > 0.
* **`redraw_values_and_deadlines()`** allows fresh randomization between runs.

### 3.2 sequenceOptimizer

```python
class sequenceOptimizer:
    @staticmethod
    def STSPSolver(depot, points, speed, max_time):
        # runs 2-opt (swap) improvements on the TSP tour,
        # returns the best ordering and the final “cost” (aka mj metric).
```

The returned `mj` is defined in the `Paper`

### 3.3 NegotiationAllocator

```python
class NegotiationAllocator(TaskAllocator):
    def allocate(self, initial_sequences):
        # 1) Drop phase (in order or randomized per config.randomize_sequence)
        # 2) Pick phase (ditto)
        # 3) Re-assign any leftovers + 2-opt repairs
        # 4) Repeat for max_rounds or until convergence
```

* Logs each drop/pick, tracks the “market pool” of freed waypoints.
* Returns `(rates, history)` where:

  * `rates[r]` = total revenue rate after round `r`.
  * `history[r]` = list of UAV sequences after round `r`.

### 3.4 SimulationRunner

```python
class SimulationRunner:
    def __init__(self, cfg: Config, log: Logger):
        …
    def run(self):
        –– Preflight checker
        –– Initial waypoint plot
        – For each run_idx in 1…n_runs:
             • redraw values & deadlines
             • assign initial tours via InitialAssigner
             • apply 2-opt to initial tours
             • compute & log base revenue rates
             • determine passes = [(seq/random, [enabled strategy names]), …]
             • for each pass:
                 – set cfg.randomize_sequence
                 – for each enabled strategy name:
                     • alloc = strategies[name]()
                     • (rates, history) = alloc.allocate(...)
                     • collect per‐round revenue/sequence into DataFrames
        – After all runs:
             • call _prepare_output_dirs()
             • dump per‐algorithm/sequence combo to `.xlsx` (revenues & sequences)
```

**Key methods**:

* **`_prepare_output_dirs()`**
  Creates

  ```
  output_dir/
    revenue/YYYY-MM-DD/simulation_N/
    sequences/YYYY-MM-DD/simulation_N/
    waypoints/YYYY-MM-DD/simulation_N/
  ```

  with auto‐incremented `simulation_N`.

* **`_dump_excel_data(rev_data, sequence_data, rev_dir, sequence_dir)`**
  Writes one Excel workbook per algorithm+seq/rand, each with `sheet1 … sheetN` for runs.

---

Everything under `cfg` defaults to your last‐saved settings. To override:

---

## 5. Extending with New Allocators

To integrate, for example, **CBBA** or **CNP**:

1. Implement `class CNPAllocator(TaskAllocator)` with the same `allocate(...) → (rates,history)` signature.

2. In `SimulationRunner.run()`, add to the `strategies` dict:

   ```python
   strategies = {
     "ModeGG": lambda: NegotiationAllocator(...),
     …,
     "CNP":    lambda: CNPAllocator(self.manager, self.cfg, self.log),
     "CBBA":   lambda: CBBAAllocator(self.manager, self.cfg, self.log),
   }
   ```

3. Enable via toggles `sequential_CNP=True`, `random_CNP=True`, etc., and add `if name.startswith("CNP")…` logic analogous to `ModeXX`.

---

## 6. Output Structure

```
output_dir/
  revenue/2025-07-29/
    simulation_1/
      UAVs5_GRID10_Sequential_ModeGG.xlsx
      UAVs5_GRID10_Random_ModeGR.xlsx
      … (one per algorithm+mode)
  sequences/2025-07-29/
    simulation_1/
      UAVs5_GRID10_Sequential_ModeGG_sequences.xlsx
      … (UAV waypoint sequences)
  waypoints/2025-07-29/
    simulation_1/
      UAVs5_GRID10_waypoints.xlsx
```

* **Excel outputs**:

  * In `revenue/…`: per‐algorithm revenue‐per‐round (runs → sheets).
  * In `sequences/…`: per‐algorithm sequence(s)‐per‐round.
  * In `waypoints/…`: initial waypoint coords & revenues.

---
## 📈 Interpreting Results

* **Revenue Excel sheets**: rows = negotiation rounds, columns = UAV revenue rates; final round captures convergence.

## 7. Next Steps & Roadmap

* **Add new allocators**: CNP, CBBA, TSDTA, plus any custom heuristics.
* **Parameter sweeps**: vary `num_uavs`, `grid_spacing`, `zero_prob`, etc.
* **Comparison scripts**: statistical tests, bar charts, multi‐scenario dashboards.
* **Real‐world maps**: replace synthetic grid with GIS coordinates.

### IRADA Algorithm

Apart from the family of drop/pick negotiation strategies (ModeGG, ModeGR, ModeRG, ModeRR),  
`IRADA_algorithm.py` also runs **IRADA** (Iterative Revenue–Aware Drop/Add) as a benchmark.

* IRADA is implemented directly inside `IRADA_algorithm.py` and produces its own
  `…/Benchmarking/IRADA/revenue/…` and `…/Benchmarking/IRADA/sequences/…` outputs.
* Its Excel outputs follow the same per-run sheet convention as the other algorithms,
  so all downstream analysis functions can treat IRADA results uniformly.
* IRADA runs are controlled by the `irada` toggle in `settings.yaml` (or `--irada true` via CLI).


# `Analysis.py` — Full Reference & Walkthrough
Below is a complete, annotated walkthrough of **Analysis.py**, covering every flag, function, and code sequence “from top to bottom” so you can see exactly what it’s doing (and where to tweak it). Wherever relevant, I’ve cited the exact lines in the file.

---

## 1. Module‐level imports & flags

At the very top we pull in all the libraries we need, then define two Boolean switches that turn on/off different parts of the script:

```python
import os, shutil  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.animation as animation  
from sequencelib import sequence  
from matplotlib.lines import Line2D  
from matplotlib.animation import PillowWriter  
from typing import List  

# ─── TOGGLE SECTIONS ON/OFF ───
GraphGeneration: bool   = False  
GifGeneration: bool     = False  
```

These flags determine which of the following functions actually get defined and (in the `__main__` block) which get run .

---

## 2. Graph‐generation functions (`GraphGeneration=True`)

When `GraphGeneration` is `True`, three functions are defined:

### 2.1 `analyze_revenue_excels_with_shade(excel_dir)`

Reads every `.xlsx` in `excel_dir`, each sheet = one run:

1. **Pads** all runs to the same length (repeat‐last‐value).
2. **Stacks** them into a 3-D array.
3. Computes per‐round **mean** & **stddev** for each UAV *and* normalized total.
4. Makes one combined plot: each UAV’s mean line with shaded ±1 std, plus a bold dashed total‐line with its shading.
5. Saves to `excel_dir/analysis_plots/<basename>_mean_std.png`.

```python
def analyze_revenue_excels_with_shade(excel_dir):
    """
    For each .xlsx in excel_dir:
      • read all sheets → DataFrames
      • pad to same # of rows
      • compute mean±std per-UAV & TOT_norm
      • plot mean lines + shaded std bands
      • save under excel_dir/analysis_plots/
    """
    plots_dir = os.sequence.join(excel_dir, "analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    …
    plt.savefig(out_sequence)
```



---

### 2.2 `analyze_modes_and_club_tot(excel_dir)`

Aggregates *across* modes (e.g. “Random\_ModeGG”, “Sequential\_ModeRG”):

1. For each revenue‐Excel, pads & computes per‐round mean/std as above.
2. Saves a `<basename>_stats.xlsx` containing columns

   * `mean_UAVs_avg`, `std_UAVs_avg`
   * `mean_TOT_norm`, `std_TOT_norm`
3. Finally, plots all modes’ **mean** normalized‐totals on one comparison chart → `combined_TOT_norm_comparison.png`.

```python
def analyze_modes_and_club_tot(excel_dir):
    # read each *.xlsx → pad → compute stats → save *_stats.xlsx
    # then combine TOT_norm means across modes into one plot
```



---

### 2.3 `analyze_revenue_excels_graphs(excel_dir)`

Produces **four** sets of plots for *each* revenue‐Excel:

1. **Per‐UAV** mean±std shading plot.
2. **Total** (sum over UAVs) mean±std shading plot.
3. **Consolidated**: all UAV mean‐curves + mean(total)/n\_uavs curve.
   Saved under `excel_dir/analysis_plots/<basename>/…`.
4. **Boxplots** all Total Revenue Rates per Simulation Round per Algorithm.
   Saved under `excel_dir/boxplots/…`.
   

```python
def analyze_revenue_excels_graphs(excel_dir):
    """
    For each *.xlsx:
     1) One UAV mean±std plot
     2) One total mean±std plot
     3) One consolidated plot (UAV means + TOT/n_uavs)
    """
    plots_root = os.sequence.join(excel_dir, "analysis_plots")
    os.makedirs(plots_root, exist_ok=True)
    …
    plt.savefig(os.sequence.join(out_dir, "Consolidated_mean.png"))
```



---

### 2.4 `plot_consolidated_total_revenue(excel_dir, output_sequence=None)`

Reads *all* `.xlsx` in a folder, for each:

* Pads runs
* Computes mean *total* (sum of UAV columns) per round
* Stores in a dict `{basename: mean_series}`
  Finally plots every algorithm’s mean‐total curve on one chart:

```python
def plot_consolidated_total_revenue(excel_dir, output_sequence=None):
    # … compute consolidated[basename] = mean_tot_series
    plt.plot(rounds, mean_tot, label=algo)
    plt.savefig(output_sequence or "<excel_dir>/combined_total_revenue.png")
```



---

### 2.5 `organize_and_boxplot(sim_rev_dir: str)`

**Reorganizes** all `.xlsx` in `sim_rev_dir` into two subfolders:

   * `Sequential/`
   * `Random/`
     (based on the 3rd underscore‐token in the filename)
In each mode‐folder, reads each algorithm’s `.xlsx`, computes **final** total revenue per run (`sum of all UAV columns in last row`), collects them into a list, and makes **one** boxplot of “final totals per algorithm”:

   ```python
   fig, ax = plt.subplots(figsize=(1.5*len(labels), 5))
   ax.boxplot(all_data, tick_labels=labels, patch_artist=True)
   ax.set_title(f"{mode} Final Totals per Algorithm")
   fig.savefig(sim_rev_dir/boxplots/{mode}/final_totals_box.png)
   ```



---

## 3. GIF‐generation functions (`GifGeneration=True`)

When `GifGeneration` is `True`, we define:

### 3.1 `subscript(n: int) -> str`

Helper to turn digits `0–9` into Unicode subscripts (₀₁₂…₉) for nicely formatted labels.

### 3.2 `generate_gifs_from_sequences(sim_sequences_dir)`

For each `<base>_sequences.xlsx` under `sim_sequences_dir`:

1. Finds the matching revenue‐Excel and waypoint‐Excel in the parallel `…/revenue/…` and `…/waypoints/…` trees.
2. For each run‐sheet:

   * Reads per‐run waypoint coordinates **and revenues** (so point‐colors reflect that run’s zeros vs nonzeros).
   * Computes grid‐spacing `d` from the first two waypoints.
   * Builds a **scalable** figure size based on the grid extents + a sidebar for text.
   * Pre‐draws every waypoint as black (rev > 0) or gray (rev = 0), plus the depot at (0,0).
   * Defines an `update(frame)` that:

     * Clears only the *dynamic* artists (previous sequences + sidebar text).
     * Re‐draws each UAV’s tour in its assigned color.
     * If `m_j > 1`, draws a dashed “jump” line from last→first.
     * In the sidebar (using LaTeX–style `$m_{j}$`, `$Z_{j}$`), prints for UAV₀…UAVₙ:

       ```
       m₀=…, Z₀=…
       m₁=…, Z₁=…
       …
       Z=…      ← total rev
       ```
   * Animates and writes out a GIF via `PillowWriter` → saved in `…/gifs/<mode>/<base>_<run>.gif`.

```python
def generate_gifs_from_sequences(sim_sequences_dir):
    # … load seq Excel, rev Excel, waypoints Excel
    # … compute d, figsize, draw static grid & depot
    def update(frame):
        # remove previous dynamic layers
        # plot UAV sequences, jump lines if m>1
        # fig.text(...) sidebar with m_j, Z_j, total Z
    ani = FuncAnimation(fig, update, …)
    ani.save(..., writer=PillowWriter(fps=1))
```

---

## 4. `if __name__ == "__main__":`

Finally, depending on the three flags, the script will run:

```python
if __name__ == "__main__":
    if GraphGeneration:
        analyze_revenue_excels_graphs("…/revenue/...")  
        plot_consolidated_total_revenue("…/revenue/...")  
        print("Consolidated total revenue plot saved.")
        organize_and_boxplot("<absolute>/output…/simulation_X")  
    if GifGeneration:
        generate_gifs_from_sequences("…/sequences/.../simulation_X")  
        print("All plots saved!")
```

---

## 5. Specialized IRADA Analysis

`Analysis.py` produces **three additional boxplots** that incorporate IRADA side-by-side
with the other algorithms:

1. **Boxplot of percentage of flight-time remaining per tour**  
   (`boxplot_flight_time_left`)  
   Each UAV’s tour distance is divided by speed → used flight time.  
   Remaining time is normalized by `max_flight_time` → percentage left.  
   Plotted as a distribution across all runs and algorithms (incl. IRADA).

2. **Boxplot of total revenue rate including IRADA**  
   (`organize_and_boxplot`)  
   Collects final per-run totals for every algorithm, adds IRADA totals from
   `Benchmarking/IRADA/revenue`, and renders a combined boxplot.

3. **Per-UAV contribution boxplots**  
   For each algorithm (and IRADA), the percentage contribution of each UAV
   to the **total revenue rate** is computed and box-plotted.  
   This reveals fairness or skew — e.g., if UAV₀ dominates vs. balanced load.

All of these plots are saved under the relevant simulation folder, e.g.:

```
Results/revenue/YYYY-MM-DD/simulation_X/boxplots_with_irada/
├─ final_totals_all.png
├─ flight_time_left.png
├─ Random_ModeGG/uav_contribution.png
├─ Sequential_ModeGR/uav_contribution.png
└─ IRADA/uav_contribution.png
```


---

### Outputs

```
analysis_plots/
   combined_total_revenue.png
   boxplots_with_irada/
      final_totals_all.png         # revenue per algorithm (incl. IRADA)
      flight_time_left.png         # flight-time left %
      <Algo>/uav_contribution.png  # UAV contribution per algo
```

---

# 🔧 How to Run (with `Simulate.sh`)

1. Upgrade to **Python 3.10+** so you can keep the new style (for environments running python <3.10).

```bash
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev
```
2. (Suggested) Run `Simulate.sh` in a virtual environment. Sample virtual environment (with Python 3.10) creation as follows:

```bash
python3.10 -m venv venv/uav
source venv/uav/bin/activate
pip install -r requirements.txt
```

---

All runs are orchestrated through **Simulate.sh**.

## 🚀 Quick Start

```bash
chmod +x Simulate.sh
./Simulate.sh
```

This will run:

1. **MPG_algorithms.py** → prepares waypoints & settings.
2. **IRADA_algorithm.py** → runs all algorithms, outputs Excel.
3. **Analysis.py** → generates plots & boxplots.

Logs are stored under `run_logs/`.

---

## ⚙️ Overriding Parameters

Edit the `OVERRIDES` array inside `Simulate.sh`:

```bash
declare -a OVERRIDES=(
  "--num_uavs 10 --speed 12 --max_flight_time 1900 --n_runs 3"
  "--grid_width 5 --grid_height 5 --grid_spacing 7 --speed 3 --n_runs 3"
)
```

* Each string = one run.
* Flags correspond to `Config` entries or `settings.yaml`.

Examples:

```bash
# Run once with 12 UAVs, 15×15 grid, 2000 flight-time
"--num_uavs 12 --grid_width 15 --grid_height 15 --max_flight_time 2000 --n_runs 1"

# Enable Sequential GG only, disable Random RR
"--sequential_GG true --random_RR false"
```

---

## 📂 Outputs

After each run:

```
Results/
  revenue/YYYY-MM-DD/simulation_X/
  sequences/YYYY-MM-DD/simulation_X/
  waypoints/YYYY-MM-DD/simulation_X/
Benchmarking/IRADA/
  revenue/YYYY-MM-DD/simulation_X/
  sequences/YYYY-MM-DD/simulation_X/
run_logs/
  run_1_prep.txt
  run_1_main.txt
  run_1_analysis.txt
```

* **Revenue Excel** → per-algo totals (per round).
* **Sequences Excel** → UAV tours.
* **Waypoints Excel** → grid coords & revenues.
* **Plots** → consolidated revenue, IRADA boxplots, UAV contribution, flight-time left.

---

# 📈 Interpreting Results

* **Revenue plots**: compare convergence of total revenue across algorithms.
* **Boxplots**: distribution of results across runs (total revenue, UAV contributions, flight-time left).
* **IRADA**: always benchmarked side-by-side with other strategies.
* **GIFs**: optional animated UAV routes (if `GifGeneration=True` in `Analysis.py`).

---

# 🔧 Extending the Framework

* Add new allocators (e.g., CNP, CBBA, TS-DTA) by subclassing `TaskAllocator`.
* Toggle them in `Config` via `sequential_X`, `random_X`.
* New Excel outputs are auto-picked up by `Analysis.py`.

---


## 🔧 Extending the Framework

* **Add new allocators**: subclass `TaskAllocator`, implement `allocate(pool)` → `(rates, history)`.
* **Toggle in Config**: add `sequential_NewAlgo`, `random_NewAlgo`, include in `strategies` dict.
* **Analysis**: new Excel files are auto‑picked up by `Analysis.py` functions.

---

*Work in progress—future improvements:* advanced multi‑objective metrics, dynamic deadlines, real‑world maps.


