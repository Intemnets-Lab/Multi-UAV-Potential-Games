# Multi-UAV Potential Game Framework

> **A game-theoretic approach to decentralized multi-UAV persistent monitoring with convergence guarantees**

This repository implements the algorithms and simulation framework described in the thesis:
**"Path Optimization for UAV Waypoint Navigation Using Potential Game Theory"**
(Loyola Marymount University, 2025)

### 🎯 What does it do?

Coordinates **3–10 UAVs** to monitor a grid of waypoints (e.g., wildfire perimeters) by:
- Modeling coordination as an **exact potential game** with guaranteed Nash equilibrium convergence
- Linking revisit frequency to **Nyquist sampling requirements** for temporal coverage guarantees  
- Supporting **controlled overlap** at high-priority locations for redundancy
- Benchmarking against **IRADA** (state-of-the-art distributed allocation)

### 🚀 Why use this framework?

Unlike heuristic or centralized approaches, this provides:
- ✅ **Convergence guarantees** via potential game theory
- ✅ **Decentralized negotiation** (no single point of failure)
- ✅ **Tunable redundancy** (overlap mode for safety-critical regions)
- ✅ **Reproducible benchmarking** (open-source outputs, configs, plots)

---

---
## 📦 Quick Start (2 minutes)

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

---
## **Installation & First Run**

Before running the simulation framework, ensure you have the following installed:

**Required Software:**
- **Python 3.10+** (tested on 3.10.18-3.11)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

**Recommended Tools:**
- **VS Code** or **PyCharm** (for code editing)
- **Terminal/Command Prompt** (for running scripts)

***

### Requirements

Create a `requirements.txt` file with:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
pyyaml>=5.4.0
openpyxl>=3.0.0
imageio>=2.9.0
```


***

### **Step 1: Clone the Repository**

```bash
# Clone the repository
git clone https://github.com/Intemnets-Lab/Multi-UAV-Potential-Games.git

# Navigate into the created directory
cd Multi-UAV-Potential-Games

# Verify you're in the right place
ls
# You should see files like: Games.py, IRADA.py, Analysis.py, settings.yaml, etc.

```

***

### **Step 2: Install Dependencies**

Create a virtual environment (recommended) and install required packages:

```bash
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
```bash
pip install numpy pandas matplotlib pyyaml openpyxl imageio
```

***

### **Step 3: Verify `settings.yaml` Configuration**

Open `settings.yaml` and verify/modify the basic parameters:

```yaml

# Simulation
simulation:
  n_runs: 100           # Number of simulation runs
  seed: 42              # Random seed (null for random)
  enable_logging: true  # Enable detailed logs

# Grid configuration
grid:
  width: 5              # Grid width (number of columns)
  height: 5             # Grid height (number of rows)
  spacing: 1000         # Spacing between waypoints (meters)
  zero_prob: 0.3         # Probability of zero-revenue waypoints
  lambda: 0.1           # For IRADA only

# UAV configuration
uav:
  num_uavs: 2           # Number of UAVs
  speed: 20             # UAV speed (m/s)
  max_flight_time: 1800 # Max flight time (seconds, e.g., 30 min)

# Revenue configuration
revenue:
  random: true          # Use random revenue (true/false)
  fixedvalue: 50        # Fixed revenue if random=false
  min: 10               # Min random revenue
  max: 100              # Max random revenue



# Algorithms to run (true/false)
algorithms:
  sequentialGG: true
  sequentialGR: true
  sequentialRG: true
  sequentialRR: true
  randomGG:     true
  randomGR:     true
  randomRG:     true
  randomRR:     true
```

**Key Parameters for First Run:**
- **`num_uavs: 2`** - Start small to verify the setup works
- **`grid.width: 5`, `grid.height: 5`** - Generates a 5×5 grid (24 waypoints + 1 depot)
- **`n_runs: 5`** - Run 5 simulations for statistical analysis
- **`algorithms`** - Enable only `sequentialGG` and `randomRG` for faster testing (sample)

***

### **Step 4: Run Your First Simulation**

#### **Option A: Run Non-Overlap & Overlap Games**

```bash
python Games.py
```

**What Happens:**
1. Creates dated folders under `Results/NonOverlap/` and `Results/Overlap/`
2. Runs all enabled algorithms (from `settings.yaml`)
3. Generates Excel files for:
   - **Revenue rates** (`revenue/YYYY-MM-DD/simulationN/*.xlsx`)
   - **Waypoint sequences** (`sequences/YYYY-MM-DD/simulationN/*_sequences.xlsx`)
   - **Waypoint grids** (`waypoints/YYYY-MM-DD/simulationN/*.xlsx`)
4. Outputs negotiation logs (if `enablelogging: true`)

**Expected Output:**
```
Simulation Started
NonOverlap SimRun 1/5
DEBUG: SimRun 1 Mode NonOverlap Overlap=False
DEBUG: NonOverlap SimRun 1 Running Preflight
Running negotiation/output for NonOverlap, SimRun 1, preflight status=True
Wrote SimRun1 to NonOverlap/ModeGGSequential
Overlap SimRun 1/5
...
Simulation Complete
```

***

#### **Option B: Run IRADA Benchmark**

```bash
python IRADA.py
```

**What Happens:**
1. Searches for the latest Non-Overlap waypoint file (uses same grid for fair comparison)
2. Runs IRADA allocator for `nruns` simulations
3. Outputs to `BenchmarkingIRADA/revenue/` and `BenchmarkingIRADA/sequences/`

**Expected Output:**
```
IRADA Run 1/5
IRADA run took 3.45s
IRADA Run 2/5
...
All IRADA runs done.
```

***

#### **Option C: Run Analysis & Generate Plots**

After running simulations, generate visualizations:

```bash
python Analysis.py
```

**What Happens:**
1. Automatically finds the latest simulation folders
2. Generates per-algorithm revenue plots
3. Creates consolidated comparison plots
4. Produces boxplots for statistical analysis
5. Saves outputs to `Visualizations/`

**Expected Output:**
```
INFO: Using max_rounds=50
RUN: Generating NonOverlap graphs
Saved plot: Visualizations/NonOverlap/plots/.../UAV0_meanstd.png
...
Saved NonOverlap-vs-IRADA final-total boxplot
```

***

### **Step 5: Verify Outputs**
```
Results/
├── NonOverlap/
│   ├── revenue/2025-12-16/simulation_1/
│   │   ├── UAVs2_GRID5_ModeGG_Sequential.xlsx
│   │   └── UAVs2_GRID5_ModeGR_Sequential.xlsx
│   ├── sequences/2025-12-16/simulation_1/
│   │   └── UAVs2_GRID5_1800_20_ModeGG_Sequential_sequences.xlsx
│   └── waypoints/2025-12-16/simulation_1/
│       └── UAVs2_GRID5_waypoints.xlsx
├── Overlap/
│   └── (same structure)
└── BenchmarkingIRADA/
    └── (same structure)

Visualizations/
├── NonOverlap/plots/2025-12-07/simulation_1/
│   ├── analysis_plots/
│   │   ├── UAVs9_GRID13_ModeRR_Sequential/
│   │   │   ├── UAV0_mean_std.png
│   │   │   ├── UAV1_mean_std.png
│   │   │   ├── Total_mean_std.png
│   │   │   └── Consolidated_mean.png          ← Per-algorithm consolidated
│   │   └── UAVs9_GRID13_ModeGG_Sequential/
│   │       └── (same structure)
│   ├── combined_total_revenue_rate.png         ← Cross-algorithm comparison
│   └── boxplots_uav_contribution/
│       ├── uav_contribution_ModeGG_Sequential.png
│       └── uav_contribution_ModeRR_Sequential.png
│
├── Overlap/plots/2025-12-07/simulation_1/
│   ├── analysis_plots/
│   │   ├── UAVs9_GRID13_ModeRG_Sequential/
│   │   │   ├── UAV0_mean_std.png
│   │   │   ├── UAV1_mean_std.png
│   │   │   ├── Total_mean_std.png              ← Note: Total_mean_std.png (with underscores)
│   │   │   └── Consolidated_mean.png
│   │   └── (other algorithms)
│   ├── combined_total_revenue_rate.png
│   └── boxplots_uav_contribution/
│       ├── uav_contribution_ModeRG_Sequential.png
│       └── uav_contribution_ModeGR_Random.png
│
├── IRADA/plots/2025-12-07/simulation_1/
│   ├── analysis_plots/
│   │   └── IRADA/
│   │       ├── UAV0_mean_std.png
│   │       ├── UAV1_mean_std.png
│   │       ├── Total_mean_std.png
│   │       └── Consolidated_mean.png
│   ├── combined_total_revenue_rate.png
│   └── boxplots_uav_contribution/
│       └── uav_contribution_IRADA.png
│
└── Comparisons/2025-12-07/simulation_1/
    ├── final_total_nonoverlap_vs_irada.png
    ├── final_total_overlap_only.png
    └── flight_time_left_all_algorithms.png

```

***

### **Step 6: Inspect Key Outputs**

#### **Revenue Workbook** (`UAVs2_GRID5_ModeGG_Sequential.xlsx`)
- **Sheets**: `SimRun1`, `SimRun2`, ..., `SimRun5`
- **Columns**: `negotiation_round`, `UAV0`, `UAV1`, ...
- **Values**: Revenue rate per UAV per negotiation round

#### **Sequences Workbook** (`*_sequences.xlsx`)
- **Columns**: `negotiation_round`, `UAV0`, `m0`, `UAV1`, `m1`, ...
- **`UAVk`**: Waypoint sequence (e.g., "3-7-12")
- **`mk`**: Travel time/cost (mⱼ) for that sequence

#### **Waypoints Workbook** (`UAVs2_GRID5_waypoints.xlsx`)
- **Columns**: `Waypoint`, `Revenue`, `X`, `Y`
- **Rows**: One per waypoint (grid positions and revenues)

#### **Plots** (in `Visualizations/`)
- **`Consolidated_mean.png`**: Shows all UAV revenue trends + system mean
- **`finaltotal_nonoverlapvsirada.png`**: Boxplot comparing final performance

***

### **Common Issues & Fixes**

#### **Issue 1: `FileNotFoundError: settings.yaml`**
**Fix:** Ensure `settings.yaml` is in the same directory as the Python scripts.

#### **Issue 2: `ModuleNotFoundError: No module named 'yaml'`**
**Fix:** Install missing dependencies:
```bash
pip install pyyaml openpyxl
```

#### **Issue 3: Preflight Check Fails**
**Symptom:** `ERROR: Preflight failed (tour exceeds max_flight_time)`
**Fix:** Increase `maxflighttime` or decrease `grid.width/height` in `settings.yaml`:
```yaml
uav:
  max_flight_time: 3600  # Increase to 60 minutes
```

#### **Issue 4: IRADA Can't Find Waypoints**
**Symptom:** `FileNotFoundError: Expected NonOverlap waypoint folder does NOT exist`
**Fix:** Run `Games.py` first to generate Non-Overlap waypoint files, then run `IRADA.py`.

#### **Issue 5: No Plots Generated**
**Fix:** Check `Analysis.py`:
```python
  GraphGeneration: bool = True  # Must be true for Analysis.py
  GifGeneration: bool = False   # To visualize the tours 
```

***

### **Quick Test Run (30 seconds)**

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
  sequentialGG: true
  # Set all others to false
```

Then run:
```bash
python Games.py && python Analysis.py
```

You should see:
- 2 runs completed in ~10-20 seconds
- Excel files in `Results/`
- Plots in `Visualizations/`
- However, 100 runs with real-world parameters (from the configuration table below) completed in ~12 hours for 3 UAVs

***

### **Next Steps**

After verifying the basic setup:
1. **Scale Up**: Increase `num_uavs` to 3-5, `grid.width/height` to 5-10
2. **Enable More Algorithms**: Turn on Random strategies in `settings.yaml`
3. **Run Batch Simulations**: Use `Simulate.sh` (see next section)
4. **Explore Parameter Sensitivity**: Vary `speed`, `max_flight_time`, `zero_prob`

***

This section gives a complete walkthrough from installation to first successful run, with troubleshooting for common issues.


### Batch Execution with `simulate.sh`

For running multiple parameter sweeps:

```
#!/bin/bash
# Example: Test different UAV counts and grid sizes

for uavs in 2 3 5; do
  for grid in 5 10 15; do
    python Games.py --num_uavs $uavs --grid_width $grid --grid_height $grid --n_runs 10
    python IRADA.py --num_uavs $uavs --grid_width $grid --grid_height $grid --n_runs 10
    python Analysis.py --num_uavs $uavs --grid_width $grid --grid_height $grid
  done
done
```

**Usage:**
```
chmod +x Simulate.sh
./Simulate.sh
```
```

***
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

***


### Algorithm Naming Convention

Each algorithm is identified by three components:

| **Component** | **Options** | **Meaning** |
|--------------|------------|-------------|
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

**Short Labels (used in plots):**
- `NSGG` = NonOverlap, Sequential, Greedy-Greedy
- `ORGR` = Overlap, Random, Greedy-Random
- `IRADA` = IRADA benchmark (chronological event-driven)
```

***
---
## **Project Structure**

```
├── Games.py            # Main simulation (Non-Overlap & Overlap games)
├── IRADA.py            # IRADA benchmark allocator
├── Analysis.py         # Post-processing and visualization
├── settings.yaml       # Configuration file
├── Simulate.sh         # Batch execution script
├── Results/            # Simulation outputs
│   ├── NonOverlap/
│   │   ├── revenue/
│   │   ├── sequences/
│   │   └── waypoints/
│   ├── Overlap/
│   └── (same structure)
└── BenchmarkingIRADA/  # IRADA outputs
```

***

## 📂 Code Structure

```
├─ Games.py   # Simulation runner with algorithms, Excel, log files
├─ IRADA.py   # IRADA with algorithms, Excel, log files
├─ Analysis.py        # Post‑processing: plots, boxplots, animations from Excel
├─ Results/
```

---
## **File 1: Games.py**

### **Purpose**
Runs the Non-Overlapping and Overlapping game simulations with negotiation-based task allocation.

### **Core Classes & Functions**

#### **1. Config Class**
```python
class Config:
    def __init__(self, data: dict)
    @classmethod
    def fromyaml(cls, path='settings.yaml')
    def override(self, overrides: dict)
```
- **`__init__`**: Loads simulation parameters from YAML (grid size, UAV count, speed, max flight time, revenue ranges, algorithm toggles).
- **`fromyaml`**: Factory method to create Config from `settings.yaml`.
- **`override`**: Applies CLI overrides (e.g., `--numuavs 5`).

***

#### **2. Logger Class**
```python
class Logger:
    def __init__(self, outputdir, filename='negotiationlog.txt', enabled=True)
    def info(self, msg)
    def debug(self, msg)
```
- **`__init__`**: Creates a timestamped log file in `outputdir`.
- **`info/debug`**: Writes messages with timestamps to the log (e.g., negotiation rounds, UAV decisions).

***

#### **3. WaypointManager Class**
```python
class WaypointManager:
    def __init__(self, config, log, presetwaypoints=None, presetvalues=None)
    def generategrid(self) -> List[Tuple[float, float]]
    def drawrevenues(self) -> List[int]
    def applyzeroprob(self)
    def initclonesthresholdbased(self)
    def ensureclonesexistandwire(self, sequences)
```
- **`__init__`**: Initializes the waypoint grid (excluding depot at (0,0)) and revenues. If `presetwaypoints/presetvalues` are provided (for Overlap mode), uses them; otherwise generates fresh.
- **`generategrid`**: Creates a grid of waypoints at `(x*spacing, y*spacing)` for x in [0, W), y in [0, H), excluding depot.
- **`drawrevenues`**: Randomly assigns revenue values to each waypoint from `[revenuemin, revenuemax]`.
- **`applyzeroprob`**: Sets revenue to 0 for a fraction of waypoints based on `zeroprob` (simulating low-value areas).
- **`initclonesthresholdbased`**: (Overlap mode) Creates clone waypoints for high-revenue POIs above `clonethreshold`. Clones have identical coordinates and revenues.
- **`ensureclonesexistandwire`**: Ensures clones match their originals' revenues (called after revenue redraw).

***

#### **4. PathOptimizer Class**
```python
class PathOptimizer:
    @staticmethod
    def euclidean(a, b) -> float
    def STSPSolver(self, depot, waypoints, speed, maxflighttime) -> (List[int], float)
    def simulatemj(self, depot, waypoints, speed, maxflighttime) -> (float, float, float, float)
```
- **`euclidean`**: Computes Euclidean distance between two points.
- **`STSPSolver`**: Uses 2-opt heuristic to optimize the Traveling Salesman Problem (TSP) tour starting/ending at depot. Returns best order and `mⱼ` (time/distance cost).
- **`simulatemj`**: Simulates multi-loop (MJ) tours: calculates forward leg, return leg, and jump-back distances for a given sequence.

***

#### **5. UAVAgent Class**
```python
class UAVAgent:
    def __init__(self, uid, manager, optimizer, config, logger)
    def remainingcapacity(self, t: float) -> float
    def currentPOIs(self) -> List[int]
    def weightedcenter(self, t: float) -> Tuple[float, float]
    def setpath(self, path: List[int])
    def revenuerate(self) -> float
    def position(self, t: float) -> Tuple[float, float]
```
- **`__init__`**: Initializes UAV with ID, waypoint manager, optimizer, config, and logger.
- **`remainingcapacity`**: Returns remaining flight time budget at time `t`.
- **`currentPOIs`**: Returns the list of waypoints currently in the UAV's path.
- **`weightedcenter`**: Computes the revenue-weighted centroid of owned waypoints (used for IRADA's η computation).
- **`setpath`**: Assigns a new path (waypoint sequence) to the UAV.
- **`revenuerate`**: Computes average revenue per unit time for the last tour: `totalrevenue / totaltime`.
- **`position`**: Returns UAV's current coordinates (last waypoint visited or depot).

***

#### **6. InitialAssigner Class**
```python
class InitialAssigner:
    def __init__(self, config)
    def uniform(self, W) -> List[List[int]]
```
- **`__init__`**: Stores config.
- **`uniform`**: Randomly distributes waypoints `W` (list of POI indices) uniformly among UAVs using round-robin shuffling.

***

#### **7. TaskAllocator (Base Class)**
```python
class TaskAllocator:
    def __init__(self, manager, config, logger, optimizer)
    def setupagents(self, initialpaths=None) -> List[UAVAgent]
    def allocate(self, taskpool) -> (List[List[float]], List[List[List[int]]])
```
- **`__init__`**: Base class for allocation strategies.
- **`setupagents`**: Creates UAVAgent instances with initial paths (or uniform assignment).
- **`allocate`**: **Abstract method** – implemented by subclasses (e.g., NegotiationAllocator).

***

#### **8. NegotiationAllocator (Subclass of TaskAllocator)**
```python
class NegotiationAllocator(TaskAllocator):
    def __init__(self, manager, config, logger, dropstrategy, pickstrategy)
    def allocate(self, initialpaths) -> (List[List[float]], List[List[List[int]]])
```
- **`__init__`**: Defines drop strategy (greedy/random) and pick strategy (greedy/random), plus sequential/random ordering.
- **`allocate`**: **Core negotiation loop**:
  1. Each round, UAVs drop waypoints (greedy = lowest revenue, random = random drop).
  2. UAVs pick waypoints from the shared pool (greedy = highest revenue, random = random pick).
  3. Computes revenue rates, checks convergence (stagnation patience), and rollback detection.
  4. Returns `rates` (per-UAV revenue rate per round) and `history` (sequences per round).

**Key Features:**
- **Rollback detection**: Detects cycles in state and breaks the loop.
- **Convergence**: Stops if revenue doesn't improve for `self.patience` rounds.

***

#### **9. PreflightChecker Class**
```python
class PreflightChecker:
    def __init__(self, manager, config, log)
    def run(self) -> bool
```
- **`__init__`**: Initializes with waypoint manager, config, and logger.
- **`run`**: Checks if **any** UAV's initial tour exceeds `maxflighttime`. Returns `True` if all tours are feasible, `False` otherwise. (Only runs for Non-Overlap mode.)

***

#### **10. SimulationRunner Class**
```python
class SimulationRunner:
    def __init__(self, cfg, log)
    def findorcreatesimfolder(self) -> str
    def run(self)
    def definestrategies(self) -> Dict
    def prepareinitialsequences(self) -> List[List[int]]
    def computemjmatrix(self, history) -> (List[List[float]], List[List[List[int]]])
    def makeoutputs(self, rates, mjmatrix, history) -> (DataFrame, DataFrame, DataFrame)
    def writeincremental(self, modekey, stratname, runidx, dfrev, dfseq, dfwp)
```
- **`__init__`**: Sets up manager, assigner, optimizer, and output folders.
- **`findorcreatesimfolder`**: Finds the next available `simulationN` folder under `Results/{mode}/revenue/YYYY-MM-DD/`.
- **`run`**: **Main orchestrator**:
  1. Creates fresh Non-Overlap grid for each run.
  2. Runs Non-Overlap game (with preflight check).
  3. Adds clones for Overlap game using the same base grid.
  4. Executes all enabled strategies (ModeGG/GR/RG/RR × Sequential/Random).
  5. Writes outputs incrementally.
- **`definestrategies`**: Returns dictionary of strategy names → factory functions (e.g., `ModeGGSequential` → greedy drop, greedy pick, sequential).
- **`prepareinitialsequences`**: Filters zero-revenue waypoints and assigns them uniformly to UAVs.
- **`computemjmatrix`**: For each round's sequences, optimizes tours using 2-opt TSP and computes `mⱼ` values. Returns `mjmatrix` and `optimizedhistory`.
- **`makeoutputs`**: Creates three DataFrames:
  - **`dfrev`**: Revenue rates per UAV per round.
  - **`dfseq`**: Waypoint sequences and mⱼ values per UAV per round (interleaved columns).
  - **`dfwp`**: Waypoint coordinates and revenues.
- **`writeincremental`**: Appends a new sheet (`SimRun{runidx}`) to existing Excel files for revenue, sequences, and waypoints.

***

## **File 2: IRADA.py**

### **Purpose**
Implements the **IRADA** (Iterative Resource Allocation with Dynamic Adjustment) benchmark allocator using chronological event-driven scheduling.

### **Core Functions**

#### **1. IRADAAllocator Class**
```python
class IRADAAllocator(TaskAllocator):
    def __init__(self, manager, config, log, maxrounds=1000)
    def allocate(self, initialpaths) -> (List[List[float]], List[List[List[int]]])
```
- **`__init__`**: Initializes with max rounds and `κ` (kappa) coefficient.
- **`allocate`**: **Event-driven simulation**:
  1. Initializes each UAV with an "ownership" set (initial waypoints).
  2. Computes first waypoint picks using restricted pool (ownership).
  3. Uses a priority queue (heap) to process events chronologically: `(arrival_time, uav_id, waypoint)`.
  4. When a UAV arrives at a waypoint, it:
     - Records the trip segment.
     - Selects the next waypoint (or depot) using `selectnexttargetIRADA`.
     - Schedules the next arrival event.
  5. When a UAV returns to depot, it closes a "round" (depot→trip→depot) and computes revenue rate.
  6. Stops when all UAVs complete `maxrounds` depot returns.
  7. Logs communication timestamps (`lastcomm`) between UAVs.

***

#### **2. IRADA Score Functions**
```python
def computephi(agent, poiidx, t, allagents) -> float
def computeepsilon(agent, poiidx, t) -> float
def computeeta(agent, poiidx, t, allagents) -> float
def selectnexttargetIRADA(agent, t, allagents, includedepot=True, restrictpool=None) -> int
```
- **`computephi(φ)`**: **Information value coefficient**:
  - `φᵢ(t) = Î(i,t)` (estimated revenue/information at waypoint `i` at time `t`).
  
- **`computeepsilon(ε)`**: **Feasibility coefficient**:
  - `εᵢ,ᵥ(t) = exp(-γ · min(0, Rᵢ,ᵥ(t)))`
  - Where `Rᵢ,ᵥ(t) = C_remain(t) - dist(qᵥ, pᵢ) - dist(pᵢ, depot) - Cₘₐᵣgᵢₙ`
  - Penalizes waypoints that violate flight time constraints.

- **`computeeta(η)`**: **Communication coefficient**:
  - `ηᵢ,ᵥ(t) = Π_{u≠v, i∈ownership_u} [1 - exp(-λ(t - t_comm(v,u)))] · exp(-||pᵢ - c_u(t)||² / ||c_u(t) - c_v(t)||²)`
  - Encourages coordination: penalizes selecting waypoints owned by recently communicated UAVs and far from the agent's weighted center.

- **`selectnexttargetIRADA`**: Computes `score = φ · ε · η` for all waypoints (or depot) and selects the highest. If `restrictpool` is provided (first pick), only considers those waypoints.

***

#### **3. ChronoSimulationRunner Class**
```python
class ChronoSimulationRunner:
    def __init__(self, cfg, log)
    def run(self)
    def prepareoutputdirs(self) -> (str, str)
    def dumpexceldata(self, revdata, pathdata, revdir, pathdir)
```
- **`__init__`**: Initializes IRADA-specific runner.
- **`run`**: **Main IRADA execution**:
  1. Loads Non-Overlap waypoint file (using `findlatestwaypointsresultsroot`) to ensure IRADA uses the same grid as Non-Overlap.
  2. Runs `IRADAAllocator.allocate()` for `nruns` times.
  3. Collects per-UAV revenue rates and trip sequences.
  4. Writes outputs to `BenchmarkingIRADA/revenue/` and `BenchmarkingIRADA/sequences/`.
- **`prepareoutputdirs`**: Creates dated `simulationN` folders under `BenchmarkingIRADA/`.

Let me complete the **Analysis.py** section of the README with detailed function documentation:

***

## **File 3: Analysis.py**

### **Purpose**
Post-processes simulation outputs to generate visualizations, statistical analyses, and comparative plots across all three game modes (Non-Overlap, Overlap, IRADA).

### **Core Functions**

***

#### **1. Configuration & Setup Functions**

```python
class Config:
    def __init__(self, data: dict)
    @classmethod
    def fromyaml(cls, path='settings.yaml')
    def override(self, overrides: dict)
```
- **`__init__`**: Loads analysis configuration including paths (`resultsdir`, `visualizationdir`, `iradabenchmarkingdir`), simulation parameters, and master switches (`GraphGeneration`, `GifGeneration`).
- **`fromyaml`**: Factory method to create Config from `settings.yaml`.
- **`override`**: Applies CLI overrides for flexible parameter tuning.

```python
def setplotstyle()
```
- **Purpose**: Configures global matplotlib styling (font family, sizes) for publication-ready plots.
- **Sets**: Font family (Times New Roman), title size (18), axis labels (24), tick labels (24), legend (24).

***

#### **2. Path & File Management Functions**

```python
def findlatestsimulation(root: Path) -> Path
```
- **Purpose**: Finds the most recent simulation folder under a given root directory.
- **Logic**: 
  1. Sorts date folders (YYYY-MM-DD) under root.
  2. Finds highest `simulationN` folder under the latest date.
- **Returns**: Path to `Results/{mode}/{type}/YYYY-MM-DD/simulationN/`.

```python
def modefrompath(p: Path) -> str
```
- **Purpose**: Extracts game mode from path components.
- **Returns**: `"NonOverlap"`, `"Overlap"`, `"IRADA"`, or `"Other"`.

```python
def labelfromrevenuefile(f: Path) -> str
```
- **Purpose**: Creates concise algorithm labels from revenue workbook filenames.
- **Examples**:
  - `UAVs2GRID5ModeGGSequential.xlsx` → `"ModeGGSequential"`
  - `UAVs2GRID5IRADA.xlsx` → `"IRADA"`

```python
def shortalgolabel(gamelabel: str, algolabel: str) -> str
```
- **Purpose**: Maps verbose labels to compact taxonomy codes for plots.
- **Examples**:
  - `("NonOverlap", "ModeGGSequential")` → `"NSGG"`
  - `("Overlap", "ModeGRRandom")` → `"ORGR"`
  - `("IRADA", "IRADA")` → `"IRADA"`
- **Format**: `{Game}{Order}{Mode}` where:
  - Game: N (NonOverlap), O (Overlap), IRADA
  - Order: S (Sequential), R (Random)
  - Mode: GG, GR, RG, RR

```python
def visplotsrootforsim(visroot: Path, mode: str, revsimpath: Path) -> Path
def visgifsrootforsim(visroot: Path, mode: str, seqsimpath: Path) -> Path
def viscomparisonsroot(visroot: Path, nonrevsim: Path) -> Path
```
- **Purpose**: Constructs output paths for visualizations.
- **Returns**: 
  - `Visualizations/{mode}/plots/YYYY-MM-DD/simulationN/`
  - `Visualizations/{mode}/gifs/YYYY-MM-DD/simulationN/`
  - `Visualizations/Comparisons/YYYY-MM-DD/simulationN/`

***

#### **3. Revenue Analysis Functions**

```python
def analyzerevenue_excelsgraphs(exceldir: str, outroot: str = None)
```
- **Purpose**: **Primary revenue analysis function** - generates three types of plots for each revenue workbook:
  
  **Plot 1: Per-UAV Mean±Std Revenue Rate**
  - One plot per UAV showing mean revenue rate ± standard deviation across runs.
  - X-axis: Negotiation round
  - Y-axis: Revenue rate
  - Shaded region: Standard deviation band
  - Output: `{UAVk}_meanstd.png`

  **Plot 2: Total Revenue Rate Mean±Std**
  - Aggregated total revenue rate across all UAVs.
  - Shows system-wide performance per round.
  - Output: `Total_meanstd.png`

  **Plot 3: Consolidated Mean Plot**
  - All UAV means + system mean on one plot.
  - Legend positioned outside (right) with dynamic figure width.
  - Output: `Consolidated_mean.png`

- **Methodology**:
  1. Reads all sheets (SimRun1, SimRun2, ...) from each `.xlsx` file.
  2. Extracts UAV columns and pads to max rounds using forward-fill.
  3. Stacks across runs: `arr[run, round, uav]`.
  4. Computes `mean(axis=0)` and `std(axis=0, ddof=1)` across runs.

***

```python
def plotconsolidatedtotalrevenue(exceldir: str, outputpath: str = None)
```
- **Purpose**: **Cross-algorithm comparison** - plots mean total revenue rate for all algorithms on a single figure.
- **Features**:
  - Dynamic figure width based on number of algorithms (`basew + extra_per_algo × (nalgos - 3)`).
  - Uses short taxonomy labels (NSGG, ORGR, IRADA).
  - Legend outside plot area (right side).
- **Output**: `combined_total_revenue_rate.png` in `Visualizations/Comparisons/`.

***

#### **4. Boxplot Functions**

```python
def boxplotfinaltotalswitirada(revdirs: List[str/Path], outpng: str = None)
```
- **Purpose**: Creates **two separate boxplots** of final total revenue per run:
  
  **Boxplot 1: NonOverlap vs IRADA**
  - Includes all NonOverlap strategies + IRADA.
  - Excludes Overlap data.
  - Sorts so IRADA appears last (visual separation).
  - Output: `finaltotal_nonoverlapvsirada.png`

  **Boxplot 2: Overlap Only**
  - Includes only Overlap strategies.
  - Separate comparison to isolate Overlap performance.
  - Output: `finaltotal_overlaponly.png`

- **Methodology**:
  1. Reads all revenue workbooks from provided directories.
  2. Extracts final row (last negotiation round) from each sheet.
  3. Sums UAV columns to get total revenue.
  4. Groups by algorithm label.
  5. Uses orange median lines, grid for readability.

***

```python
def boxplotuavcontributionall(revsim: str/Path, outpng: str = None)
```
- **Purpose**: Analyzes **individual UAV contributions** to total revenue.
- **Generates**: One boxplot per algorithm showing final revenue distribution across UAVs.
- **Use Case**: Identifies workload balance - are some UAVs consistently underperforming?
- **Output**: Saved to `{revsim}/boxplots/uavcontribution/` with one plot per algorithm.

***

```python
def boxplotflighttimeleft(seqroots: List[str/Path], cfg: Config, outpng: str, nonoverlapwpsim: str/Path = None)
```
- **Purpose**: Computes **remaining flight time** for the final tour of each UAV in each run.
- **Methodology**:
  1. Parses UAV sequences from the last negotiation round.
  2. For **IRADA**: Computes single depot→tour→depot distance (requires NonOverlap waypoints file).
  3. For **Non-Overlap/Overlap**: Uses `mⱼ` values from sequences workbook.
  4. Calculates: `remaining = maxflighttime - (distance / speed)`.
  5. Plots boxplot showing feasibility margin.
- **Output**: `flighttimeleft_allalgorithms.png` in `Visualizations/Comparisons/`.
- **Insight**: Negative values indicate infeasible tours (constraint violations).

***

#### **5. Simulation Picking & Overrides**

```python
def picksim(rootseq: Path, rootrev: Path, manualdate: str = None, manualsim: str = None) -> (Path, Path)
```
- **Purpose**: Selects which simulation to analyze.
- **Logic**:
  - If `manualdate` and `manualsim` are provided (e.g., from YAML overrides), uses those.
  - Otherwise, finds the latest simulation using `findlatestsimulation()`.
- **Returns**: Tuple of `(sequences_sim_path, revenue_sim_path)`.

***

#### **6. Main Execution Block**

```python
if __name__ == "__main__":
    # 1. Load config from settings.yaml + CLI overrides
    # 2. Set plot style
    # 3. Pick simulations (NonOverlap, Overlap, IRADA)
    # 4. Generate per-algorithm graphs (if GraphGeneration enabled)
    # 5. Create consolidated comparisons
    # 6. Generate boxplots (final totals, UAV contributions, flight time)
```

**Execution Flow**:
1. **Config Loading**: Loads `settings.yaml`, applies CLI overrides (`--numuavs`, `--gridwidth`, etc.).
2. **Simulation Selection**:
   - NonOverlap: Uses manual overrides (`NONDATE`, `NONSIM`) or finds latest.
   - Overlap: Uses manual overrides (`OVERDATE`, `OVERSIM`) or finds latest.
   - IRADA: Uses manual overrides (`IRADADATE`, `IRADASIM`) or finds latest.
3. **Graph Generation** (if `GraphGeneration=True`):
   - Calls `analyzerevenue_excelsgraphs()` for each mode.
   - Generates per-UAV, total, and consolidated plots.
4. **Consolidated Comparisons**:
   - Calls `plotconsolidatedtotalrevenue()` for each mode.
   - Creates cross-algorithm comparison plots.
5. **Boxplot Generation**:
   - Final total revenue comparisons (NonOverlap vs IRADA, Overlap only).
   - UAV contribution analysis per mode.
   - Flight time left analysis across all modes.

***

#### **7. Helper Functions**

```python
def algolabelfromseqfile(seqfile: Path) -> str
```
- **Purpose**: Infers algorithm label from sequence filename.
- **Examples**:
  - `UAVs2GRID5_1003_ModeGG_Random_sequences.xlsx` → `"ModeGGRandom"`
  - `UAVs2GRID5IRADA_sequences.xlsx` → `"IRADA"`

```python
def getmaxroundsfromalgorithms(outputdir: str, datestr: str, simdir: str) -> int
```
- **Purpose**: Scans revenue and sequence workbooks to determine the maximum number of negotiation rounds across all runs/algorithms.
- **Use Case**: Ensures consistent x-axis limits when comparing algorithms with different convergence times.

```python
def loadwaypointrevenues(pathoroutputdir, datestr=None, simdir=None, cfg=None, runidx=1) -> (List[int], List[Tuple])
```
- **Purpose**: **Flexible waypoint loader** for IRADA analysis.
- **Two modes**:
  - **Case A (direct path)**: Load from exact path (e.g., `UAVs2GRID5waypoints.xlsx`).
  - **Case B (legacy)**: Construct path from `outputdir`, `datestr`, `simdir`, `cfg`.
- **Returns**: `(revenues, coords)` for a specific run (sheet `SimRun{runidx}`).

***

### **Key Analysis Outputs**

| **Output Type** | **Files Generated** | **Purpose** |
|-----------------|-------------------|-------------|
| **Per-Algorithm Revenue** | `{UAVk}_meanstd.png`, `Total_meanstd.png`, `Consolidated_mean.png` | Tracks revenue convergence per UAV and system-wide |
| **Cross-Algorithm Comparison** | `combined_total_revenue_rate.png` | Compares all strategies (NSGG, ORGR, IRADA, etc.) |
| **Final Total Boxplots** | `finaltotal_nonoverlapvsirada.png`, `finaltotal_overlaponly.png` | Statistical comparison of final performance |
| **UAV Contribution Boxplots** | `uavcontribution/*.png` | Analyzes workload distribution across UAVs |
| **Flight Time Left Boxplot** | `flighttimeleft_allalgorithms.png` | Validates constraint satisfaction |

***

### **Usage Example**

```bash
# Run analysis with default settings
python Analysis.py

# Override specific simulation
python Analysis.py --numuavs 3 --gridwidth 10 --nruns 50

# Use manual simulation selection (edit settings.yaml)
# NONDATE: "2025-12-15"
# NONSIM: "simulation3"
python Analysis.py
```

***

This completes the comprehensive function documentation for all three core files.

This is an **excellent and comprehensive README**! You've covered all the critical components. Here are a few minor suggestions to make it even more complete:

***

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
sim_logs/
  run_1_prep.txt
  run_1_main.txt
  run_1_analysis.txt
```

* **Revenue Excel** → per-algo totals (per round).
* **Sequences Excel** → UAV tours.
* **Waypoints Excel** → grid coords & revenues.
* **Plots** → consolidated revenue, IRADA boxplots, per UAV contribution to total revenue rate, flight-time left.

---

## 📈 Interpreting Results

* **Revenue plots**: compare convergence of total revenue across algorithms.
* **Boxplots**: distribution of results across runs (total revenue, UAV contributions, flight-time left).
* **IRADA**: always benchmarked side-by-side with other strategies.
* **GIFs**: optional animated UAV routes (if `GifGeneration=True` in `Analysis.py`).

---

---
## 🔧 Extending the Framework

* Add new allocators (e.g., CNP, CBBA, TS-DTA) by subclassing `TaskAllocator`.
* Toggle them in `Config` via `sequential_X`, `random_X`.
* New Excel outputs are auto-picked up by `Analysis.py`.

---


* **Add new allocators**: subclass `TaskAllocator`, implement `allocate(pool)` → `(rates, history)`.
* **Toggle in Config**: add `sequential_NewAlgo`, `random_NewAlgo`, include in `strategies` dict.
* **Analysis**: new Excel files are auto‑picked up by `Analysis.py` functions.

---

*Work in progress—future improvements:* advanced multi‑objective metrics, dynamic deadlines, real‑world maps.


### Performance Tuning Tips

#### **For Large Grids (>10×10):**
- Increase `max_flight_time` to avoid preflight failures
- Enable only 2-3 algorithms initially (disable Random modes)
- Reduce `n_runs` to 5 for faster iteration

#### **For Many UAVs (>5):**
- Expect longer negotiation times (10-50 rounds)
- Use `enablelogging: false` to speed up execution
- Monitor `rollback_stasis` in logs (indicates cycles)

#### **For Statistical Significance:**
- Use `n_runs ≥ 30` for publication-ready results
- Set fixed `seed` for reproducibility across experiments
- Run IRADA with same `maxrounds` as longest MPG convergence

#### **Memory Optimization:**
- Disable GIF generation for large experiments
- Use `Analysis.py` with manual date/sim selection to avoid scanning all folders
```

***
---
## **Frequently Asked Questions (FAQ)**

**Q1: Why does Overlap sometimes perform worse than NonOverlap?**  
A: Clones add waypoints but don't increase total revenue. If `clone_threshold` is too low, UAVs waste time revisiting the same high-value locations instead of covering more area.

**Q2: Can I run only IRADA without MPG?**  
A: No. IRADA requires a NonOverlap waypoint file to ensure fair comparison on the same grid. Run `Overlap.py` first, then `IRADA.py`.

**Q3: What's the difference between `sequences.xlsx` and `revenue.xlsx`?**  
A: 
- `sequences.xlsx`: Lists which waypoints each UAV visits per round
- `revenue.xlsx`: Shows the revenue *rate* (revenue/time) achieved per round

**Q4: How do I reproduce thesis results exactly?**  
A: Use the same `seed`, `grid` parameters, and `n_runs` from the thesis config. Seed ensures identical random revenue/assignment.

**Q5: Can I visualize UAV paths on a map?**  
A: Not built-in. Export waypoint coordinates from `waypoints.xlsx` and plot using `matplotlib.pyplot.scatter()` or GIS tools.
```

***

### **9. MIT License (open-source)**

```markdown

### Troubleshooting Convergence Issues

#### **Symptom: Negotiation doesn't converge (>100 rounds)**
**Causes:**
- Grid too large relative to `max_flight_time`
- Too many zero-revenue waypoints (`zero_prob` too high)
- Rollback stasis (cyclic state repetition)

**Fixes:**
```
simulation:
  patience: 10          # Reduce patience for faster termination
  rollback_limit: 3     # Lower rollback tolerance
```

#### **Symptom: IRADA outperforms MPG significantly**
**Possible Reasons:**
- MPG stuck in local Nash equilibrium
- Initial assignment biased (try `AngleAssigner` instead of `uniform`)
- IRADA benefits from continuous optimization vs. discrete negotiation

**Analysis:**
- Compare `boxplots_uav_contribution` to check workload balance
- Inspect `negotiationlog.txt` for repeated drop/pick patterns
```

***
---
## **Known Limitations**

1. **2-opt TSP Heuristic**: Not guaranteed to find global optimum (use for speed over exactness)
2. **Static Revenue Model**: Waypoint values don't decay over time (future work: temporal dynamics)
3. **Homogeneous UAVs**: All UAVs have identical `speed` and `max_flight_time`
4. **Euclidean Distance**: Assumes flat terrain (no elevation or no-fly zones)
5. **Clone Threshold**: Fixed per simulation (future: adaptive cloning based on demand)
```


