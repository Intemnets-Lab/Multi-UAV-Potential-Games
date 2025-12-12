"""
simulation.py

Refactored, single-file implementation with modular classes for:
- Config: Simulation parameters
- Logger: File logging
- WaypointManager: Grid, values, deadlines
- PathOptimizer: 2-opt TSP + MJ simulation
- UAVAgent: Path and decision logic
- InitialAssigner: Uniform distribution
- TaskAllocator (base) + GreedyAllocator
- Visualizer: Plots, GIF, Excel
- SimulationRunner: Orchestrator
"""
import os
import sys
import random
from random import shuffle
import json
import math
import time
import copy
import numpy as np
import argparse
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from openpyxl import load_workbook
import pandas as pd
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from typing import List, Type
import random
from typing import List, Optional, Tuple, Dict
import yaml
from pathlib import Path

# --- settings.yaml loader (inline; no common_config.py needed) ---
from types import SimpleNamespace

def _dict_to_ns(d):
    return SimpleNamespace(**{k: _dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})

class Config:
    def __init__(self, data: dict):
        # ----- project -----
        self.results_dir = data["project"]["results_dir"]
        self.irada_benchmark_dir = data["project"].get("IRADA_benchmarking_dir", None)

        # ----- simulation -----
        sim = data.get("simulation", {})
        self.seed = sim.get("seed", None)
        self.n_runs = sim.get("n_runs", 1)
        self.enable_logging = sim.get("enable_logging", True)

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # ----- grid -----
        self.grid_width   = data["grid"]["width"]
        self.grid_height  = data["grid"]["height"]
        self.grid_spacing = data["grid"]["spacing"]
        self.zero_prob    = data["grid"]["zero_prob"]
        self.lambda_param = data["grid"].get("lambda", 0.1)

        # ----- uav -----
        self.num_uavs = data["uav"]["num_uavs"]
        self.speed    = data["uav"]["speed"]
        self.max_flight_time = data["uav"]["max_flight_time"]

        # ----- revenue -----
        self.random_revenue = data["revenue"]["random"]
        self.fixed_revenue  = data["revenue"]["fixed_value"]
        self.revenue_min    = data["revenue"]["min"]
        self.revenue_max    = data["revenue"]["max"]

       # ----- overlap mode -----
        overlap_cfg = data.get("overlap", {})
        # Per-run flag (will be toggled by SimulationRunner)
        self.overlap = False
        # Threshold for cloning (from YAML)
        self.clone_threshold = overlap_cfg.get("clone_threshold", None)
        # How to assign clones initially: "same", "random", or "balanced"
        self.clone_assignment = overlap_cfg.get("clone_assignment", "random")

        # ----- algorithms -----
        for algo, enabled in data["algorithms"].items():
            setattr(self, algo, enabled)

    @classmethod
    def from_yaml(cls, path="settings.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(data)


    def override(self, overrides: dict):
        """Apply CLI overrides (flattened keys)."""
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __repr__(self):
        return f"<Config {self.__dict__}>"

    
def load_config(yaml_path="settings.yaml", cli_args=None):
    # 1. load defaults from yaml
    with open(yaml_path, "r") as f:
        base = yaml.safe_load(f)

    # 2. argparse only for keys that exist in yaml
    parser = argparse.ArgumentParser()
    for key, val in base.items():
        parser.add_argument(f"--{key}", type=type(val), default=None)

    args = parser.parse_args(cli_args)
    overrides = {k: v for k, v in vars(args).items() if v is not None}

    # 3. merge (CLI overrides > YAML defaults)
    merged = {**base, **overrides}

    return Config(**merged)

# === LOGGER ===
class Logger:
    def __init__(self, output_dir, filename="negotiation_log.txt", enabled: bool = True):
        self.log_path = os.path.join(output_dir, filename)
        self.enabled = enabled
        if not enabled:
            return
        with open(self.log_path,'w',encoding='utf-8') as f:
            f.write("=== Negotiation Log ===\n\n")
    def _write(self, level, msg):
        if not self.enabled:
            return
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_path,'a',encoding='utf-8') as f:
            f.write(f"[{ts}] {level}: {msg}\n")
    def info(self,msg):  self._write('INFO', msg)
    def debug(self,msg): self._write('DEBUG', msg)
    def error(self,msg): self._write('ERROR', msg)
    def log(self,msg):   self.info(msg)

# === WAYPOINT MANAGER ===
class WaypointManager:
    """Handles grid waypoints, revenues, clone creation, and mapping."""

    def __init__(self, config, log, preset_waypoints=None, preset_values=None):
        self.config = config
        self.log = log
        self.depot = (0.0, 0.0)

        # ---- Grid + Revenues ----
        if preset_waypoints is not None and preset_values is not None:
            self.waypoints = list(preset_waypoints)
            self.values    = list(preset_values)
        else:
            self.waypoints = self._generate_grid()
            self.values    = self._draw_revenues()

        self._apply_zero_prob()  # Apply zero_prob here, cleaner!

        # ---- Clones ----
        self.clone_able = []  # Cloneable waypoint indices
        self.clone_map  = {}  # {orig <-> clone}

        if getattr(self.config, "overlap", False):
            self._init_clones_threshold_based()

    def _apply_zero_prob(self):
        """Set revenue to zero per zero_prob, after creation."""
        zp = getattr(self.config, "zero_prob", 0)
        if zp > 0:
            rng = random.Random(getattr(self.config, "seed", None))
            self.values = [v if rng.random() >= zp else 0 for v in self.values]

    def _draw_revenues(self):
        min_r = getattr(self.config, "revenue_min", 10)
        max_r = getattr(self.config, "revenue_max", 100)
        rng = random.Random(getattr(self.config, "seed", None))
        return [rng.randint(min_r, max_r) for _ in self.waypoints]

    def redraw_revenues(self):
        self.values = self._draw_revenues()
        self._apply_zero_prob()
        # ensure clones match originals
        for a, b in self.clone_map.items():
            orig, clone = min(a, b), max(a, b)
            self.values[clone] = self.values[orig]

    def _generate_grid(self):
        spacing = self.config.grid_spacing
        W = self.config.grid_width
        H = self.config.grid_height
        return [(x * spacing, y * spacing)
                for y in range(H) for x in range(W)
                if not (x == 0 and y == 0)]

    def _init_clones_threshold_based(self):
        thr = getattr(self.config, "clone_threshold", None)
        if thr is None:
            return
        vmax = max(self.values)
        cutoff = thr * vmax if 0 < thr < 1 else thr
        num_orig = self.config.grid_width * self.config.grid_height - 1  # exclude depot
        for idx in range(num_orig):
            if self.values[idx] > cutoff:
                clone_idx = len(self.waypoints)
                self.waypoints.append(self.waypoints[idx])
                self.values.append(self.values[idx])
                self.clone_map[idx] = clone_idx
                self.clone_map[clone_idx] = idx
                self.clone_able.extend([idx, clone_idx])
                self.log.info(
                    f"[CLONE] Created clone WP{clone_idx} for WP{idx} (rev={self.values[idx]}, cutoff={cutoff})"
                )

    def ensure_clones_exist_and_wire(self, sequences):
        # Your existing logic is fine—can inline but functionally it's lean
        if not getattr(self.config, "overlap", False) or not self.clone_map:
            return sequences
        log = self.log
        new_sequences = [list(seq) for seq in sequences]
        n_uavs = len(new_sequences)
        seen = set()
        clone_pairs = []
        for a, b in self.clone_map.items():
            key = tuple(sorted((a, b)))
            if key not in seen:
                seen.add(key)
                clone_pairs.append(key)
        assign_mode = getattr(self.config, "clone_assignment", "same").lower()
        def choose_uav(preferred=None):
            if assign_mode == "same":
                return preferred if preferred is not None else 0
            if assign_mode == "random":
                return random.randrange(n_uavs)
            if assign_mode == "balanced":
                return min(range(n_uavs), key=lambda u: len(new_sequences[u]))
            return preferred if preferred is not None else 0
        for a, b in clone_pairs:
            owner_a = owner_b = None
            for u, seq in enumerate(new_sequences):
                if a in seq: owner_a = u
                if b in seq: owner_b = u
            if owner_a is not None and owner_b is not None:
                continue
            if owner_a is None and owner_b is None:
                u = choose_uav()
                new_sequences[u].extend([a, b])
                continue
            if owner_a is not None and owner_b is None:
                u = choose_uav(preferred=owner_a)
                new_sequences[u].append(b)
                continue
            if owner_b is not None and owner_a is None:
                u = choose_uav(preferred=owner_b)
                new_sequences[u].append(a)
                continue
        return new_sequences

    def num(self):
        return len(self.waypoints)
    def shared_pool(self):
        return [i for i, v in enumerate(self.values) if v > 0]

# === PREFLIGHT CHECKER ===    
class PreflightChecker:
    def __init__(self, manager: WaypointManager, cfg: Config, log: Logger):
        self.mgr, self.cfg, self.log = manager, cfg, log

    def run(self):
        # include *all* waypoints (even zero‑rev ones)
        pool = list(range(len(self.mgr.waypoints)))
        coords = [self.mgr.waypoints[i] for i in pool]
        msg = f"Preflight: checking {len(pool)} total waypoints (including zero‑revenue)"
        print(f"\n📌 {msg}")
        self.log.info(msg)
        if len(pool) <= 1:
            msg = (
                f"Insufficient Waypoints: "
                f"found {len(pool)} waypoint"
                + ("s" if len(pool) != 1 else "")
                + "; need at least 2 to form a route."
            )
            print(f"❌ {msg}")
            self.log.error(msg)
            raise RuntimeError(msg)
        # 1) Depot → first
        depot = self.mgr.depot
        first = coords[0]
        leg0 = PathOptimizer.euclidean(depot, first)
        print("\n🔎 Tour leg distances:")
        line = f"    Depot→{pool[0]}: {leg0:.2f}m"
        print(line); self.log.debug(line)
        # 2) forward segments
        total_dist = leg0
        for idx in range(len(coords)-1):
            d = PathOptimizer.euclidean(coords[idx], coords[idx+1])
            total_dist += d
            line = f"    {pool[idx]}→{pool[idx+1]}: {d:.2f}m"
            print(line); self.log.debug(line)
        # 3) last → depot
        last = coords[-1]
        ret = PathOptimizer.euclidean(last, depot)
        total_dist += ret
        line = f"    {pool[-1]}→Depot: {ret:.2f}m"
        print(line); self.log.debug(line)
        # summary
        speed = self.cfg.speed
        tour_time = total_dist / speed
        info = (
            f"\n⏱ Total tour distance = {total_dist:.2f}m, "
            f"at speed={speed:.2f} ⇒ time = {tour_time:.2f}s "
            f"(budget={self.cfg.max_flight_time:.2f}s)"
        )
        print(info)
        self.log.info(info)
        if tour_time > self.cfg.max_flight_time:
            shortage = tour_time - self.cfg.max_flight_time
            minutes  = math.ceil(shortage / 60)
            msg = (
                f"❌ Mission infeasible: "
                f"tour needs {tour_time:.2f}s, "
                f"short by {shortage:.2f}s (~{minutes} min)."
            )
            print(msg)
            self.log.error(msg)
            return False
        success = "✅ Preflight check passed: full tour fits within max_flight_time"
        print(success)
        self.log.info(success)
        return True

class PathOptimizer:
    @staticmethod
    def euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @classmethod
    def simulate_mj(cls, depot, pts, speed, max_t):
        """
        Compute mj and related tour metrics for a *given ordered* list of points.
        Returns:
            mj, fwd, ret, total_time
        """
        n = len(pts)
        if n == 0:
            return 0, 0.0, 0.0, 0.0

        first = cls.euclidean(depot, pts[0])
        fwd   = sum(cls.euclidean(pts[i], pts[i+1]) for i in range(n-1))
        ret   = cls.euclidean(pts[-1], depot)
        jump  = cls.euclidean(pts[-1], pts[0]) if n > 1 else 0.0

        denom = fwd + jump
        if denom <= 0:
            mj = 1
            total_dist = first + ret
        else:
            mj = math.floor((speed * max_t - first - ret + jump) / denom)
            mj = max(mj, 1)
            total_dist = first + mj * fwd + (mj - 1) * jump + ret

        total_time = total_dist / speed
        return mj, fwd, ret, total_time

    @classmethod
    def STSPSolver(cls, depot, pts, speed, max_t):
        """
        2-opt STSP on 'pts' (coordinates, not indices).
        Returns:
            best_order : list of indices into pts
            mj         : loops feasible under (speed, max_t)
        """
        n = len(pts)
        if n == 0:
            return [], 0

        path = list(range(n))

        def cycle_len(p):
            d = cls.euclidean(depot, pts[p[0]])
            for i in range(n - 1):
                d += cls.euclidean(pts[p[i]], pts[p[i+1]])
            d += cls.euclidean(pts[p[-1]], depot)
            return d

        best = path[:]
        best_d = cycle_len(path)
        improved = True

        # Full 2-opt (i can be 0)
        while improved:
            improved = False
            for i in range(0, n - 1):
                for k in range(i + 1, n):
                    new_p = path[:i] + path[i:k+1][::-1] + path[k+1:]
                    d = cycle_len(new_p)
                    if d < best_d:
                        best, best_d, path = new_p, d, new_p
                        improved = True

        # Compute mj *using simulate_mj only*
        reordered_pts = [pts[i] for i in best]
        mj, _, _, _ = cls.simulate_mj(depot, reordered_pts, speed, max_t)
        return best, mj

# =====================================================================
class UAVAgent:
    def __init__(self, uid, manager, optimizer, config, logger):
        self.uid      = uid
        self.manager  = manager
        self.opt      = optimizer
        self.cfg      = config
        self.log      = logger
        self.sequence = []

    def revenue_rate(self, seq=None):
        """Calculate revenue rate for given seq (or current if None). Always collapses clones/bases using exclude_repeated_locs."""
        if seq is None:
            seq = self.sequence
        if not seq:
            return 0.0
        coords = self.manager.waypoints
        values = self.manager.values
        clone_map = getattr(self.manager, "clone_map", {})
        uniq_seq = self.exclude_repeated_locs(seq, clone_map, coords)
        # Compute unique path for MJ/tour and reward
        seq_coords = [coords[i] for i in uniq_seq]
        mj, _, _, t = self.opt.simulate_mj(self.manager.depot, seq_coords,
                                           self.cfg.speed, self.cfg.max_flight_time)
        rev = sum(values[wp] for wp in uniq_seq)
        f = mj / t if t > 0 else 0.0
        self.log.info(f"[SUMMARY] UAV{self.uid} revenue_rate={rev*f:.3f} for seq={seq}")
        return rev * f

    @staticmethod
    def exclude_repeated_locs(seq, clone_map, waypoints):
        # (Keep the first occurrence of each coordinate)
        kept = []
        seen = set()
        for wp in seq:
            coord = tuple(waypoints[wp])
            if coord not in seen:
                kept.append(wp)
                seen.add(coord)
        return kept

    def drop_waypoint(self, select_mode="greedy"):
        """
        Try dropping each WP (and its paired twin, if present), keep best positive-gain candidate.
        Always drop both base and clone if both are present in this UAV.
        """
        S_j = list(self.sequence)
        Z_old = self.revenue_rate()
        clone_map = getattr(self.manager, "clone_map", {})
        inv_clone_map = {v: k for k, v in clone_map.items()}
        candidates = []  # (wp, clone_wp, new_seq, gain)

        for i, wp in enumerate(S_j):
            # Always check for its twin (base or clone)
            twin = clone_map.get(wp, inv_clone_map.get(wp, None))
            # Remove both wp and its twin (if present), for gain evaluation
            test_seq = [x for x in S_j if x != wp and x != twin]
            gain = self.revenue_rate(test_seq) - Z_old
            if gain > 0:
                clone_wp = twin if twin in S_j else None
                candidates.append((wp, clone_wp, test_seq, gain))
        
        # ===== ADD THIS DEBUG LOG HERE =====
        if candidates:
            self.log.debug(f"[DROP_CANDIDATES] UAV{self.uid} has {len(candidates)} drop candidates:")
            for wp, clone_wp, _, gain in candidates:
                twin_info = f" (+twin {clone_wp})" if clone_wp is not None else ""
                self.log.debug(f"  - WP{wp}{twin_info}: gain={gain:.4f}")
        else:
            self.log.debug(f"[DROP_CANDIDATES] UAV{self.uid}: No beneficial drops found")
    # ===================================

        if not candidates:
            return None, None, S_j, 0.0

        if select_mode == "random":
            wp_sel, clone_sel, new_seq, gain_sel = random.choice(candidates)
        else:
            wp_sel, clone_sel, new_seq, gain_sel = max(candidates, key=lambda x: x[3])

        self.log.info(
            f"[DROP] UAV{self.uid} dropped={wp_sel}"
            + (f" and twin={clone_sel}" if clone_sel is not None else "")
            + f" gain={gain_sel:.3f}"
        )

        # --- CRITICAL: Remove BOTH from the sequence when dropping ---
        self.sequence = [x for x in self.sequence if x != wp_sel and x != clone_sel]

        return wp_sel, clone_sel, list(self.sequence), gain_sel

    def pick_waypoint(self, pool, select_mode="greedy"):
        """
        Try to pick each WP (pool is [(wp, owner),...]), insert at all positions, keep best positive gain.
        If pick would result in duplicate location (base+clone), gain will be zero and not chosen.
        """
        S_j = list(self.sequence)
        Z_old = self.revenue_rate()
        clone_map = getattr(self.manager, "clone_map", {})
        candidates = []  # (wp, new_seq, gain)

        for wp, owner in pool:
            # Try all possible insertions
            for pos in range(len(S_j) + 1):
                test_seq = S_j[:pos] + [wp] + S_j[pos:]
                # Deduplicate for calculation: reward/tour gain will be 0 if clone+base present
                gain = self.revenue_rate(test_seq) - Z_old
                if gain > 0:
                    candidates.append((wp, test_seq, gain))

        # ===== ADD THIS DEBUG LOG HERE =====
        if candidates:
            # Group by waypoint (since same WP can have multiple positions)
            wp_best_gains = {}
            for wp, _, gain in candidates:
                if wp not in wp_best_gains or gain > wp_best_gains[wp]:
                    wp_best_gains[wp] = gain
            
            self.log.debug(f"[PICK_CANDIDATES] UAV{self.uid} has {len(wp_best_gains)} pickable waypoints:")
            for wp, best_gain in sorted(wp_best_gains.items(), key=lambda x: -x[1]):
                self.log.debug(f"  - WP{wp}: best_gain={best_gain:.4f}")
        else:
            self.log.debug(f"[PICK_CANDIDATES] UAV{self.uid}: No beneficial picks found")
        # ===================================

        if not candidates:
            return None, None, S_j, 0.0

        if select_mode == "random":
            wp_sel, new_seq, gain_sel = random.choice(candidates)
        else:
            wp_sel, new_seq, gain_sel = max(candidates, key=lambda x: x[2])

        self.log.info(f"[PICK] UAV{self.uid} picked={wp_sel} gain={gain_sel:.3f}")
        self.sequence = new_seq
        return wp_sel, None, new_seq, gain_sel

# =====================================================================
#                           INITIAL ASSIGNER
# =====================================================================
class InitialAssigner:
    """
    Uniform random assignment of waypoint indices to UAVs.
    Returns raw sequences (not STSP-ordered).
    """
    def __init__(self, config: Config, logger: Logger = None):
        self.cfg = config
        self.log = logger

    def uniform(self, W):
        """
        Input:
            W : list of waypoint indices (all available tasks)

        Output:
            sequences : List[List[int]] for each UAV
        """
        num_uavs = self.cfg.num_uavs

        if self.log:
            self.log.debug(f"[INIT] Uniform assign {len(W)} WPs among {num_uavs} UAVs")

        sequences = [[] for _ in range(num_uavs)]

        rng = random.Random(self.cfg.seed if self.cfg.seed is not None else None)
        tasks = list(W)
        rng.shuffle(tasks)

        # round-robin assignment
        for i, wp in enumerate(tasks):
            sequences[i % num_uavs].append(wp)

        if self.log:
            for u, seq in enumerate(sequences):
                self.log.debug(f"[INIT] UAV{u} initial seq = {seq}")

        return sequences

# =====================================================================
#                           TASK ALLOCATOR (BASE)
# =====================================================================
class TaskAllocator:
    """
    Base class for Game-Theoretic allocators (NegotiationAllocator etc.)
    Handles agent creation and initialization.
    """

    def __init__(self, manager: WaypointManager, config: Config,
                 logger: Logger, optimizer: Type[PathOptimizer] = PathOptimizer):

        self.manager = manager
        self.cfg     = config
        self.log     = logger
        self.opt     = optimizer

    # ------------------------------------------------------------------
    # Create UAVAgent objects and assign initial sequences
    # ------------------------------------------------------------------
    def _setup_agents(self, initial_sequences=None):
        if initial_sequences is None:
            pool = self.manager.shared_pool()
            if self.log:
                self.log.debug(
                    f"[ALLOC] No initial sequences supplied → using uniform assignment over {len(pool)} tasks"
                )
            initial_sequences = InitialAssigner(self.cfg, self.log).uniform(pool)

        # === ADD THIS BLOCK HERE ===
        if getattr(self.cfg, "overlap", False) and self.manager.clone_map:
            initial_sequences = self.manager.ensure_clones_exist_and_wire(initial_sequences)
        # === END BLOCK ===

        agents = [
            UAVAgent(u, self.manager, self.opt, self.cfg, self.log)
            for u in range(self.cfg.num_uavs)
        ]
        for u, seq in enumerate(initial_sequences):
            agents[u].sequence = list(seq)
            if self.log:
                self.log.debug(f"[ALLOC] UAV{u} start seq = {agents[u].sequence}")
        return agents

class NegotiationAllocator(TaskAllocator):
    """
    UAV-centric negotiation allocator; supports overlap and clones with detailed logging.
    """

    def __init__(self, manager, config, log, drop_select, pick_select, max_rounds=100, patience=5):
        super().__init__(manager, config, log)
        self.drop_select = drop_select.lower()
        self.pick_select = pick_select.lower()
        self.max_rounds  = max_rounds
        self.patience    = patience

    def allocate(self, initial_sequences=None):
        overlap = bool(getattr(self.cfg, "overlap", False))
        clone_map = getattr(self.manager, "clone_map", {})
        inv_clone_map = {v: k for k, v in clone_map.items()}
        agents = self._setup_agents(initial_sequences)
        pool = []
        rates = []
        stagnant = 0
        eps = 1e-4
        history = []
        rollback_state_history = []
        rollback_repeat_limit = 5

        Z_old = sum(a.revenue_rate() for a in agents)
        self.log.info(f"Initial total revenue rate Z_old = {Z_old:.6f}")

        for rnd in range(1, self.max_rounds + 1):
            history.append([list(a.sequence) for a in agents])
            self.log.info(f"===== Negotiation Round {rnd} =====")

            # Log UAV details before round
            for i, a in enumerate(agents):
                rate = a.revenue_rate()
                self.log.info(f"  UAV{i} revenue_rate: {rate:.4f}")
                self.log.info(f"[DEBUG] UAV{i} full seq: {a.sequence}, used for route: {UAVAgent.exclude_repeated_locs(a.sequence, clone_map, self.manager.waypoints)}")

            # MJ values for each UAV (show path cost, etc.)
            mj_row = []
            for i, a in enumerate(agents):
                seq = a.sequence
                filtered = UAVAgent.exclude_repeated_locs(seq, clone_map, self.manager.waypoints)
                if len(filtered) > 1:
                    pts = [self.manager.waypoints[j] for j in filtered]
                    best_order, mj_val = self.opt.STSPSolver(self.manager.depot, pts, self.cfg.speed, self.cfg.max_flight_time)
                    # ✅ ADD DEBUG LOGGING HERE
                    optimized_indices = [filtered[idx] for idx in best_order]
                    self.log.info(f"  UAV{i} sequence BEFORE 2-opt: {filtered}")
                    self.log.info(f"  UAV{i} sequence AFTER 2-opt:  {optimized_indices}")
                else:
                    mj_val = 0
                mj_row.append(mj_val)
                self.log.info(f"  UAV{i} MJ: {mj_val:.4f}")
            # Total revenue this round
            Z_new = sum(a.revenue_rate() for a in agents)
            self.log.info(f"  Total revenue (Z_new): {Z_new:.4f}")

            # -------- DROP PHASE --------
            self.log.info("----> Drop Phase <----")
            drop_pool = []
            drop_agents = list(agents)
            if self.cfg.randomize_sequence:
                shuffle(drop_agents)  # or shuffle(pick_agents)
                self.log.info(f"[MARKET] Drop phase agent order: {[a.uid for a in drop_agents]}")
            for a in drop_agents:
                wp, clone_wp, new_seq, gain = a.drop_waypoint(self.drop_select)
                if wp is not None:
                    a.sequence = new_seq
                    pool.append((wp, a.uid))
                    drop_pool.append(wp)
                    if clone_wp is not None:
                        pool.append((clone_wp, a.uid))
                        drop_pool.append(clone_wp)
                self.log.info(f"[DEBUG] UAV{a.uid} full seq after drop: {a.sequence}, used for route: {UAVAgent.exclude_repeated_locs(a.sequence, clone_map, self.manager.waypoints)}")
            self.log.info(f"  Dropped waypoints this round (pool): {drop_pool}")

            # -------- PICK PHASE --------
            self.log.info("----> Pick Phase <----")
            pick_agents = list(agents)
            if self.cfg.randomize_sequence:
                shuffle(pick_agents)
                self.log.info(f"[MARKET] Pick phase agent order: {[a.uid for a in pick_agents]}")
            picked_this_round = []
            for a in pick_agents:
                if not pool:
                    break
                wp, _, new_seq, gain = a.pick_waypoint(pool, self.pick_select)
                if wp is not None:
                    a.sequence = new_seq
                    picked_this_round.append(wp)
                    pool = [(w, u) for (w, u) in pool if w != wp]
            self.log.info(f"  Picked waypoints this round: {picked_this_round}")
            for i, a in enumerate(agents):
                self.log.info(f"[DEBUG] UAV{i} full seq after pick: {a.sequence}, used for route: {UAVAgent.exclude_repeated_locs(a.sequence, clone_map, self.manager.waypoints)}")

            # -------- REINSERT REMAINING POOL --------
            self.log.info("----> Pool Reassignment Phase <----")
            reassigned_this_round = []
            for w, owner_uid in pool:
                agent_idx = next((i for i, a in enumerate(agents) if a.uid == owner_uid), None)
                if agent_idx is not None:
                    agent = agents[agent_idx]
                    # Optimal sequence logic for BOTH overlap and nonoverlap
                    best_gain = float('-inf')
                    best_seq = None
                    Z_agent_old = agent.revenue_rate()
                    S_j = list(agent.sequence)
                    for pos in range(len(S_j)+1):
                        candidate_seq = S_j[:pos] + [w] + S_j[pos:]
                        gain = agent.revenue_rate(candidate_seq) - Z_agent_old
                        if gain > best_gain:
                            best_gain = gain
                            best_seq = candidate_seq
                    if best_seq is not None:
                        agent.sequence = best_seq
                        reassigned_this_round.append(w)
                    self.log.info(
                        f"[DEBUG] UAV{agent.uid} full seq after pool reassignment: {agent.sequence}, "
                        f"used for route: {UAVAgent.exclude_repeated_locs(agent.sequence, clone_map, self.manager.waypoints)}"
                    )
            pool = []
            self.log.info(f"  Reassigned waypoints this round: {reassigned_this_round}")

            # -------- OVERLAPPING: LOG CLONE INFO --------
            if overlap and clone_map:
                self.log.info("----> Clone Pairs and Their Locations <----")
                for orig, clone in clone_map.items():
                    found_uav_orig = [a.uid for a in agents if orig in a.sequence]
                    found_uav_clone = [a.uid for a in agents if clone in a.sequence]
                    if found_uav_orig or found_uav_clone:
                        self.log.info(
                            f"   Clone pair ({orig}, {clone}): orig in UAV {found_uav_orig}, clone in UAV {found_uav_clone}")

            # -------- REVENUE RATE PER UAV --------
            current_rates = [a.revenue_rate() for a in agents]
            rates.append(current_rates)
            self.log.info(
                f"ROUND {rnd} END — Z_new={sum(current_rates):.6f}, ΔZ={sum(current_rates) - Z_old:.6f}"
            )

            # Rollback round if total revenue dropped
            if sum(current_rates) < Z_old - 1e-6:
                self.log.info("[ROLLBACK] Revenue decreased. Rolling back round.")
                old_state = history[-1]
                rollback_snapshot = tuple(tuple(seq) for seq in old_state)
                rollback_state_history.append(rollback_snapshot)
                # Only keep recent history (for better memory/speed)
                if len(rollback_state_history) > rollback_repeat_limit:
                    rollback_state_history.pop(0)
                repeated = rollback_state_history.count(rollback_snapshot)
                for idx, a in enumerate(agents):
                    self.log.info(f"[DEBUG] UAV{idx} full seq: {a.sequence}, used for route: {UAVAgent.exclude_repeated_locs(a.sequence,clone_map, self.manager.waypoints)}")
                    a.sequence = old_state[idx]
                Z_new = Z_old
                if repeated >= rollback_repeat_limit:
                    self.log.info(f"[ROLLBACK] Detected repeated rollback state {repeated} times. Ending negotiation (rollback stasis detected): {rollback_snapshot}")
                    break
            else:
                Z_old = sum(current_rates)
                rollback_state_history.clear()

            # ------- ROBUST CONVERGENCE -------
            if rnd > 1 and abs(sum(rates[-1]) - sum(rates[-2])) < eps:
                stagnant += 1
                if stagnant >= self.patience:
                    self.log.info(f"Converged after {rnd} rounds: Z_new did not improve for {self.patience} rounds.")
                    break
            else:
                stagnant = 0

        history.append([list(a.sequence) for a in agents])
        return rates, history

# ============================================================
# SIMULATION RUNNER
# ============================================================
class SimulationRunner:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log
        self.cfg.randomize_sequence = False
        self.manager = WaypointManager(self.cfg, self.log)
        self.assigner = InitialAssigner(self.cfg)
        self.opt = PathOptimizer
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.sim_folder = self._find_or_create_sim_folder()

    def _find_or_create_sim_folder(self):
        # Find next free simulation folder number
        for mode in ["NonOverlap", "Overlap"]:
            base_dir = os.path.join(self.cfg.results_dir, mode, "revenue", self.date_str)
            os.makedirs(base_dir, exist_ok=True)
            existing = [d for d in os.listdir(base_dir) if d.startswith("simulation_")]
            sim_num = len(existing) + 1
            return f"simulation_{sim_num}"

    def run(self):
        print("\n=== Simulation Started ===")
        strategies = self._define_strategies()

        for run_idx in range(1, self.cfg.n_runs + 1):

            # --- Always create a fresh NonOverlap manager/grid first for this run ---
            self.cfg.overlap = False
            nonoverlap_manager = WaypointManager(self.cfg, self.log)
            base_waypoints = nonoverlap_manager.waypoints.copy()
            base_values = nonoverlap_manager.values.copy()

            # --- Then run both games using these presets; add clones only in Overlap ---
            for mode_key, mode_flag in [("NonOverlap", False), ("Overlap", True)]:
                print(f"\n=== {mode_key} SimRun {run_idx} ===")
                print(f"(DEBUG) SimRun {run_idx} | Mode: {mode_key} | Overlap: {mode_flag}")
                self.cfg.overlap = mode_flag

                self.manager = WaypointManager(
                    self.cfg, self.log,
                    preset_waypoints=list(base_waypoints),
                    preset_values=list(base_values)
                )
                self.assigner = InitialAssigner(self.cfg)

                # ---- Preflight for NonOverlap only ----
                should_run = True
                precheck_passed = None
                if not mode_flag:
                    self.log.info(f"[INFO]📌 Running PreFlightCheck for Non-Overlap SimRun {run_idx}")
                    print(f"(DEBUG) NonOverlap SimRun {run_idx} | Running Preflight")
                    precheck = PreflightChecker(self.manager, self.cfg, self.log)
                    precheck_passed = precheck.run()
                    print(f"(DEBUG) Preflight returned: {precheck_passed}")
                    if not precheck_passed:
                        self.log.info(f"[ERROR] Preflight failed: tour exceeds max_flight_time. Skipping SimRun {run_idx}.")
                        should_run = False
                else:
                    self.log.info(f"[INFO]⏭️ Skipping PreFlightCheck for Overlap SimRun {run_idx}")

                print(f"(DEBUG) should_run: {should_run}")

                if not should_run:
                    continue  # Skip running this mode for this SimRun if preflight fails

                print(f"Running negotiation/output for {mode_key}, SimRun {run_idx}, preflight status: {precheck_passed if not mode_flag else 'N/A'}")

                for strat_name, make_alloc in strategies.items():
                    self.log.info(
                        f"\n=== Running {'Overlap' if self.cfg.overlap else 'NonOverlap'} Game: " +
                        f"{strat_name} ({'Random' if 'Random' in strat_name else 'Sequential'}) " +
                        f"— SimulationRun {run_idx} ===\n"
                    )
                    print(f"(DEBUG) SimRun {run_idx} | Mode: {mode_key} | Strategy: {strat_name}")
                    alloc = make_alloc()
                    rates, history = alloc.allocate(self._prepare_initial_sequences())
                    mj_matrix, optimized_history = self._compute_mj_matrix(history)
                    df_rev, df_seq, df_wp = self._make_outputs(rates, mj_matrix, optimized_history)
                    self._write_incremental(mode_key, strat_name, run_idx, df_rev, df_seq, df_wp)
                    print(f"✅ Wrote SimRun{run_idx} to {mode_key}/{strat_name}")

        print("\n=== Simulation Complete ===")

    def _define_strategies(self):
        def make_strategy(drop, pick, is_random):
            def factory():
                self.cfg.randomize_sequence = is_random
                return NegotiationAllocator(self.manager, self.cfg, self.log, drop, pick)
            return factory

        return {
            "ModeGG_Sequential": make_strategy("greedy", "greedy", False),
            "ModeGR_Sequential": make_strategy("greedy", "random", False),
            "ModeRG_Sequential": make_strategy("random", "greedy", False),
            "ModeRR_Sequential": make_strategy("random", "random", False),
            "ModeGG_Random": make_strategy("greedy", "greedy", True),
            "ModeGR_Random": make_strategy("greedy", "random", True),
            "ModeRG_Random": make_strategy("random", "greedy", True),
            "ModeRR_Random": make_strategy("random", "random", True),
        }

    def _prepare_initial_sequences(self):
        # Only select waypoints with value > 0 for allocation!
        W = [i for i, v in enumerate(self.manager.values) if v > 0]
        initial = self.assigner.uniform(W)
        optimized = []
        for seq in initial:
            filtered = UAVAgent.exclude_repeated_locs(seq, getattr(self.manager, "clone_map", {}), self.manager.waypoints)
            optimized.append(filtered)
        return optimized

    def _compute_mj_matrix(self, history):
        clone_map = getattr(self.manager, "clone_map", {})
        mj_matrix = []
        optimized_history = []  # ✅ NEW: Store optimized sequences
        
        for sequences in history:
            row_mj = []
            optimized_round = []  # ✅ NEW: Store optimized sequences for this round
            
            for u, seq in enumerate(sequences):
                filtered = UAVAgent.exclude_repeated_locs(seq, clone_map, self.manager.waypoints)
                
                if len(filtered) > 1:
                    pts = [self.manager.waypoints[i] for i in filtered]
                    best_order, mj = self.opt.STSPSolver(self.manager.depot, pts, self.cfg.speed, self.cfg.max_flight_time)
                    
                    # ✅ NEW: Convert back to waypoint indices
                    optimized_seq = [filtered[idx] for idx in best_order]
                    pts_reordered = [pts[idx] for idx in best_order]
                    self.log.info(f"  UAV{u} optimized WP indices: {optimized_seq}")
                    self.log.info(f"  UAV{u} coordinates: {[self.manager.waypoints[wp] for wp in optimized_seq]}")
                    
                    # Also manually verify mj calculation
                    mj_manual, fwd, ret, t = self.opt.simulate_mj(self.manager.depot, pts_reordered, self.cfg.speed, self.cfg.max_flight_time)
                    first = self.opt.euclidean(self.manager.depot, pts_reordered[0]) if pts_reordered else 0
                    jump = self.opt.euclidean(pts_reordered[-1], pts_reordered[0]) if len(pts_reordered) > 1 else 0
                    self.log.info(f"  UAV{u} DEBUG: first={first:.3f}, fwd={fwd:.3f}, ret={ret:.3f}, jump={jump:.3f}, mj={mj_manual}")
                    # ✅ END DEBUG BLOCK
                else:
                    mj = 0
                    optimized_seq = filtered  # Single point or empty
                
                row_mj.append(mj)
                optimized_round.append(optimized_seq)
            
            mj_matrix.append(row_mj)
            optimized_history.append(optimized_round)
        
        return mj_matrix, optimized_history  # ✅ Return BOTH

    def _make_outputs(self, rates, mj_matrix, history):
        num_uavs = self.cfg.num_uavs
        # Revenue DataFrame
        df_rev = pd.DataFrame(rates, columns=[f"UAV{u}" for u in range(num_uavs)])
        df_rev.insert(0, "negotiation_round", range(len(df_rev)))
        # Sequences DataFrame, with MJ columns interleaved
        seq_dict = {f"UAV{u}": ["-".join(map(str, h[u])) for h in history] for u in range(num_uavs)}
        mj_dict = {f"m_{u}": [mj_matrix[r][u] for r in range(len(history))] for u in range(num_uavs)}
        rounds = [f"{r}" for r in range(len(history))]
        # Interleave columns
        interleaved = []
        for u in range(num_uavs):
            interleaved.append(f"UAV{u}")
            interleaved.append(f"m_{u}")
        # Build DataFrame
        seq_df = pd.DataFrame({**seq_dict, **mj_dict}, index=rounds)
        seq_df.reset_index(inplace=True)  # "index" becomes "negotiation_round"
        seq_df = seq_df[["index"] + interleaved]
        seq_df.rename(columns={"index": "negotiation_round"}, inplace=True)
        # Waypoints DataFrame
        coords = self.manager.waypoints
        values = self.manager.values
        df_wp = pd.DataFrame({
            "Waypoint": list(range(len(coords))),
            "Revenue": values,
            "X": [p[0] for p in coords],
            "Y": [p[1] for p in coords],
        })
        return df_rev, seq_df, df_wp

    def _write_incremental(self, mode_key, strat_name, run_idx, df_rev, df_seq, df_wp):
        max_ft = getattr(self.cfg, "max_flight_time", "")
        speed = getattr(self.cfg, "speed", "")
        rev_dir = os.path.join(self.cfg.results_dir, mode_key, "revenue", self.date_str, self.sim_folder)
        seq_dir = os.path.join(self.cfg.results_dir, mode_key, "sequences", self.date_str, self.sim_folder)
        wp_dir = os.path.join(self.cfg.results_dir, mode_key, "waypoints", self.date_str, self.sim_folder)
        os.makedirs(rev_dir, exist_ok=True)
        os.makedirs(seq_dir, exist_ok=True)
        os.makedirs(wp_dir, exist_ok=True)
        rev_fname = f"UAVs{self.cfg.num_uavs}_GRID{self.cfg.grid_width}_{strat_name}.xlsx"
        seq_fname = f"UAVs{self.cfg.num_uavs}_GRID{self.cfg.grid_width}_{max_ft}_{speed}_{strat_name}_sequences.xlsx"
        wp_fname = f"UAVs{self.cfg.num_uavs}_GRID{self.cfg.grid_width}_waypoints.xlsx"
        rev_path = os.path.join(rev_dir, rev_fname)
        seq_path = os.path.join(seq_dir, seq_fname)
        wp_path = os.path.join(wp_dir, wp_fname)
        self._append_sheet_to_excel(rev_path, df_rev, f"SimRun{run_idx}", index=False)
        self._append_sheet_to_excel(seq_path, df_seq, f"SimRun{run_idx}", index=False)
        self._append_sheet_to_excel(wp_path, df_wp, f"SimRun{run_idx}", index=False)
        print(f"✅ Wrote SimRun{run_idx} to {mode_key}/{strat_name}")

    def _append_sheet_to_excel(self, file_path, df, sheet_name, index=False):
        if os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=index)
        else:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=index)

    # ------------------------------------------------------------
    # CLI overrides helper (call as SimulationRunner.parse_cli_overrides())
    # ------------------------------------------------------------
    def parse_cli_overrides():
        parser = argparse.ArgumentParser(description="Run UAV sequence negotiation simulation")

        parser.add_argument("--num_uavs", type=int, help="Override number of UAVs")
        parser.add_argument("--grid_width", type=int, help="Override grid width")
        parser.add_argument("--grid_height", type=int, help="Override grid height")
        parser.add_argument("--grid_spacing", type=int, help="Override grid spacing")

        parser.add_argument("--speed", type=int, help="Override UAV speed")
        parser.add_argument("--max_flight_time", type=int, help="Override max flight time")

        parser.add_argument("--n_runs", type=int, help="Number of runs")
        parser.add_argument("--seed", type=int, help="Random seed")

        return parser.parse_args()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_uavs", type=int)
    parser.add_argument("--grid_width", type=int)
    parser.add_argument("--grid_height", type=int)
    parser.add_argument("--grid_spacing", type=int)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--max_flight_time", type=int)
    parser.add_argument("--n_runs", type=int)
    args = parser.parse_args()

    # 1. Load from YAML
    cfg = Config.from_yaml("settings.yaml")

    # 2. Apply overrides
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg.override(overrides)

    print("[INFO] Final config:", cfg)

    log = Logger(cfg.results_dir, enabled=cfg.enable_logging)
    runner = SimulationRunner(cfg, log)
    runner.run()
