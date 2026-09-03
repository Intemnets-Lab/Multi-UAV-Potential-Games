"""
Games.py — UAV waypoint-negotiation simulation.

Config          Loads settings.yaml (+ CLI overrides)
Logger          Timestamped file logging
WaypointManager Grid generation, revenues, overlap/clone handling
PreflightChecker  Checks the full waypoint tour fits max_flight_time
PathOptimizer   2-opt TSP solver + "MJ" (feasible loop count) math
UAVAgent        One UAV's route, and its drop/pick negotiation moves
InitialAssigner Uniform round-robin starting assignment
TaskAllocator / NegotiationAllocator   Multi-round negotiation game
SimulationRunner  Orchestrates all runs/modes/strategies, writes Excel


=====================================================================
INITIAL ASSIGNMENT SOURCES (settings.yaml: simulation.initial_assignment_source)
=====================================================================
Controls where each SimRun's starting UAV split comes from. In every
case Games.py only ever needs to know WHICH WAYPOINTS EACH UAV STARTS
WITH - never mj, revenue, or tour order, since its own 2-opt establishes
the tour and mj/revenue get computed fresh during negotiation regardless
of source.

  "random" (default when use_pregenerated_grids is false)
      No external file needed. A fresh grid is generated and split via
      uniform random round-robin (InitialAssigner.uniform).

  "cluster_ga"  (requires use_pregenerated_grids: true)
      Reads {cluster_sequences_dir}/UAVs{N}_GRID{W}_{max_flight_time}_{speed}_cluster_ga_sequences.xlsx
      One sheet per SimRun (see naming note below); only row
      negotiation_round==0 is used. Columns: UAV0, m_0, UAV1, m_1, ...
      (m_j columns are ignored - only UAV{u} sequences are read).

  "kmeans_centroids"  (requires use_pregenerated_grids: true)
      Reads {cluster_sequences_dir}/K-means/centroids_UAV_{N}.xlsx
      One sheet per SimRun. Columns: Cluster_k (maps directly to UAV
      index), Centroid_X, Centroid_Y, Assigned_Waypoints (a Python-list-
      formatted string, e.g. "[3, 17, 42, 91]").

  "generic"  (requires use_pregenerated_grids: true) - RECOMMENDED for
      anyone bringing their own comparison baseline (a different
      clustering method, a greedy heuristic, or anything else), since
      it's the simplest format to produce and needs no knowledge of this
      project's other file conventions:
      Reads {generic_assignment_dir}/UAVs{N}_GRID{W}_initial_assignment.xlsx
      One sheet per SimRun. Columns: UAV0, UAV1, ..., UAV{N-1}, each a
      hyphen-joined list of waypoint IDs (e.g. "3-17-42-91") - the same
      convention used by every sequences file this project writes. No
      mj, no revenue, no negotiation_round column needed.
      A blank, correctly-shaped template for this format can be
      generated with make_generic_assignment_template.py (same folder).

  Sheet naming (cluster_ga / kmeans_centroids / generic, and the
  pregenerated grid file itself): each SimRun's sheet is matched by the
  exact name "SimRun{n}" first; if that's not found, the n-th sheet in
  the workbook is used instead (see _find_simrun_sheet), so files from
  different contributors work regardless of their exact sheet-naming
  convention or casing - the only requirement is that sheets are in
  SimRun order.
=====================================================================
"""
import os
import ast
import random
from random import shuffle
import math
import numpy as np
import argparse
import pandas as pd
from typing import Type
from datetime import datetime
import yaml


class Config:
    """Simulation parameters loaded from settings.yaml (see from_yaml),
    with optional CLI overrides applied afterward (see override)."""

    def __init__(self, data: dict):
        self.results_dir = data["project"]["results_dir"]
        self.irada_benchmark_dir = data["project"].get("IRADA_benchmarking_dir", None)
        self.grids_dir = data["project"].get("grids_dir", "Grids")
        self.cluster_sequences_dir = data["project"].get("cluster_sequences_dir", "Cluster_sequences")
        self.generic_assignment_dir = data["project"].get("generic_assignment_dir", "Generic_initial_assignment")

        sim = data.get("simulation", {})
        self.seed = sim.get("seed", None)
        self.n_runs = sim.get("n_runs", 1)
        self.enable_logging = sim.get("enable_logging", True)
        self.use_pregenerated_grids = sim.get("use_pregenerated_grids", False)
        self.use_kmeans_centroids = sim.get("use_kmeans_centroids", False)

        # initial_assignment_source is the preferred way to pick where the
        # initial UAV split comes from: "random" | "cluster_ga" |
        # "kmeans_centroids" | "generic". If not set explicitly, it's
        # derived from the older use_pregenerated_grids/use_kmeans_centroids
        # booleans so existing settings.yaml files keep working unchanged.
        self.initial_assignment_source = sim.get("initial_assignment_source", None)
        if self.initial_assignment_source is None:
            # Not set explicitly: derive from the older booleans so existing
            # settings.yaml files keep working unchanged.
            if not self.use_pregenerated_grids:
                self.initial_assignment_source = "random"
            elif self.use_kmeans_centroids:
                self.initial_assignment_source = "kmeans_centroids"
            else:
                self.initial_assignment_source = "cluster_ga"
        else:
            # Set explicitly: it's now the single source of truth. Derive
            # use_pregenerated_grids/use_kmeans_centroids FROM it, rather
            # than trusting whatever those booleans happen to say - so
            # setting initial_assignment_source alone is always enough,
            # with no risk of it silently being ignored because a
            # now-secondary boolean wasn't also kept in sync.
            self.use_pregenerated_grids = (self.initial_assignment_source != "random")
            self.use_kmeans_centroids = (self.initial_assignment_source == "kmeans_centroids")

        if self.use_pregenerated_grids and self.n_runs > 100:
            print(f"[INFO] use_pregenerated_grids is on; capping n_runs from {self.n_runs} to 100 "
                  f"(the pregenerated cluster files only go up to SimRun100).")
            self.n_runs = 100
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.grid_width   = data["grid"]["width"]
        self.grid_height  = data["grid"]["height"]
        self.grid_spacing = data["grid"]["spacing"]
        self.zero_prob    = data["grid"]["zero_prob"]
        self.lambda_param = data["grid"].get("lambda", 0.1)

        self.num_uavs = data["uav"]["num_uavs"]
        self.speed    = data["uav"]["speed"]
        self.max_flight_time = data["uav"]["max_flight_time"]

        self.random_revenue = data["revenue"]["random"]
        self.fixed_revenue  = data["revenue"]["fixed_value"]
        self.revenue_min    = data["revenue"]["min"]
        self.revenue_max    = data["revenue"]["max"]

        overlap_cfg = data.get("overlap", {})
        self.overlap = False  # per-run flag, toggled by SimulationRunner
        self.clone_threshold = overlap_cfg.get("clone_threshold", None)
        self.clone_assignment = overlap_cfg.get("clone_assignment", "random")  # "same"/"random"/"balanced"

        for algo, enabled in data["algorithms"].items():
            setattr(self, algo, enabled)

    @classmethod
    def from_yaml(cls, path="settings.yaml"):
        with open(path, "r") as f:
            return cls(yaml.safe_load(f))

    def override(self, overrides: dict):
        """Apply CLI overrides (flattened keys)."""
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.use_pregenerated_grids and self.n_runs > 100:
            print(f"[INFO] use_pregenerated_grids is on; capping n_runs from {self.n_runs} to 100 "
                  f"(the pregenerated cluster files only go up to SimRun100).")
            self.n_runs = 100

    def __repr__(self):
        return f"<Config {self.__dict__}>"


class Logger:
    """Writes timestamped INFO/DEBUG/ERROR lines to one log file.
    No-ops entirely when enabled=False."""

    def __init__(self, output_dir, filename="negotiation_log.txt", enabled: bool = True):
        self.log_path = os.path.join(output_dir, filename)
        self.enabled = enabled
        if enabled:
            os.makedirs(output_dir, exist_ok=True)
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write("=== Negotiation Log ===\n\n")

    def _write(self, level, msg):
        if self.enabled:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] {level}: {msg}\n")

    def info(self, msg):  self._write('INFO', msg)
    def debug(self, msg): self._write('DEBUG', msg)
    def error(self, msg): self._write('ERROR', msg)


class WaypointManager:
    """Handles grid waypoints, revenues, and clone creation/mapping for overlap mode."""

    def __init__(self, config, log, preset_waypoints=None, preset_values=None):
        self.config = config
        self.log = log
        self.depot = (0.0, 0.0)

        if preset_waypoints is not None and preset_values is not None:
            self.waypoints = list(preset_waypoints)
            self.values    = list(preset_values)
            # zero_prob was already applied once, by whoever generated these
            # values originally - re-rolling it here (with a possibly
            # different seed, e.g. a worker process's own seed) would zero
            # out a DIFFERENT set of waypoints than what actually got
            # reported/shared, corrupting which revenue values are real.
        else:
            self.waypoints = self._generate_grid()
            self.values    = self._draw_revenues()
            self._apply_zero_prob()

        self.clone_able = []  # cloneable waypoint indices
        self.clone_map  = {}  # {orig <-> clone}
        if getattr(self.config, "overlap", False):
            self._init_clones_threshold_based()

    def _apply_zero_prob(self):
        """Zero out each revenue independently with probability zero_prob."""
        zp = getattr(self.config, "zero_prob", 0)
        if zp > 0:
            rng = random.Random(getattr(self.config, "seed", None))
            self.values = [v if rng.random() >= zp else 0 for v in self.values]

    def _draw_revenues(self):
        """Random integer revenue per waypoint, in [revenue_min, revenue_max]."""
        min_r = getattr(self.config, "revenue_min", 10)
        max_r = getattr(self.config, "revenue_max", 100)
        rng = random.Random(getattr(self.config, "seed", None))
        return [rng.randint(min_r, max_r) for _ in self.waypoints]

    def redraw_revenues(self):
        """Re-roll all revenues, keeping each clone equal to its original."""
        self.values = self._draw_revenues()
        self._apply_zero_prob()
        for a, b in self.clone_map.items():
            orig, clone = min(a, b), max(a, b)
            self.values[clone] = self.values[orig]

    def _generate_grid(self):
        """width x height grid spaced grid_spacing apart, skipping the depot cell (0, 0)."""
        spacing, W, H = self.config.grid_spacing, self.config.grid_width, self.config.grid_height
        return [(x * spacing, y * spacing)
                for y in range(H) for x in range(W) if not (x == 0 and y == 0)]

    def _init_clones_threshold_based(self):
        """Clone every waypoint whose revenue exceeds clone_threshold (an
        absolute value, or a fraction of the max revenue if in (0, 1))."""
        thr = getattr(self.config, "clone_threshold", None)
        if thr is None:
            return
        cutoff = thr * max(self.values) if 0 < thr < 1 else thr
        num_orig = self.config.grid_width * self.config.grid_height - 1  # exclude depot
        for idx in range(num_orig):
            if self.values[idx] > cutoff:
                clone_idx = len(self.waypoints)
                self.waypoints.append(self.waypoints[idx])
                self.values.append(self.values[idx])
                self.clone_map[idx] = clone_idx
                self.clone_map[clone_idx] = idx
                self.clone_able.extend([idx, clone_idx])
                self.log.info(f"[CLONE] WP{clone_idx} clones WP{idx} (rev={self.values[idx]}, cutoff={cutoff})")

    def ensure_clones_exist_and_wire(self, sequences):
        """In overlap mode, make sure every clone pair has at least one
        owner among `sequences`, assigning orphans per config.clone_assignment."""
        if not getattr(self.config, "overlap", False) or not self.clone_map:
            return sequences
        new_sequences = [list(seq) for seq in sequences]
        n_uavs = len(new_sequences)
        clone_pairs = list(dict.fromkeys(tuple(sorted((a, b))) for a, b in self.clone_map.items()))
        assign_mode = getattr(self.config, "clone_assignment", "same").lower()

        def choose_uav(preferred=None):
            if assign_mode == "random":
                return random.randrange(n_uavs)
            if assign_mode == "balanced":
                return min(range(n_uavs), key=lambda u: len(new_sequences[u]))
            return preferred if preferred is not None else 0  # "same" (or unknown mode)

        for a, b in clone_pairs:
            owner_a = next((u for u, seq in enumerate(new_sequences) if a in seq), None)
            owner_b = next((u for u, seq in enumerate(new_sequences) if b in seq), None)
            if owner_a is not None and owner_b is not None:
                continue
            elif owner_a is None and owner_b is None:
                u = choose_uav()
                new_sequences[u].extend([a, b])
            elif owner_b is None:
                new_sequences[choose_uav(owner_a)].append(b)
            else:
                new_sequences[choose_uav(owner_b)].append(a)
        return new_sequences

    def shared_pool(self):
        """All waypoint indices with positive revenue - the only ones worth assigning."""
        return [i for i, v in enumerate(self.values) if v > 0]


class PreflightChecker:
    """Walks the planned tour once (depot -> every waypoint in order -> depot)
    and checks the total time fits within max_flight_time."""

    def __init__(self, manager: WaypointManager, cfg: Config, log: Logger):
        self.mgr, self.cfg, self.log = manager, cfg, log

    def run(self):
        coords = self.mgr.waypoints
        if len(coords) < 2:
            raise RuntimeError(f"Insufficient waypoints: found {len(coords)}, need at least 2 to form a route.")

        route = [self.mgr.depot] + coords + [self.mgr.depot]
        total_dist = sum(PathOptimizer.euclidean(route[i], route[i + 1]) for i in range(len(route) - 1))
        tour_time = total_dist / self.cfg.speed
        feasible = tour_time <= self.cfg.max_flight_time

        msg = (f"Preflight: {len(coords)} waypoints, tour = {total_dist:.1f}m, "
               f"{tour_time:.1f}s of {self.cfg.max_flight_time:.1f}s budget "
               f"— {'OK' if feasible else 'INFEASIBLE'}")
        print(msg)
        self.log.info(msg)
        return feasible


class PathOptimizer:
    """Pure geometry/route math: distances, MJ (feasible loop count), and a
    2-opt solver for the single-vehicle TSP each UAV needs to solve."""

    @staticmethod
    def euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @classmethod
    def simulate_mj(cls, depot, pts, speed, max_t):
        """MJ and related tour metrics for a *given ordered* list of points.
        Returns (mj, fwd_distance, return_distance, total_time)."""
        n = len(pts)
        if n == 0:
            return 0, 0.0, 0.0, 0.0

        first = cls.euclidean(depot, pts[0])
        fwd   = sum(cls.euclidean(pts[i], pts[i + 1]) for i in range(n - 1))
        ret   = cls.euclidean(pts[-1], depot)
        jump  = cls.euclidean(pts[-1], pts[0]) if n > 1 else 0.0

        denom = fwd + jump
        if denom <= 0:
            mj, total_dist = 1, first + ret
        else:
            mj = max(math.floor((speed * max_t - first - ret + jump) / denom), 1)
            total_dist = first + mj * fwd + (mj - 1) * jump + ret

        return mj, fwd, ret, total_dist / speed

    @classmethod
    def _2opt(cls, n, score_fn):
        """Generic 2-opt hill-climb over orderings of range(n), maximizing
        score_fn(order) (must return a comparable value)."""
        best = list(range(n))
        best_score = score_fn(best)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for k in range(i + 1, n):
                    candidate = best[:i] + best[i:k + 1][::-1] + best[k + 1:]
                    s = score_fn(candidate)
                    if s > best_score:
                        best, best_score, improved = candidate, s, True
        return best

    @classmethod
    def STSPSolver(cls, depot, pts, speed, max_t):
        """Find an ordering of `pts` that maximizes MJ. Runs two 2-opt
        searches with different objectives and keeps whichever finds the
        higher mj:
          - fwd+jump: the smooth, continuous quantity that actually drives
            mj (it's the per-extra-loop cost in simulate_mj's formula) -
            wins most of the time, since it gives 2-opt real gradient to
            climb.
          - simple tour length (depot -> pts -> depot): mj itself is too
            "flat" a signal for 2-opt alone (it's a floor() of a ratio, so
            most single swaps don't change it, and greedy search stalls);
            distance is a decent fallback that occasionally finds a better
            ordering fwd+jump's search missed.
        Returns (best_order, mj)."""
        n = len(pts)
        if n == 0:
            return [], 0

        def fwd_jump_score(order):
            ordered = [pts[i] for i in order]
            fwd = sum(cls.euclidean(ordered[i], ordered[i + 1]) for i in range(n - 1))
            jump = cls.euclidean(ordered[-1], ordered[0]) if n > 1 else 0.0
            return -(fwd + jump)

        def cycle_len_score(order):
            d = cls.euclidean(depot, pts[order[0]])
            for i in range(n - 1):
                d += cls.euclidean(pts[order[i]], pts[order[i + 1]])
            return -(d + cls.euclidean(pts[order[-1]], depot))

        candidates = [cls._2opt(n, fwd_jump_score), cls._2opt(n, cycle_len_score)]
        scored = [(cls.simulate_mj(depot, [pts[i] for i in order], speed, max_t), order)
                  for order in candidates]
        (mj, _, _, tour_time), best_order = max(scored, key=lambda item: (item[0][0], -item[0][3]))
        return best_order, mj

    @classmethod
    def _or_opt_relocate(cls, depot, pts, order, speed, max_t):
        """Or-opt refinement: repeatedly try relocating a single point to a
        different position in the tour (either orientation doesn't matter
        for a lone point), accepting only strictly-improving moves. This is
        a different move than 2-opt's segment reversal - it fixes a single
        badly-placed stop that no reversal can, at the cost of an O(n^2)
        pass. Converges to an or-opt local optimum from `order`."""
        n = len(order)
        order = list(order)
        if n <= 2:
            return order, cls.simulate_mj(depot, [pts[i] for i in order], speed, max_t)[0]

        def mj_of(ordr):
            mj, _, _, t = cls.simulate_mj(depot, [pts[i] for i in ordr], speed, max_t)
            return mj, t

        best_mj, best_t = mj_of(order)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                node = order[i]
                rest = order[:i] + order[i + 1:]
                for j in range(len(rest) + 1):
                    candidate = rest[:j] + [node] + rest[j:]
                    mj, t = mj_of(candidate)
                    if (mj, -t) > (best_mj, -best_t):
                        order, best_mj, best_t = candidate, mj, t
                        improved = True
                        break
                if improved:
                    break
        return order, best_mj

    @classmethod
    def _double_bridge(cls, order, rng):
        """Split into 4 segments A|B|C|D at 3 random cut points, reconnect
        as A|C|B|D. Unlike a 2-opt swap, this can't be undone by any single
        2-opt move - it's what lets ILS escape a 2-opt local optimum
        instead of just re-finding it."""
        n = len(order)
        if n < 8:
            return list(order)
        p1, p2, p3 = sorted(rng.sample(range(1, n), 3))
        return order[:p1] + order[p2:p3] + order[p1:p2] + order[p3:]

    @classmethod
    def ils_polish(cls, depot, pts, speed, max_t, budget_iters=2, seed=None):
        """Iterated Local Search: 2-opt to a local optimum, refine with
        or-opt, then repeatedly double-bridge-perturb + re-optimize,
        keeping the best found. Reliably beats plain STSPSolver (verified:
        0 regressions across 200+ trials, wins up to ~70% of the time on
        larger waypoint sets) - but each call costs several times more than
        a plain STSPSolver call, so this is meant for a BOUNDED number of
        one-shot calls (e.g. once per UAV after negotiation ends), not for
        repeated use inside a hot loop like negotiation's own candidate
        evaluation, where the negotiation's own exploration already
        captures most of the achievable gain and this isn't worth the cost.
        Returns (best_order, mj)."""
        rng = random.Random(seed)
        order, _ = cls.STSPSolver(depot, pts, speed, max_t)
        best_order, best_mj = cls._or_opt_relocate(depot, pts, order, speed, max_t)
        current_order = best_order

        for _ in range(budget_iters):
            perturbed_indices = cls._double_bridge(current_order, rng)
            perturbed_pts = [pts[i] for i in perturbed_indices]
            local_order, _ = cls.STSPSolver(depot, perturbed_pts, speed, max_t)
            local_order, mj = cls._or_opt_relocate(depot, perturbed_pts, local_order, speed, max_t)
            resolved_order = [perturbed_indices[i] for i in local_order]
            if mj > best_mj:
                best_mj, best_order = mj, resolved_order
                current_order = resolved_order

        return best_order, best_mj


class NegotiationReporter:
    """
    Every negotiation log line lives here, so UAVAgent/NegotiationAllocator
    never build a log string themselves - they call a named method (e.g.
    `report.dropped(...)`), keeping the algorithm code readable while the
    log output stays fully detailed and consistently formatted.

    Layout convention used throughout:
        ===== Round N =====        section header (a round)
          > Phase Name              subsection header (a phase within a round)
            UAV0: ...                per-agent line inside a phase
                WP5: gain=+1.20      per-candidate line inside a per-agent block
    """

    def __init__(self, log):
        self.log = log

    # ---------------------------------------------------------------- top level
    def negotiation_start(self, num_uavs, total_rate):
        self.log.info(f"\n{'=' * 60}")
        self.log.info(f"NEGOTIATION START — {num_uavs} UAVs, initial total revenue rate = {total_rate:.4f}")
        self.log.info(f"{'=' * 60}")

    def round_start(self, round_num, agents):
        self.log.info(f"\n===== Round {round_num} =====")
        for a in agents:
            self.log.debug(f"    UAV{a.uid} start: rate={a.revenue_rate():.4f}  seq={a.sequence}")

    def round_end(self, round_num, current_rates, prior_total):
        total = sum(current_rates)
        delta = total - prior_total
        arrow = "▲" if delta > 1e-9 else ("▼" if delta < -1e-9 else "―")
        breakdown = "  ".join(f"UAV{u}={r:.2f}" for u, r in enumerate(current_rates))
        self.log.info(f"  Round {round_num} total revenue rate: {total:.4f}  ({arrow} {delta:+.4f})   [{breakdown}]")

    def rollback(self, round_num, repeated):
        self.log.info(f"  ROLLBACK — revenue fell this round; reverting every UAV to its pre-round state (repeat #{repeated})")

    def rollback_stasis(self, repeated):
        self.log.info(f"  Rollback stasis detected ({repeated}x) — ending negotiation.")

    def converged(self, round_num, patience):
        self.log.info(f"\n{'=' * 60}")
        self.log.info(f"CONVERGED after {round_num} rounds ({patience} rounds with no improvement)")
        self.log.info(f"{'=' * 60}")

    # ---------------------------------------------------------------- phases
    def phase_header(self, name):
        self.log.info(f"  > {name}")

    def agent_order(self, phase_label, agents):
        self.log.debug(f"    (random order for {phase_label}): {[a.uid for a in agents]}")

    def dropped_summary(self, dropped):
        self.log.info(f"    Pool after Drop Phase: {dropped if dropped else '(nothing dropped)'}")

    def picked_summary(self, picked):
        self.log.info(f"    Picked this round: {picked if picked else '(nothing picked)'}")

    def reassigned_summary(self, reassigned):
        self.log.info(f"    Reassigned to previous owner (unclaimed): {reassigned if reassigned else '(none left over)'}")

    # ---------------------------------------------------------------- per-agent decisions
    def drop_candidates(self, uid, candidates):
        if not candidates:
            self.log.debug(f"    UAV{uid}: nothing to consider dropping (empty sequence)")
            return
        self.log.debug(f"    UAV{uid}: {len(candidates)} waypoint(s) considered for dropping:")
        for wp, twin, _, gain in sorted(candidates, key=lambda c: -c[3]):
            twin_txt = f" (+twin WP{twin})" if twin is not None else ""
            self.log.debug(f"        WP{wp}{twin_txt}: gain={gain:+.3f}")

    def no_drop(self, uid, candidates):
        best_wp, best_twin, _, best_gain = max(candidates, key=lambda c: c[3])
        twin_txt = f" (+twin WP{best_twin})" if best_twin is not None else ""
        self.log.debug(f"    UAV{uid}: no drop — best option was WP{best_wp}{twin_txt} at gain={best_gain:+.3f}")

    def dropped(self, uid, wp, twin, gain):
        twin_txt = f" + twin WP{twin}" if twin is not None else ""
        self.log.info(f"    UAV{uid} dropped WP{wp}{twin_txt}   (gain={gain:+.3f})")

    def pick_candidates(self, uid, candidates):
        if not candidates:
            self.log.debug(f"    UAV{uid}: nothing to consider picking")
            return
        self.log.debug(f"    UAV{uid}: {len(candidates)} waypoint(s) considered for picking:")
        for wp, _, gain in sorted(candidates, key=lambda c: -c[2]):
            self.log.debug(f"        WP{wp}: gain={gain:+.3f}")

    def no_pick(self, uid, candidates):
        best_wp, _, best_gain = max(candidates, key=lambda c: c[2])
        self.log.debug(f"    UAV{uid}: no pick — best option was WP{best_wp} at gain={best_gain:+.3f}")

    def picked(self, uid, wp, gain):
        self.log.info(f"    UAV{uid} picked WP{wp}   (gain={gain:+.3f})")

    def reassigned(self, uid, wp):
        self.log.info(f"    UAV{uid} <- WP{wp}  (unclaimed, returned to previous owner)")

    def polished(self, uid, before, after):
        delta = after - before
        tag = f"+{delta:.3f}" if delta > 1e-9 else "no change"
        self.log.info(f"    UAV{uid}: {before:.4f} -> {after:.4f}  ({tag})")


class UAVAgent:
    """A single UAV: holds a sequence of waypoint indices and can compute
    its own revenue rate, or negotiate by dropping/picking waypoints.
    Invariant: self.sequence is always 2-opt-optimized for its current
    waypoint set - every drop or pick re-optimizes before being applied."""

    def __init__(self, uid, manager, optimizer, config, report):
        self.uid = uid
        self.manager = manager
        self.opt = optimizer
        self.cfg = config
        self.report = report
        self.sequence = []

    def revenue_rate(self, seq=None):
        """Revenue rate for `seq` (or the current sequence), AS ORDERED -
        this does not reorder anything. Total revenue of its unique stops,
        times how many loops per second the tour allows. Duplicate
        locations (a clone next to its base) collapse first."""
        seq = self.sequence if seq is None else seq
        if not seq:
            return 0.0
        clone_map = getattr(self.manager, "clone_map", {})
        unique_seq = self.exclude_repeated_locs(seq, clone_map, self.manager.waypoints)
        coords = [self.manager.waypoints[i] for i in unique_seq]

        mj, _, _, tour_time = self.opt.simulate_mj(self.manager.depot, coords, self.cfg.speed, self.cfg.max_flight_time)
        revenue = sum(self.manager.values[wp] for wp in unique_seq)
        return revenue * (mj / tour_time if tour_time > 0 else 0.0)

    def _two_opt(self, seq):
        """2-opt-reorder `seq` for maximum mj. Optimizes the deduplicated
        (unique-location) stops only - a duplicate (a clone right next to
        its base) costs nothing wherever it sits, so leaving it in the
        search just wastes effort and can actually find a WORSE order than
        optimizing the smaller, deduplicated problem directly (verified:
        same mj, different - worse - tour time, when duplicates were left
        in). Each duplicate then gets placed immediately after its kept
        twin, so the full ownership set is still represented in what's
        returned - callers that need mj re-derive it via revenue_rate,
        which dedupes the same way, so the two stay consistent."""
        if len(seq) <= 1:
            return list(seq)
        clone_map = getattr(self.manager, "clone_map", {})
        waypoints = self.manager.waypoints
        unique_seq = self.exclude_repeated_locs(seq, clone_map, waypoints)
        if len(unique_seq) <= 1:
            return list(seq)

        coords = [waypoints[i] for i in unique_seq]
        order, _ = self.opt.STSPSolver(self.manager.depot, coords, self.cfg.speed, self.cfg.max_flight_time)
        optimized_unique = [unique_seq[idx] for idx in order]

        duplicates_by_coord = {}
        for wp in seq:
            if wp not in unique_seq:
                duplicates_by_coord.setdefault(tuple(waypoints[wp]), []).append(wp)

        result = []
        for wp in optimized_unique:
            result.append(wp)
            result.extend(duplicates_by_coord.get(tuple(waypoints[wp]), []))
        return result

    def _final_polish(self):
        """Like _two_opt, but with the stronger (and pricier) ILS+Or-opt
        search instead of plain 2-opt. Meant to run once, after negotiation
        has settled which waypoints this UAV ends up with - at that point
        the extra cost buys real quality, unlike using it as every
        negotiation candidate's evaluator (tested; not worth it there).
        Same dedupe-before-optimize approach as _two_opt, for the same
        reason (a duplicate location wastes search effort and can distort
        the result if left in)."""
        if len(self.sequence) <= 1:
            return list(self.sequence)
        clone_map = getattr(self.manager, "clone_map", {})
        waypoints = self.manager.waypoints
        unique_seq = self.exclude_repeated_locs(self.sequence, clone_map, waypoints)
        if len(unique_seq) <= 1:
            return list(self.sequence)

        coords = [waypoints[i] for i in unique_seq]
        order, _ = self.opt.ils_polish(self.manager.depot, coords, self.cfg.speed, self.cfg.max_flight_time)
        optimized_unique = [unique_seq[idx] for idx in order]

        duplicates_by_coord = {}
        for wp in self.sequence:
            if wp not in unique_seq:
                duplicates_by_coord.setdefault(tuple(waypoints[wp]), []).append(wp)

        result = []
        for wp in optimized_unique:
            result.append(wp)
            result.extend(duplicates_by_coord.get(tuple(waypoints[wp]), []))
        return result

    @staticmethod
    def exclude_repeated_locs(seq, clone_map, waypoints):
        """Keep only the first stop at each physical location (a clone and
        its base share a location, so only one of them survives)."""
        kept, seen = [], set()
        for wp in seq:
            coord = tuple(waypoints[wp])
            if coord not in seen:
                kept.append(wp)
                seen.add(coord)
        return kept

    def drop_waypoint(self, select_mode="greedy"):
        """Drop whichever waypoint (plus its clone twin, if present) and
        re-optimizing helps revenue rate most. Returns (wp, twin, new_seq,
        gain); wp is None if no drop would help. new_seq is already
        2-opt-optimized for the reduced waypoint set."""
        candidates = self._drop_candidates()
        self.report.drop_candidates(self.uid, candidates)
        beneficial = [c for c in candidates if c[3] > 0]
        if not beneficial:
            self.report.no_drop(self.uid, candidates)
            return None, None, list(self.sequence), 0.0
        wp, twin, new_seq, gain = (random.choice(beneficial) if select_mode == "random"
                                    else max(beneficial, key=lambda c: c[3]))
        self.report.dropped(self.uid, wp, twin, gain)
        self.sequence = new_seq
        return wp, twin, list(self.sequence), gain

    def _drop_candidates(self):
        """Every (waypoint, twin_or_None, resulting_2opt_seq, gain)
        evaluated for dropping - including non-beneficial ones, so the log
        can show what was considered even when nothing helped. Callers
        only ACT on the ones with gain > 0."""
        current_seq = list(self.sequence)
        current_rate = self.revenue_rate()  # self.sequence is already 2-opt-optimal
        clone_map = getattr(self.manager, "clone_map", {})
        inv_clone_map = {v: k for k, v in clone_map.items()}

        candidates = []
        for wp in current_seq:
            twin = clone_map.get(wp, inv_clone_map.get(wp))
            reduced_seq = [x for x in current_seq if x != wp and x != twin]
            optimized_seq = self._two_opt(reduced_seq)
            gain = self.revenue_rate(optimized_seq) - current_rate
            clone_wp = twin if twin in current_seq else None
            candidates.append((wp, clone_wp, optimized_seq, gain))
        return candidates

    def pick_waypoint(self, pool, select_mode="greedy"):
        """Append whichever pool waypoint to the end of the sequence and
        re-optimizing helps revenue rate most. Returns (wp, None, new_seq,
        gain); wp is None if no pick would help. new_seq is already
        2-opt-optimized for the enlarged waypoint set."""
        candidates = self._pick_candidates(pool)
        self.report.pick_candidates(self.uid, candidates)
        beneficial = [c for c in candidates if c[2] > 0]
        if not beneficial:
            self.report.no_pick(self.uid, candidates)
            return None, None, list(self.sequence), 0.0
        wp, new_seq, gain = (random.choice(beneficial) if select_mode == "random"
                              else max(beneficial, key=lambda c: c[2]))
        self.report.picked(self.uid, wp, gain)
        self.sequence = new_seq
        return wp, None, new_seq, gain

    def _pick_candidates(self, pool):
        """Every (waypoint, resulting_2opt_seq, gain) evaluated from
        appending a pool waypoint to the end of the sequence and
        re-optimizing via 2-opt - including non-beneficial ones (a clone
        appended alongside its already-present base yields zero gain,
        since they collapse to one stop, so it'll show up here at gain=0)."""
        current_seq = list(self.sequence)
        current_rate = self.revenue_rate()  # self.sequence is already 2-opt-optimal

        candidates = []
        for wp, _owner in pool:
            optimized_seq = self._two_opt(current_seq + [wp])
            gain = self.revenue_rate(optimized_seq) - current_rate
            candidates.append((wp, optimized_seq, gain))
        return candidates


class InitialAssigner:
    """Initial waypoint-to-UAV assignment: uniform random round-robin
    split. K-means-based initial assignment is handled separately, only
    via the pregenerated-grids path (see _load_pregenerated_clusters /
    _load_kmeans_centroids) - it's sourced from externally-provided
    cluster/centroid files there, never computed by this class.

    Every call reseeds its own RNG from config.seed, so repeated calls
    (e.g. once per strategy) all start from the identical split - that's
    what makes comparing strategies apples-to-apples."""

    def __init__(self, config: Config, manager: WaypointManager = None, logger: Logger = None):
        self.cfg = config
        self.manager = manager
        self.log = logger
        self.num_uavs = config.num_uavs

    def _fresh_rng(self):
        return random.Random(self.cfg.seed if self.cfg.seed is not None else None)

    def uniform(self, waypoint_pool):
        """Shuffle waypoint_pool, then deal it round-robin across UAVs."""
        sequences = [[] for _ in range(self.num_uavs)]
        tasks = list(waypoint_pool)
        self._fresh_rng().shuffle(tasks)
        for i, wp in enumerate(tasks):
            sequences[i % self.num_uavs].append(wp)

        if self.log:
            self.log.debug(f"[INIT] {len(tasks)} waypoints across {self.num_uavs} UAVs: {sequences}")
        return sequences


class TaskAllocator:
    """Base class for negotiation allocators: builds UAVAgents and gives
    each one a starting sequence."""

    def __init__(self, manager: WaypointManager, config: Config,
                 logger: Logger, optimizer: Type[PathOptimizer] = PathOptimizer):
        self.manager = manager
        self.cfg     = config
        self.log     = logger
        self.opt     = optimizer
        self.report  = NegotiationReporter(logger)

    def _setup_agents(self, initial_sequences=None):
        if initial_sequences is None:
            initial_sequences = InitialAssigner(self.cfg, self.manager, self.log).uniform(self.manager.shared_pool())

        # In overlap mode, make sure every clone pair is actually assigned to a UAV
        if getattr(self.cfg, "overlap", False) and self.manager.clone_map:
            initial_sequences = self.manager.ensure_clones_exist_and_wire(initial_sequences)

        agents = [UAVAgent(u, self.manager, self.opt, self.cfg, self.report) for u in range(self.cfg.num_uavs)]
        for u, seq in enumerate(initial_sequences):
            agents[u].sequence = agents[u]._two_opt(list(seq))
        return agents


class NegotiationAllocator(TaskAllocator):
    """
    UAV-centric negotiation: each round, every UAV can drop a waypoint it no
    longer wants and pick one it does; leftovers get reinserted wherever
    they help most. Runs until revenue stops improving or max_rounds hits.
    """

    def __init__(self, manager, config, log, drop_select, pick_select, max_rounds=100, patience=5, on_round=None):
        super().__init__(manager, config, log)
        self.drop_select = drop_select.lower()
        self.pick_select = pick_select.lower()
        self.max_rounds  = max_rounds
        self.patience    = patience
        self.on_round    = on_round or (lambda round_num, total_rate: None)

    def _maybe_shuffled(self, agents, phase_label):
        """Copy of `agents`, shuffled if this run uses random negotiation order."""
        ordered = list(agents)
        if self.cfg.randomize_sequence:
            shuffle(ordered)
            self.report.agent_order(phase_label, ordered)
        return ordered

    def _drop_phase(self, agents):
        """Ask each UAV whether it wants to drop a waypoint (and its clone
        twin, if any). Returns the pool of (waypoint, previous_owner_uid)."""
        self.report.phase_header("Drop Phase")
        pool, dropped = [], []
        for a in self._maybe_shuffled(agents, "Drop"):
            wp, twin, new_seq, gain = a.drop_waypoint(self.drop_select)
            if wp is not None:
                a.sequence = new_seq
                pool += [(wp, a.uid)] + ([(twin, a.uid)] if twin is not None else [])
                dropped += [wp] + ([twin] if twin is not None else [])
        self.report.dropped_summary(dropped)
        return pool

    def _pick_phase(self, agents, pool):
        """Let each UAV try to pick one waypoint from the pool. Returns
        whatever's left in the pool for the reassignment phase."""
        self.report.phase_header("Pick Phase")
        picked = []
        for a in self._maybe_shuffled(agents, "Pick"):
            if not pool:
                break
            wp, _, new_seq, gain = a.pick_waypoint(pool, self.pick_select)
            if wp is not None:
                a.sequence = new_seq
                picked.append(wp)
                pool = [(w, u) for (w, u) in pool if w != wp]
        self.report.picked_summary(picked)
        return pool

    def _reassign_leftover_pool(self, agents, pool):
        """Any waypoints nobody picked get appended to their previous
        owner's sequence and 2-opt-reoptimized (same as a pick - this
        always happens, even if the resulting rate is still a loss;
        there's no "declining" a leftover waypoint)."""
        self.report.phase_header("Reassignment Phase")
        reassigned = []
        for wp, owner_uid in pool:
            agent = next((a for a in agents if a.uid == owner_uid), None)
            if agent is None:
                continue
            agent.sequence = agent._two_opt(list(agent.sequence) + [wp])
            self.report.reassigned(agent.uid, wp)
            reassigned.append(wp)
        self.report.reassigned_summary(reassigned)

    def _apply_rollback_if_needed(self, round_num, agents, rates, history, prior_total,
                                   rollback_snapshots, repeat_limit):
        """If total revenue dropped this round, roll every agent back to its
        pre-round sequence, AND correct rates[-1]/history[-1] to match -
        they were already recorded with this round's (now-discarded)
        attempt, before we knew it would be rolled back. Returns
        (total_revenue_rate, stop_negotiation)."""
        current_rates = rates[-1]
        prior_state = history[-2]  # state before this round's actions
        if sum(current_rates) >= prior_total - 1e-6:
            rollback_snapshots.clear()
            return sum(current_rates), False

        snapshot = tuple(tuple(seq) for seq in prior_state)
        rollback_snapshots.append(snapshot)
        if len(rollback_snapshots) > repeat_limit:  # keep only recent history
            rollback_snapshots.pop(0)
        repeated = rollback_snapshots.count(snapshot)
        self.report.rollback(round_num, repeated)

        for idx, a in enumerate(agents):
            a.sequence = prior_state[idx]
        rates[-1] = list(rates[-2])
        history[-1] = [list(seq) for seq in prior_state]

        if repeated >= repeat_limit:
            self.report.rollback_stasis(repeated)
            return prior_total, True
        return prior_total, False

    def _check_convergence(self, rates, round_num, stagnant_rounds, eps):
        """Count rounds with no meaningful change in total revenue rate.
        Returns (updated_stagnant_rounds, converged)."""
        if round_num <= 1 or abs(sum(rates[-1]) - sum(rates[-2])) >= eps:
            return 0, False
        stagnant_rounds += 1
        if stagnant_rounds >= self.patience:
            self.report.converged(round_num, self.patience)
            return stagnant_rounds, True
        return stagnant_rounds, False

    def _final_polish(self, agents, rates, history):
        """Once negotiation has settled which waypoints each UAV ends up
        with, spend more effort finding the best tour order for that
        settled set via ILS+Or-opt (see PathOptimizer.ils_polish) - a
        bounded, one-shot cost per UAV, unlike using it as every
        negotiation candidate's evaluator throughout (tested; the
        negotiation's own waypoint-combination search already captures
        most of the achievable gain there, so it isn't worth the cost)."""
        self.report.phase_header("Final Polish (ILS + Or-opt)")
        for a in agents:
            before = a.revenue_rate()
            a.sequence = a._final_polish()
            after = a.revenue_rate()
            self.report.polished(a.uid, before, after)
        rates[-1] = [a.revenue_rate() for a in agents]
        history[-1] = [list(a.sequence) for a in agents]

    def allocate(self, initial_sequences=None):
        """Run the negotiation to convergence (or max_rounds).
        Returns (rates, history): per-round revenue rates and sequence
        snapshots, both 0-indexed with round 0 = the initial (pre-negotiation)
        state, so rates[k] and history[k] always describe the same state."""
        agents = self._setup_agents(initial_sequences)
        history = [[list(a.sequence) for a in agents]]
        rates = [[a.revenue_rate() for a in agents]]
        rollback_snapshots = []
        ROLLBACK_REPEAT_LIMIT, CONVERGENCE_EPS = 5, 1e-4
        stagnant_rounds = 0

        total_rate = sum(rates[0])
        self.report.negotiation_start(len(agents), total_rate)

        for round_num in range(1, self.max_rounds + 1):
            self.report.round_start(round_num, agents)

            pool = self._drop_phase(agents)
            pool = self._pick_phase(agents, pool)
            self._reassign_leftover_pool(agents, pool)

            current_rates = [a.revenue_rate() for a in agents]
            rates.append(current_rates)
            history.append([list(a.sequence) for a in agents])
            self.report.round_end(round_num, current_rates, total_rate)
            self.on_round(round_num, sum(current_rates))

            total_rate, should_stop = self._apply_rollback_if_needed(
                round_num, agents, rates, history, total_rate, rollback_snapshots, ROLLBACK_REPEAT_LIMIT
            )
            if should_stop:
                break

            stagnant_rounds, converged = self._check_convergence(rates, round_num, stagnant_rounds, CONVERGENCE_EPS)
            if converged:
                break

        self._final_polish(agents, rates, history)
        return rates, history


import multiprocessing
import pickle
import contextlib
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED


def _detect_resources():
    """Detect available compute for dynamic scaling. CPU core count always
    works. GPU detection needs the optional `torch` package - if it's not
    installed, or no CUDA device is found, GPU usage is simply skipped;
    everything still runs correctly, just CPU-only. (See the note on
    SimulationRunner about why GPU compute isn't wired in beyond detection
    for this workload.)"""
    cpu_count = os.cpu_count() or 1
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return cpu_count, gpu_name


def _compute_mj_matrix(manager, cfg, history):
    """For every round's history, dedupe each UAV's route (a clone right
    next to its base collapses to one stop) and record its MJ. Deliberately
    does NOT re-run STSPSolver here - the stored sequence is already
    2-opt-optimal (see UAVAgent._two_opt/_final_polish), and 2-opt is a
    local search sensitive to its starting order, so re-solving from a
    different arrangement could converge to a different local optimum
    than what actually drove revenue_rate during negotiation - silently
    making this report inconsistent with the revenue sheet (verified this
    happening). Using the stored order directly, the same way revenue_rate
    does, guarantees the two always match, by construction."""
    clone_map = getattr(manager, "clone_map", {})
    mj_matrix, optimized_history = [], []

    for sequences in history:
        row_mj, optimized_round = [], []
        for seq in sequences:
            filtered = UAVAgent.exclude_repeated_locs(seq, clone_map, manager.waypoints)
            if filtered:
                coords = [manager.waypoints[i] for i in filtered]
                mj, _, _, _ = PathOptimizer.simulate_mj(manager.depot, coords, cfg.speed, cfg.max_flight_time)
            else:
                mj = 0
            row_mj.append(mj)
            optimized_round.append(filtered)
        mj_matrix.append(row_mj)
        optimized_history.append(optimized_round)

    return mj_matrix, optimized_history


def _cluster_assignment(manager, assigner):
    """Which UAV each waypoint was initially assigned to (uniform random
    round-robin split - see InitialAssigner.uniform). None for zero-revenue
    waypoints, which aren't assigned to anyone. Column is still called
    "Cluster" in the output sheet for consistency with the pregenerated-
    grids path, where the assignment genuinely is a K-means cluster."""
    assignment = [None] * len(manager.waypoints)
    for uid, cluster in enumerate(assigner.uniform(manager.shared_pool())):
        for wp in cluster:
            assignment[wp] = uid
    return assignment


def _cluster_assignment_from_sequences(manager, sequences):
    """Same shape as _cluster_assignment, but derived directly from an
    already-decided set of initial UAV sequences instead of re-running
    K-means. Used for pregenerated-grid mode, where there's exactly ONE
    real clustering (loaded from file, shared by every strategy for this
    run) - deriving the report from it directly guarantees the "Cluster"
    column exactly matches what negotiation actually used, rather than a
    second, independent K-means call that could in principle land
    differently."""
    assignment = [None] * len(manager.waypoints)
    for uid, seq in enumerate(sequences):
        for wp in seq:
            assignment[wp] = uid
    return assignment


def _find_simrun_sheet(xl, run_idx):
    """Finds the sheet for a given 1-indexed SimRun number in an already-
    open pd.ExcelFile, tolerant of naming differences across different
    contributors' files ('SimRun1', 'Sim_Run_1', 'Run 1', 'Simulation1',
    etc.) - matches the exact 'SimRun{run_idx}' name first (the existing
    convention), and falls back to positional order (the run_idx-th sheet
    in the workbook) if that exact name isn't found. This means ANY
    consistent sequential naming works with zero configuration, as long
    as sheets are in SimRun order - not requiring an exact string match is
    the whole point, since different people will name these differently."""
    exact = f"SimRun{run_idx}"
    if exact in xl.sheet_names:
        return exact
    if 1 <= run_idx <= len(xl.sheet_names):
        return xl.sheet_names[run_idx - 1]
    raise ValueError(
        f"Can't find a sheet for SimRun{run_idx}: no sheet named '{exact}' and the "
        f"workbook only has {len(xl.sheet_names)} sheet(s). Sheets found: {xl.sheet_names}")


def _load_pregenerated_grid(cfg, run_idx):
    """Load one SimRun's stored (waypoint, revenue, x, y) grid from
    Grids/NonOverlap/... - this is the SOLE source of grid data for BOTH
    modes. Overlap mode does NOT read Grids/Overlap/ at all: its clones
    get generated the same way WaypointManager already does it for
    freshly-random grids (revenue > clone_threshold -> append a clone with
    identical coordinates/revenue), which is both simpler and guarantees
    the clone-to-parent K-means cluster mapping is unambiguous (see
    _pregenerated_initial_sequences)."""
    path = os.path.join(cfg.grids_dir, "NonOverlap", f"UAVs{cfg.num_uavs}_GRID{cfg.grid_width}_waypoints.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pregenerated grid file not found: {path}\n"
            f"(use_pregenerated_grids is on; expected one grid file per num_uavs from 3-10 "
            f"under {cfg.grids_dir}/NonOverlap/)")
    xl = pd.ExcelFile(path)
    df = xl.parse(_find_simrun_sheet(xl, run_idx)).sort_values("Waypoint")
    waypoints = list(zip(df["X"].tolist(), df["Y"].tolist()))
    values = df["Revenue"].tolist()
    return waypoints, values


def _load_pregenerated_clusters(cfg, run_idx):
    """Load one SimRun's initial K-means UAV split from
    Cluster_sequences/... - only negotiation_round == 0 is used (the pure
    K-means split, before any of the GA process that produced the rest of
    that file's rounds); this becomes round 0 of OUR negotiation too.

    NOTE: this file is actually K-means+GA output, not raw K-means - the
    GA has already optimized the tour order (and possibly membership)
    before round 0 is ever recorded. Negotiating on top of an
    already-GA-optimized starting point makes any revenue-rate comparison
    against that same GA's own final result unfair, since our result only
    has to improve on GA's work, not reproduce K-means' original weaker
    starting point. See _load_kmeans_centroids for the corrected source."""
    path = os.path.join(cfg.cluster_sequences_dir,
                         f"UAVs{cfg.num_uavs}_GRID{cfg.grid_width}_{cfg.max_flight_time}_{cfg.speed}"
                         f"_cluster_ga_sequences.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pregenerated cluster file not found: {path}\n"
            f"(use_pregenerated_grids is on; expected one cluster file per num_uavs from 3-10 "
            f"under {cfg.cluster_sequences_dir}/)")
    xl = pd.ExcelFile(path)
    df = xl.parse(_find_simrun_sheet(xl, run_idx))
    row = df[df["negotiation_round"] == 0].iloc[0]

    sequences = []
    for u in range(cfg.num_uavs):
        raw = row.get(f"UAV{u}")
        if pd.isna(raw) or not str(raw).strip():
            sequences.append([])
        else:
            sequences.append([int(x) for x in str(raw).split("-")])
    return sequences


def _load_kmeans_centroids(cfg, run_idx):
    """Load one SimRun's RAW K-means cluster membership (no GA
    optimization applied) from Cluster_sequences/K-means/
    centroids_UAV_{num_uavs}.xlsx - Cluster_k maps directly to UAV index.
    Unlike _load_pregenerated_clusters, this file carries only cluster
    membership (no tour order, no mj) - our own _two_opt establishes the
    initial tour order from scratch, exactly as it would for any freshly
    assigned cluster, so negotiation starts from a genuinely unoptimized
    baseline rather than one the GA already improved."""
    path = os.path.join(cfg.cluster_sequences_dir, "K-means", f"centroids_UAV_{cfg.num_uavs}.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"K-means centroids file not found: {path}\n"
            f"(use_kmeans_centroids is on; expected one file per num_uavs from 3-10 "
            f"under {cfg.cluster_sequences_dir}/K-means/)")
    xl = pd.ExcelFile(path)
    df = xl.parse(_find_simrun_sheet(xl, run_idx))

    sequences = [[] for _ in range(cfg.num_uavs)]
    for row in df.itertuples():
        uav_idx = int(row.Cluster_k)
        if not (0 <= uav_idx < cfg.num_uavs):
            raise ValueError(
                f"centroids_UAV_{cfg.num_uavs}.xlsx SimRun{run_idx}: Cluster_k={uav_idx} "
                f"is out of range for {cfg.num_uavs} UAVs.")
        raw = row.Assigned_Waypoints
        waypoints = ast.literal_eval(str(raw)) if not isinstance(raw, list) else raw
        sequences[uav_idx] = [int(w) for w in waypoints]
    return sequences


def _load_generic_initial_assignment(cfg, run_idx):
    """Load one SimRun's initial UAV split from an easy-to-produce external
    baseline file at {generic_assignment_dir}/UAVs{num_uavs}_GRID{grid_width}_initial_assignment.xlsx.

    Format (deliberately the simplest thing any baseline algorithm can
    produce): one sheet per SimRun (name-flexible - see
    _find_simrun_sheet), columns UAV0..UAV{num_uavs-1}, each a hyphen-
    joined list of waypoint IDs, e.g. "3-17-42-91" - the SAME convention
    used by every sequences file in this codebase, so there's nothing new
    to learn. No mj, no revenue, no negotiation_round, no special
    structure: Games.py only ever needs to know which waypoints each UAV
    starts with. Its own _two_opt establishes tour order from there, and
    mj/revenue get derived fresh during negotiation, exactly as they
    would for a random initial split - this is intentionally the lowest-
    friction path for anyone bringing their own comparison baseline
    (K-means, greedy, or anything else) without needing to match this
    project's internal file conventions."""
    path = os.path.join(cfg.generic_assignment_dir,
                         f"UAVs{cfg.num_uavs}_GRID{cfg.grid_width}_initial_assignment.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Generic initial-assignment file not found: {path}\n"
            f"(initial_assignment_source is 'generic'; expected one file per num_uavs "
            f"under {cfg.generic_assignment_dir}/, with columns UAV0..UAV{{N-1}} of "
            f"hyphen-joined waypoint IDs, one sheet per SimRun - see the template "
            f"generator script for an easy way to build this.)")
    xl = pd.ExcelFile(path)
    df = xl.parse(_find_simrun_sheet(xl, run_idx))
    if len(df) == 0:
        raise ValueError(f"{path}, SimRun{run_idx}: sheet has no data rows.")
    row = df.iloc[0]

    sequences = []
    for u in range(cfg.num_uavs):
        col = f"UAV{u}"
        if col not in df.columns:
            raise ValueError(
                f"{path}, SimRun{run_idx}: missing column '{col}' - expected columns "
                f"UAV0..UAV{cfg.num_uavs - 1} (one per UAV).")
        raw = row[col]
        if pd.isna(raw) or not str(raw).strip():
            sequences.append([])
        else:
            sequences.append([int(x) for x in str(raw).split("-")])
    return sequences


def _pregenerated_initial_sequences(cfg, manager, run_idx):
    """Initial UAV assignment for pregenerated-grid mode. Source is
    controlled by cfg.initial_assignment_source: "cluster_ga" (Thais's
    K-means+GA sequences), "kmeans_centroids" (Thais's raw K-means
    membership), or "generic" (anyone's easy-to-produce baseline file -
    see _load_generic_initial_assignment). For Overlap, each clone
    WaypointManager generated is appended to its parent's SAME UAV - a
    clone always shares its parent's cluster, per the thesis's
    benchmarking convention - so ensure_clones_exist_and_wire (called
    later, inside _setup_agents) has nothing left to do: every clone
    pair already has a shared owner by construction."""
    source = getattr(cfg, "initial_assignment_source", "cluster_ga")
    if source == "generic":
        sequences = _load_generic_initial_assignment(cfg, run_idx)
    elif source == "kmeans_centroids":
        sequences = _load_kmeans_centroids(cfg, run_idx)
    elif source == "cluster_ga":
        sequences = _load_pregenerated_clusters(cfg, run_idx)
    else:
        raise ValueError(
            f"Unknown initial_assignment_source: {source!r}. "
            f"Expected one of: 'cluster_ga', 'kmeans_centroids', 'generic'.")

    clone_map = getattr(manager, "clone_map", {})
    if clone_map:
        owner_of = {}
        for uid, seq in enumerate(sequences):
            for wp in seq:
                owner_of[wp] = uid
        clone_pairs = list(dict.fromkeys(tuple(sorted((a, b))) for a, b in clone_map.items()))
        for orig, clone_idx in clone_pairs:
            owner = owner_of.get(orig)
            if owner is not None:
                sequences[owner].append(clone_idx)
            else:
                # Shouldn't happen: a clone only exists because its parent's
                # revenue exceeded clone_threshold, and clone_threshold > 0,
                # so the parent always had positive revenue and was always
                # part of the loaded K-means split. Fall back to leaving it
                # unowned - ensure_clones_exist_and_wire will pick it up per
                # cfg.clone_assignment rather than silently losing it.
                pass
    return sequences


def _make_outputs(manager, cfg, rates, mj_matrix, history):
    num_uavs = cfg.num_uavs

    df_rev = pd.DataFrame(rates, columns=[f"UAV{u}" for u in range(num_uavs)])
    df_rev.insert(0, "negotiation_round", range(len(df_rev)))

    # Sequences DataFrame, with MJ columns interleaved: UAV0, m_0, UAV1, m_1, ...
    seq_dict = {f"UAV{u}": ["-".join(map(str, h[u])) for h in history] for u in range(num_uavs)}
    mj_dict = {f"m_{u}": [mj_matrix[r][u] for r in range(len(history))] for u in range(num_uavs)}
    interleaved = [col for u in range(num_uavs) for col in (f"UAV{u}", f"m_{u}")]
    seq_df = pd.DataFrame({**seq_dict, **mj_dict}, index=[str(r) for r in range(len(history))])
    seq_df.reset_index(inplace=True)
    seq_df = seq_df[["index"] + interleaved].rename(columns={"index": "negotiation_round"})

    return df_rev, seq_df


def _excel_engine():
    """xlsxwriter is preferred (measured ~1.47x faster than openpyxl for
    the batched-write pattern this file uses, verified identical output) -
    but it's an optional package, not currently required by this project,
    so fall back to openpyxl if it's not installed rather than crash.
    Resolved once and cached."""
    if not hasattr(_excel_engine, "_cached"):
        try:
            __import__("xlsxwriter")
            _excel_engine._cached = "xlsxwriter"
        except ImportError:
            _excel_engine._cached = "openpyxl"
    return _excel_engine._cached


def _write_all_sheets(file_path, sheets_by_name):
    """Write every accumulated sheet in ONE session (see _excel_engine for
    which library). Deliberately not incremental append in the first
    place: openpyxl's "append mode" re-reads and re-saves the ENTIRE
    existing file on every call, so its cost grows with total file size -
    measured 100 sequential appends to one growing file at ~54s, versus
    <1s to write the identical 100 sheets in one session from memory.
    Callers gather sheets first and call this periodically instead of
    writing on every single completion."""
    with pd.ExcelWriter(file_path, engine=_excel_engine()) as writer:
        for sheet_name in sorted(sheets_by_name, key=lambda s: int(s.replace("SimRun", ""))):
            sheets_by_name[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)


def _make_waypoints_sheet(cfg, manager, cluster_assignment):
    coords, values = manager.waypoints, manager.values
    return pd.DataFrame({
        "Waypoint": list(range(len(coords))),
        "Revenue": values,
        "X": [p[0] for p in coords],
        "Y": [p[1] for p in coords],
        "Cluster": pd.array(cluster_assignment, dtype="Int64"),
    })


def _thesis_code(mode_key, strat_name):
    """4-letter configuration code matching the thesis's Table 6.2 naming
    convention: [Game][MarketVisit][DropMode][PickMode] - e.g. NSGR is
    NonOverlap/Sequential/Greedy-drop/Random-pick, ORRG is
    Overlap/Random/Random-drop/Greedy-pick. Verified against both worked
    examples from the thesis text before use."""
    game = "N" if mode_key == "NonOverlap" else "O"
    drop_pick, order = strat_name.replace("Mode", "").split("_")
    drop, pick = drop_pick[0], drop_pick[1]
    visit = "S" if order == "Sequential" else "R"
    return f"{game}{visit}{drop}{pick}"


def _current_cpu_core():
    """Best-effort snapshot of which CPU core this process is on RIGHT NOW
    - not a guarantee it stayed there the whole task, since the OS
    scheduler can and does migrate processes between cores mid-run. Needs
    the optional `psutil` package; returns None if it's not installed or
    the platform doesn't support the call, so this never blocks a run."""
    try:
        import psutil
        return psutil.Process().cpu_num()
    except Exception:
        return None


def _write_combo_temp(sim_dir, strat_name, run_idx, df_rev, df_seq):
    """Immediately persist one combo's result to its own small, uniquely
    named temp file - pickle, not xlsx, since it's much cheaper to write
    and read, and the real xlsx conversion happens separately in the main
    process (see SimulationRunner._flush_pending_excel). Written the
    instant a negotiation finishes, before this worker even returns - a
    completed combo can represent hours of compute at production scale,
    so getting it safely onto disk immediately (rather than leaving it
    sitting only in memory until a periodic flush) closes off the one
    real gap in write-buffering: a main-process crash between flushes
    losing already-finished work. Writes to a .partial name and atomically
    renames it, so a kill mid-write can never leave a corrupt temp file."""
    temp_dir = os.path.join(sim_dir, ".combo_temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{strat_name}_{run_idx}.pkl")
    partial_path = temp_path + ".partial"
    with open(partial_path, "wb") as f:
        pickle.dump((df_rev, df_seq), f)
    os.replace(partial_path, temp_path)
    return temp_path


def _run_one_combo(payload):
    """Run ONE (mode, strategy, run_idx) combination end-to-end: build the
    manager/allocator, negotiate, compute outputs, write Excel, write a
    private temp log file (the caller merges it into the mode's real log
    afterward - can't have several processes writing one file at once).

    Module-level and only picklable arguments by design: ProcessPoolExecutor
    uses the "spawn" start method on Windows (and optionally elsewhere),
    which re-imports this file fresh in each worker process rather than
    forking - closures and bound methods can't survive that, so this can't
    be a nested function or a SimulationRunner method."""
    (cfg, base_waypoints, base_values, mode_key, mode_flag, strat_name,
     drop_select, pick_select, randomize_sequence, run_idx, results_dir,
     sim_folder, worker_seed, progress_dict) = payload

    pid = os.getpid()
    start_wall = datetime.now()
    start_perf = time.perf_counter()

    cfg.overlap = mode_flag
    cfg.randomize_sequence = randomize_sequence
    if worker_seed is not None:
        cfg.seed = worker_seed
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    # else: leave the global RNG's own OS-entropy seeding alone - this run
    # has no fixed seed, so each combo should be independently random too.

    sim_dir = os.path.join(results_dir, mode_key, sim_folder)
    log_filename = f".worker_{strat_name}_{run_idx}.tmp"
    log = Logger(sim_dir, filename=log_filename, enabled=cfg.enable_logging)
    log.info(f"=== {mode_key} / {strat_name} / SimRun {run_idx} ===")

    manager = WaypointManager(cfg, log, preset_waypoints=list(base_waypoints), preset_values=list(base_values))

    if cfg.use_pregenerated_grids:
        initial_sequences = _pregenerated_initial_sequences(cfg, manager, run_idx)
    else:
        # Random initial assignment (matches the thesis's "random initial
        # assignment" market algorithms - uniform round-robin, not
        # geographic clustering). K-means-based initial assignment only
        # happens via the pregenerated-grids path above, sourced from
        # externally-provided cluster/centroid files, never computed here.
        assigner = InitialAssigner(cfg, manager)
        pool = manager.shared_pool()
        clone_map = getattr(manager, "clone_map", {})
        initial_sequences = [UAVAgent.exclude_repeated_locs(seq, clone_map, manager.waypoints)
                              for seq in assigner.uniform(pool)]

    if progress_dict is not None:
        # Parallel mode: several processes would garble each other's output
        # if they all printed directly, so just report state into a shared
        # dict - the main process's dashboard reads it and does the printing.
        # Key includes run_idx: with the whole simulation batched together,
        # the SAME strategy from DIFFERENT SimRuns can be running at once.
        progress_key = f"R{run_idx}/{_thesis_code(mode_key, strat_name)}"

        def on_round(round_num, total_rate):
            progress_dict[progress_key] = round_num
    else:
        # Serial mode: this IS the only thing running right now, so it's
        # safe to print a live-updating line directly.
        def on_round(round_num, total_rate):
            print(f"\r    round {round_num:>4}  |  total revenue rate = {total_rate:>10.2f}" + " " * 10,
                  end="", flush=True)

    alloc = NegotiationAllocator(manager, cfg, log, drop_select, pick_select, on_round=on_round)
    rates, history = alloc.allocate(initial_sequences)
    if progress_dict is not None:
        progress_dict[progress_key] = "done"

    mj_matrix, optimized_history = _compute_mj_matrix(manager, cfg, history)
    df_rev, df_seq = _make_outputs(manager, cfg, rates, mj_matrix, optimized_history)
    # Write this combo's OWN temp file immediately (uniquely named per
    # strategy+run, so - unlike the shared final xlsx - there's no
    # multi-worker contention here to worry about). Getting the result
    # safely onto disk now, rather than only in memory until the main
    # process's next periodic flush, is the whole point: see
    # _write_combo_temp's docstring.
    temp_path = _write_combo_temp(sim_dir, strat_name, run_idx, df_rev, df_seq)

    end_wall = datetime.now()
    end_perf = time.perf_counter()
    trace = {
        "SimRun": run_idx,
        "Config": _thesis_code(mode_key, strat_name),  # thesis Table 6.2 naming convention
        "Worker_PID": pid,
        "CPU_Core": _current_cpu_core(),
        "Start_Time": start_wall.isoformat(sep=" ", timespec="milliseconds"),
        "End_Time": end_wall.isoformat(sep=" ", timespec="milliseconds"),
        "Duration_sec": round(end_perf - start_perf, 4),
        "Rounds": len(history) - 1,  # history[0] is the pre-negotiation state
        "Final_Total_Revenue_Rate": round(sum(rates[-1]), 4),
        "Worker_Seed": worker_seed,
        "Execution_Mode": "parallel" if progress_dict is not None else "serial",
    }

    return mode_key, strat_name, run_idx, os.path.join(sim_dir, log_filename), trace, temp_path


class SimulationRunner:
    """Orchestrates every SimRun x Mode (NonOverlap/Overlap) x Strategy
    combination and writes each result to Excel as it completes.

    The (mode, strategy) combinations within one SimRun are fully
    independent of each other, so they run in parallel across however many
    CPU cores _detect_resources() finds - falls back to a plain serial loop
    automatically on a single-core machine, with identical output either
    way (same _run_one_combo function runs the computation regardless).

    GPU: detected and reported, but not used for computation here. This
    workload's bottleneck is many small, sequential negotiation rounds
    (each round depends on the last) evaluating modest-sized TSPs
    (~20-40 waypoints per UAV) - GPU parallelism pays off on large, batched,
    independent numerical work, which is a poor match for that shape. Earlier
    testing in this project (vectorizing a genetic algorithm's fitness
    evaluation) found only modest gains from batching even on CPU at this
    problem scale, so a GPU path would very likely lose to CPU once you
    account for per-call transfer/kernel-launch overhead. If your grids grow
    much larger (hundreds of waypoints per UAV) this calculus could flip -
    ask if you want that path built and benchmarked properly at that point."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.randomize_sequence = False
        self.sim_folder = self._find_or_create_sim_folder()
        self._mode_loggers = {}  # one Logger per mode, created lazily, reused across SimRuns
        self.base_seed = cfg.seed  # captured once; _run_one_combo mutates cfg.seed per worker
        self.cpu_count, self.gpu_info = _detect_resources()
        self.workers = max(1, self.cpu_count)

    def _find_or_create_sim_folder(self):
        """Pick the next simulation_N folder name. Both mode trees
        (NonOverlap/Overlap) are always written together each run, so their
        folder counts stay in sync; we number off the NonOverlap tree."""
        for mode in ["NonOverlap", "Overlap"]:
            os.makedirs(os.path.join(self.cfg.results_dir, mode), exist_ok=True)
        base_dir = os.path.join(self.cfg.results_dir, "NonOverlap")
        existing = [d for d in os.listdir(base_dir) if d.startswith("simulation_")]
        return f"simulation_{len(existing) + 1}"

    def _mode_logger(self, mode_key):
        """One negotiation_log.txt per mode, living inside THIS run's own
        {mode}/simulation_N/ folder - not a single file shared across every
        simulation you ever run, so re-running never overwrites a previous
        run's log. Cached so multiple SimRuns for the same mode append to
        the same file instead of re-truncating it each time."""
        if mode_key not in self._mode_loggers:
            sim_dir = os.path.join(self.cfg.results_dir, mode_key, self.sim_folder)
            os.makedirs(sim_dir, exist_ok=True)
            self._mode_loggers[mode_key] = Logger(sim_dir, enabled=self.cfg.enable_logging)
        return self._mode_loggers[mode_key]

    def _summary_dir(self):
        """Where run-level (not mode-specific) artifacts live: the
        execution trace and the framework progress chart. Mirrors the same
        {category}/simulation_N/ layout as NonOverlap/Overlap, just with
        "Summary" as the category, instead of sitting loose at the top of
        results_dir."""
        summary_dir = os.path.join(self.cfg.results_dir, "Summary", self.sim_folder)
        os.makedirs(summary_dir, exist_ok=True)
        return summary_dir

    def run(self):
        gpu_txt = f", GPU detected ({self.gpu_info}, not used - see class docstring)" if self.gpu_info else ", no GPU detected"
        print(f"=== Simulation Started === "
              f"({self.cpu_count} CPU core(s){gpu_txt}, running up to {self.workers} combo(s) in parallel)")
        strategies = self._define_strategies()
        self.total_combos = self.cfg.n_runs * 2 * len(strategies)  # 2 modes: NonOverlap + Overlap
        self.combos_done = 0
        self.combo_durations = []  # individual combo durations, for a wave-aware ETA (see _format_eta)
        self.execution_traces = []  # per-combo technical detail: PID, core, timing, rounds (see _write_execution_trace)
        self.overall_start = time.perf_counter()

        # Each combo writes its own result to a private temp file the
        # instant it finishes (see _write_combo_temp) - the main process
        # just tracks WHERE those files are, and periodically (see
        # _flush_pending_excel) reads whatever's accumulated so far and
        # writes the real, shared xlsx in one batched session. openpyxl-
        # style incremental append (open/modify/save on every combo) was
        # measured at ~54s total for just 100 sheets on one file; batching
        # writes instead measured ~66x faster - so this buffers, rather
        # than writing per-combo, purely to avoid that per-write cost, not
        # because the DataFrames themselves are at any risk: they're
        # already safe on disk in their own temp file well before this
        # ever runs.
        self._pending_manifest = {}    # (mode_key, strat_name) -> {run_idx: temp_path}
        self._flushed_count = {}       # same keys -> how many runs were in the LAST write, to skip redundant rewrites
        self._pending_waypoints = {}   # mode_key -> {"SimRunN": df} (computed upfront, no temp file needed)
        self._last_flush = time.perf_counter()
        self.FLUSH_INTERVAL_SEC = 60

        # Generate every SimRun's base grid up front (cheap - no negotiation
        # involved yet) so the WHOLE simulation's work can be dispatched as
        # one batch and keep every worker continuously busy. Batching only
        # one run at a time (16 combos: 2 modes x 8 strategies) left most
        # cores idle whenever you had more than 16 available, and stalled
        # at the pace of that batch's single slowest combo before starting
        # the next one.
        run_grids = []
        for run_idx in range(1, self.cfg.n_runs + 1):
            if self.cfg.use_pregenerated_grids:
                waypoints, values = _load_pregenerated_grid(self.cfg, run_idx)
            else:
                self.cfg.overlap = False
                silent = Logger(self.cfg.results_dir, enabled=False)
                base_manager = WaypointManager(self.cfg, silent)
                waypoints, values = base_manager.waypoints.copy(), base_manager.values.copy()
            run_grids.append((run_idx, waypoints, values))

        self._run_all(run_grids, strategies)
        self._flush_pending_excel(force=True)  # guarantee anything still buffered gets written
        self._cleanup_combo_temp_files()        # safe now - everything's confirmed in the real xlsx files
        self._write_execution_trace()
        self._write_progress_chart()
        print("=== Simulation Complete ===")

    def _run_all(self, run_grids, strategies):
        """Build and dispatch every (run, mode, strategy) combination in
        the whole simulation as one batch."""
        manager_ctx = multiprocessing.Manager() if self.workers > 1 else contextlib.nullcontext()
        with manager_ctx as sync_manager:
            progress_dict = sync_manager.dict() if sync_manager else None

            all_payloads = []
            self._active = set()  # {(run_idx, mode_key)} pairs actually dispatched
            preflight_ok = None
            for run_idx, base_waypoints, base_values in run_grids:
                for mode_key, mode_flag in [("NonOverlap", False), ("Overlap", True)]:
                    self.cfg.overlap = mode_flag
                    log = self._mode_logger(mode_key)
                    manager = WaypointManager(
                        self.cfg, log, preset_waypoints=list(base_waypoints), preset_values=list(base_values)
                    )

                    if not mode_flag:  # Overlap reuses the same grid, so nothing new to check
                        # Waypoint geometry (and thus preflight feasibility)
                        # is identical for every run - only the revenue draw
                        # differs - so this only needs checking once, ever,
                        # not once per SimRun.
                        if preflight_ok is None:
                            preflight_ok = PreflightChecker(manager, self.cfg, log).run()
                        if not preflight_ok:
                            log.error(f"Preflight failed; skipping NonOverlap for SimRun {run_idx}.")
                            continue

                    if self.cfg.use_pregenerated_grids:
                        pregen_sequences = _pregenerated_initial_sequences(self.cfg, manager, run_idx)
                        cluster_assignment = _cluster_assignment_from_sequences(manager, pregen_sequences)
                    else:
                        assigner = InitialAssigner(self.cfg, manager)
                        cluster_assignment = _cluster_assignment(manager, assigner)
                    df_wp = _make_waypoints_sheet(self.cfg, manager, cluster_assignment)
                    self._pending_waypoints.setdefault(mode_key, {})[f"SimRun{run_idx}"] = df_wp

                    self._active.add((run_idx, mode_key))
                    all_payloads += self._build_payloads(mode_key, mode_flag, run_idx, base_waypoints,
                                                           base_values, strategies, progress_dict)

            if not all_payloads:
                return

            self._flush_waypoints()
            self._init_log_flusher(strategies)
            if self.workers > 1:
                self._run_parallel(all_payloads)
            else:
                self._run_serial(all_payloads)

    def _init_log_flusher(self, strategies):
        """Combos from different SimRuns can now finish in any order (the
        pool just grabs whatever's next), but each mode's log should still
        read top-to-bottom as SimRun 1, then 2, then 3... This tracks, per
        mode, which run_idx is "next in line" to be written, and buffers
        anything that finishes early until its turn comes."""
        self._strategy_names = list(strategies)
        self._pending_logs = {}  # (run_idx, mode_key) -> {strat_name: worker_log_path}
        runs_by_mode = {}
        for (run_idx, mode_key) in self._active:
            runs_by_mode.setdefault(mode_key, set()).add(run_idx)
        self._next_run_to_flush = {mode: min(runs) for mode, runs in runs_by_mode.items()}

    def _on_combo_done(self, mode_key, strat_name, run_idx, worker_log_path):
        key = (run_idx, mode_key)
        self._pending_logs.setdefault(key, {})[strat_name] = worker_log_path
        self._flush_ready_logs(mode_key)

    def _flush_ready_logs(self, mode_key):
        while True:
            run_idx = self._next_run_to_flush.get(mode_key)
            if run_idx is None:
                return
            key = (run_idx, mode_key)
            if key not in self._active:  # e.g. NonOverlap skipped this run - nothing to wait for
                self._next_run_to_flush[mode_key] = self._next_active_run(mode_key, run_idx)
                continue
            collected = self._pending_logs.get(key, {})
            if len(collected) < len(self._strategy_names):
                return  # still waiting on some of this run's strategies
            self.log = self._mode_logger(mode_key)
            for strat_name in self._strategy_names:
                path = collected.get(strat_name)
                if path:
                    self._merge_worker_log(path)
            del self._pending_logs[key]
            self._next_run_to_flush[mode_key] = self._next_active_run(mode_key, run_idx)

    def _next_active_run(self, mode_key, after_run_idx):
        later = [r for (r, m) in self._active if m == mode_key and r > after_run_idx]
        return min(later) if later else None

    def _run_serial(self, payloads):
        for payload in payloads:
            mode_key, strat_name, run_idx = payload[3], payload[5], payload[9]
            print(f"  {mode_key}/{strat_name} (SimRun {run_idx})...")
            result = _run_one_combo(payload)
            self.combo_durations.append(result[4]["Duration_sec"])
            print()  # end the live round-counter line before the next combo starts
            self.combos_done += 1
            self._on_combo_done(*result[:4])
            self.execution_traces.append(result[4])
            self._write_result(result)
            print(f"  \u2705 wrote {result[0]}/{result[1]} (SimRun {result[2]})   "
                  f"[{self.combos_done}/{self.total_combos} overall, ETA {self._format_eta()}]")

    def _write_result(self, result):
        """Record where this combo's result got persisted - a temp file
        the worker already wrote to disk before it even returned (see
        _write_combo_temp). This doesn't touch the DataFrames at all,
        just a path string, so there's nothing here that could lose data
        even if the main process died right after this line."""
        mode_key, strat_name, run_idx = result[0], result[1], result[2]
        temp_path = result[5]
        key = (mode_key, strat_name)
        self._pending_manifest.setdefault(key, {})[run_idx] = temp_path
        self._maybe_flush_pending_excel()

    def _maybe_flush_pending_excel(self):
        if time.perf_counter() - self._last_flush >= self.FLUSH_INTERVAL_SEC:
            self._flush_pending_excel()

    def _flush_pending_excel(self, force=False):
        """For every (mode, strategy) with new results since the last
        flush, read back whatever temp files are on record and write the
        real, shared xlsx in one batched session - much cheaper than
        appending on every single combo (openpyxl-style incremental
        append was measured at ~54s total for just 100 sheets on one
        file; batching instead measured ~66x faster). Skips a (mode,
        strategy) pair entirely if nothing new has arrived for it since
        its last write, unless force=True (used for the guaranteed final
        flush), since re-reading/re-writing unchanged data on every
        periodic cycle over a many-hour run adds up for no benefit.
        Waypoints aren't handled here - they're already fully written
        once by _flush_waypoints() before any negotiation starts."""
        max_ft, speed = self.cfg.max_flight_time, self.cfg.speed
        prefix = f"UAVs{self.cfg.num_uavs}_GRID{self.cfg.grid_width}"
        for key, runs in self._pending_manifest.items():
            if not runs:
                continue
            if not force and len(runs) == self._flushed_count.get(key, -1):
                continue  # nothing new for this strategy since it was last written

            rev_sheets, seq_sheets = {}, {}
            for run_idx, temp_path in runs.items():
                if not os.path.exists(temp_path):
                    continue  # shouldn't happen, but never let one missing file abort the whole flush
                with open(temp_path, "rb") as f:
                    df_rev, df_seq = pickle.load(f)
                sheet = f"SimRun{run_idx}"
                rev_sheets[sheet] = df_rev
                seq_sheets[sheet] = df_seq

            mode_key, strat_name = key
            sim_dir = os.path.join(self.cfg.results_dir, mode_key, self.sim_folder)
            rev_dir = os.path.join(sim_dir, "revenue")
            seq_dir = os.path.join(sim_dir, "sequences")
            os.makedirs(rev_dir, exist_ok=True)
            os.makedirs(seq_dir, exist_ok=True)
            _write_all_sheets(os.path.join(rev_dir, f"{prefix}_{strat_name}.xlsx"), rev_sheets)
            _write_all_sheets(os.path.join(seq_dir, f"{prefix}_{max_ft}_{speed}_{strat_name}_sequences.xlsx"), seq_sheets)
            self._flushed_count[key] = len(runs)

        self._last_flush = time.perf_counter()

    def _cleanup_combo_temp_files(self):
        """Only called after the guaranteed final flush has succeeded, so
        every combo's result is confirmed safely inside the real xlsx
        files before its temp file gets removed."""
        for mode_key in ("NonOverlap", "Overlap"):
            temp_dir = os.path.join(self.cfg.results_dir, mode_key, self.sim_folder, ".combo_temp")
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _flush_waypoints(self):
        """Waypoints only ever get built once, in the pre-dispatch loop
        (before any negotiation work starts) - nothing more will arrive
        for them later, so this writes them immediately rather than
        waiting for the periodic flush."""
        for mode_key, runs in self._pending_waypoints.items():
            sim_dir = os.path.join(self.cfg.results_dir, mode_key, self.sim_folder)
            wp_dir = os.path.join(sim_dir, "waypoints")
            os.makedirs(wp_dir, exist_ok=True)
            prefix = f"UAVs{self.cfg.num_uavs}_GRID{self.cfg.grid_width}"
            _write_all_sheets(os.path.join(wp_dir, f"{prefix}_waypoints.xlsx"), runs)

    def _build_payloads(self, mode_key, mode_flag, run_idx, base_waypoints, base_values, strategies, progress_dict):
        payloads = []
        for i, (strat_name, (drop_sel, pick_sel, is_random)) in enumerate(strategies.items()):
            worker_seed = None
            if self.base_seed is not None:
                mode_offset = 0 if mode_key == "NonOverlap" else 1
                # Deterministic and distinct per (run, mode, strategy), so
                # results are reproducible without every "random" strategy
                # drawing from the identical stream as its siblings.
                worker_seed = (self.base_seed * 10_000) + (run_idx * 100) + (mode_offset * 50) + i
            payloads.append((
                self.cfg, base_waypoints, base_values, mode_key, mode_flag, strat_name,
                drop_sel, pick_sel, is_random, run_idx, self.cfg.results_dir, self.sim_folder,
                worker_seed, progress_dict
            ))
        return payloads

    def _run_parallel(self, payloads):
        """Dispatch every payload across self.workers processes, redrawing
        a single status line (whole-simulation progress/ETA + each
        in-flight strategy's current round, read from the shared progress
        dict) every 0.5s while waiting - can't have each worker print its
        own round updates directly, since several processes writing to the
        same terminal at once would just garble each other's output."""
        progress_dict = payloads[0][-1]
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_run_one_combo, p) for p in payloads}
            pending = futures
            self._print_dashboard(progress_dict)
            while pending:
                done_now, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done_now:
                    result = future.result()
                    self.combos_done += 1
                    # Use the worker's OWN measured Duration_sec (its actual
                    # start-to-finish compute time, timed inside the worker
                    # itself), not wall-clock-since-this-whole-batch-was-
                    # submitted. With all 1600 payloads submitted at once,
                    # that alternative isn't "how long did this combo take" -
                    # it's "how long has it been queued behind ~22 workers
                    # processing 1600 items," which mechanically grows for
                    # every later-completing combo regardless of actual
                    # throughput (confirmed: measured avg nearly tripled
                    # between two points where real throughput barely
                    # moved), making the ETA balloon for no real reason.
                    self.combo_durations.append(result[4]["Duration_sec"])
                    self._on_combo_done(*result[:4])
                    self.execution_traces.append(result[4])
                    self._write_result(result)
                    self._clear_line()
                    print(f"  \u2705 wrote {result[0]}/{result[1]} (SimRun {result[2]})")
                self._maybe_flush_pending_excel()  # checked every ~0.5s regardless of completions,
                                                    # since a production combo can run for a long time
                self._print_dashboard(progress_dict)
        print()  # end the dashboard line

    @staticmethod
    def _clear_line():
        """Overwrite whatever's on the current terminal line, regardless of
        how long it was - using a fixed guess-and-pad width (the previous
        approach) leaves stray characters behind once a dashboard line with
        several active strategies listed grows past that guess."""
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
        print(f"\r{' ' * width}\r", end="")

    def _print_dashboard(self, progress_dict):
        elapsed = time.perf_counter() - self.overall_start
        bar_len = 24
        frac = self.combos_done / self.total_combos if self.total_combos else 1.0
        filled = int(bar_len * frac)
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

        snapshot = dict(progress_dict)
        active = sorted(f"{name}@R{r}" for name, r in snapshot.items() if r != "done")
        line = (f"  [{bar}] {self.combos_done}/{self.total_combos} ({frac * 100:3.0f}%)  "
                f"elapsed {self._format_duration(elapsed)}  ETA {self._format_eta()}")
        if active:
            shown = ", ".join(active[:3])
            extra = f" +{len(active) - 3} more" if len(active) > 3 else ""
            line += f"   now: {shown}{extra}"

        width = shutil.get_terminal_size(fallback=(120, 24)).columns
        print(f"\r{line[:width]:<{width}}", end="", flush=True)

    def _format_eta(self):
        """Wave-based estimate: average how long an individual combo has
        actually taken to compute (measured inside the worker, start to
        finish - NOT wall-clock time since the whole batch was submitted,
        which would mechanically grow for every later-completing combo
        regardless of real throughput, since it bundles in however long
        that combo sat queued behind ~workers others), then project how
        many more such "waves" of up-to-`workers` concurrent combos are
        left. A naive elapsed-time-so-far / combos-done estimate badly
        overstates ETA early on too, for a related reason: with many
        workers running concurrently, few combos have crossed the finish
        line yet even though most of them are close - it mistakes "only 2
        of 22 parallel combos have finished" for "each combo takes half
        the elapsed time," when really most of the other 20 are also
        nearly done."""
        if not self.combo_durations:
            return "…"
        remaining = self.total_combos - self.combos_done
        if remaining <= 0:
            return "0s"
        avg = sum(self.combo_durations) / len(self.combo_durations)
        waves_left = math.ceil(remaining / self.workers) if self.workers > 0 else remaining
        return self._format_duration(avg * waves_left)

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        minutes, sec = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {sec}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m"

    def _merge_worker_log(self, worker_log_path):
        if self.log.enabled and os.path.exists(worker_log_path):
            with open(worker_log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(self.log.log_path, 'a', encoding='utf-8') as f:
                f.write(content)
        if os.path.exists(worker_log_path):
            os.remove(worker_log_path)

    def _write_execution_trace(self):
        """Write out which worker process (and, best-effort, CPU core)
        handled each (SimRun, mode, strategy) combination, how long it
        took, and how many negotiation rounds it ran - three sheets:
        Executions (one row per combo), Worker Summary (aggregated per
        PID, so you can see how evenly load balanced across your cores),
        and Run Info (the overall hardware/config context)."""
        if not self.execution_traces:
            return

        df_exec = pd.DataFrame(self.execution_traces)
        df_exec.sort_values(["SimRun", "Config"], inplace=True)

        summary_rows = []
        for pid, group in df_exec.groupby("Worker_PID"):
            cores_seen = sorted(c for c in group["CPU_Core"].dropna().unique())
            summary_rows.append({
                "Worker_PID": pid,
                "Combos_Handled": len(group),
                "Total_Duration_sec": round(group["Duration_sec"].sum(), 4),
                "Avg_Duration_sec": round(group["Duration_sec"].mean(), 4),
                "Distinct_Cores_Seen": ", ".join(str(c) for c in cores_seen) if cores_seen else "n/a",
                "First_Start": group["Start_Time"].min(),
                "Last_End": group["End_Time"].max(),
            })
        df_summary = pd.DataFrame(summary_rows).sort_values("Worker_PID")

        cores_available = df_exec["CPU_Core"].notna().any()
        df_info = pd.DataFrame([{
            "Simulation_Folder": self.sim_folder,
            "CPU_Cores_Detected": self.cpu_count,
            "GPU_Detected": self.gpu_info or "none",
            "Max_Parallel_Workers": self.workers,
            "Execution_Mode": "parallel" if self.workers > 1 else "serial",
            "Distinct_Worker_PIDs_Used": df_exec["Worker_PID"].nunique(),
            "CPU_Core_Tracking": "available (psutil)" if cores_available else "unavailable (install psutil for this)",
            "Total_Combos": self.total_combos,
            "Total_Wall_Time_sec": round(time.perf_counter() - self.overall_start, 4),
            "N_Runs": self.cfg.n_runs,
            "Num_UAVs": self.cfg.num_uavs,
            "Grid": f"{self.cfg.grid_width}x{self.cfg.grid_height}",
            "Base_Seed": self.base_seed if self.base_seed is not None else "none (unseeded)",
        }]).T.reset_index()
        df_info.columns = ["Field", "Value"]

        summary_dir = self._summary_dir()
        out_path = os.path.join(summary_dir, "execution_trace.xlsx")
        with pd.ExcelWriter(out_path, engine=_excel_engine()) as writer:
            df_exec.to_excel(writer, sheet_name="Executions", index=False)
            df_summary.to_excel(writer, sheet_name="Worker Summary", index=False)
            df_info.to_excel(writer, sheet_name="Run Info", index=False)
        print(f"  Execution trace written to {out_path}")

    def _write_progress_chart(self):
        """One glance at how far along, and how fast, each of the 16
        (mode, strategy) frameworks is relative to the others - bar length
        is SimRuns completed, bar color is average time per combo (green
        fast, red slow), so both "who's ahead" and "why" show up in the
        same picture (e.g. Overlap frameworks running visibly slower than
        NonOverlap ones, since clones inflate their waypoint count).

        Built entirely from execution_traces, which is already being
        collected for the execution trace file regardless - this adds
        nothing to the simulation itself: no extra core, no per-combo
        work, just one plot rendered after every combo is already done.
        matplotlib is optional and not currently a required dependency, so
        this degrades to a skip-with-a-note rather than a crash if it's
        not installed."""
        if not self.execution_traces:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless - no display needed, safe on any machine
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib not installed - skipping framework progress chart; "
                  "pip install matplotlib to enable it)")
            return

        df = pd.DataFrame(self.execution_traces)
        grouped = df.groupby("Config").agg(
            Completed=("SimRun", "nunique"),
            Avg_Duration_sec=("Duration_sec", "mean"),
        ).reset_index()
        grouped["Label"] = grouped["Config"]
        grouped = grouped.sort_values("Completed", ascending=True)  # largest ends up on top when plotted

        fig, ax = plt.subplots(figsize=(11, 6.5))
        cmap = plt.get_cmap("RdYlGn_r")
        vmin, vmax = grouped["Avg_Duration_sec"].min(), grouped["Avg_Duration_sec"].max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)
        colors = [cmap(norm(v)) for v in grouped["Avg_Duration_sec"]]

        bars = ax.barh(grouped["Label"], grouped["Completed"], color=colors, edgecolor="#333", linewidth=0.5)
        max_completed = max(grouped["Completed"])
        for bar, completed, avg_dur in zip(bars, grouped["Completed"], grouped["Avg_Duration_sec"]):
            dur_txt = f"{avg_dur:.0f}s" if avg_dur >= 1 else f"{avg_dur * 1000:.0f}ms"
            ax.text(bar.get_width() + max_completed * 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{completed}/{self.cfg.n_runs}  (~{dur_txt}/combo)", va="center", fontsize=9)

        ax.set_xlabel(f"SimRuns completed (out of {self.cfg.n_runs})")
        ax.set_title("Framework progress & relative speed\n"
                      "bar length = SimRuns done   |   color = avg time per combo (green=fast, red=slow)")
        ax.set_xlim(0, max_completed * 1.42)
        fig.text(0.5, 0.01,
                  "Code: [N/O]on-overlap × [S/R]equential-or-random visit × [G/R]reedy-or-random drop × [G/R]reedy-or-random pick",
                  ha="center", fontsize=8, color="#555")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Avg seconds per combo")
        fig.tight_layout(rect=[0, 0.03, 1, 1])

        out_path = os.path.join(self._summary_dir(), "framework_progress.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Framework progress chart written to {out_path}")

    def _define_strategies(self):
        """(drop_select, pick_select, randomize_sequence) per named
        strategy - plain data, not closures, so it can be sent to worker
        processes (closures and bound methods can't be pickled).

        Filtered by the algorithms.* toggles in settings.yaml, so running
        one specific framework (or several) in isolation is just a matter
        of setting the others to false - not something that requires
        editing this method. A toggle not present in settings.yaml at all
        defaults to enabled, matching "on unless explicitly turned off"."""
        all_strategies = {
            "ModeGG_Sequential": ("greedy", "greedy", False),
            "ModeGR_Sequential": ("greedy", "random", False),
            "ModeRG_Sequential": ("random", "greedy", False),
            "ModeRR_Sequential": ("random", "random", False),
            "ModeGG_Random": ("greedy", "greedy", True),
            "ModeGR_Random": ("greedy", "random", True),
            "ModeRG_Random": ("random", "greedy", True),
            "ModeRR_Random": ("random", "random", True),
        }
        # Maps each strategy key to the settings.yaml algorithms.* toggle
        # name that controls it.
        toggle_for_strategy = {
            "ModeGG_Sequential": "sequential_GG",
            "ModeGR_Sequential": "sequential_GR",
            "ModeRG_Sequential": "sequential_RG",
            "ModeRR_Sequential": "sequential_RR",
            "ModeGG_Random": "random_GG",
            "ModeGR_Random": "random_GR",
            "ModeRG_Random": "random_RG",
            "ModeRR_Random": "random_RR",
        }
        enabled = {
            key: params for key, params in all_strategies.items()
            if getattr(self.cfg, toggle_for_strategy[key], True)
        }
        if not enabled:
            raise ValueError(
                "Every algorithms.* toggle in settings.yaml is false - nothing to run. "
                "Enable at least one strategy (e.g. sequential_GG: true).")
        return enabled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_uavs", type=int)
    parser.add_argument("--grid_width", type=int)
    parser.add_argument("--grid_height", type=int)
    parser.add_argument("--grid_spacing", type=int)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--max_flight_time", type=int)
    parser.add_argument("--n_runs", type=int)
    args = parser.parse_args()

    cfg = Config.from_yaml("settings.yaml")
    cfg.override({k: v for k, v in vars(args).items() if v is not None})
    print("[INFO] Final config:", cfg)

    SimulationRunner(cfg).run()
