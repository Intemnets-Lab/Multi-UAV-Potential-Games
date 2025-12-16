import math
import os
import random
import json
import math
import time
import numpy as np
from datetime import datetime
import pandas as pd
from collections import defaultdict
from datetime import datetime
from typing import List, Type
import random
from typing import List, Optional, Tuple, Dict
import glob
from pathlib import Path
import re
from pathlib import Path
import os, math, json
from pathlib import Path
import os, math, json
import yaml, argparse
from types import SimpleNamespace

def _dict_to_ns(d):
    return SimpleNamespace(**{k: _dict_to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})



def load_settings_yaml(path: Path) -> dict:
    """
    Returns a flat dict of config values from settings.yaml.
    If the file or keys are missing, returns {}.
    """
    if not path or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    # many users nest under 'common' — support both flat or nested
    if isinstance(data, dict) and "common" in data and isinstance(data["common"], dict):
        data = data["common"]
    return data


def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    # Only the overrides you care about:
    p.add_argument("--grid_width", type=int)
    p.add_argument("--grid_height", type=int)
    p.add_argument("--grid_spacing", type=float)
    p.add_argument("--num_uavs", type=int)
    p.add_argument("--speed", type=float)
    p.add_argument("--max_flight_time", type=float)
    p.add_argument("--n_runs", type=int)
    p.add_argument("--enable_logging", type=int)  # 0/1 if you want
    # (Add algorithm toggles if desired, e.g. --irada, --random_RR, etc.)
    p.add_argument("--settings", type=str)

    return p.parse_args()


# === CONFIGURATION ===
class Config:
    def __init__(self, data: dict):
        # ----- project -----
        self.results_dir = data["project"]["results_dir"]
        self.IRADA_benchmarking_dir = data["project"]["IRADA_benchmarking_dir"]
        self.seed = data["simulation"].get("seed", None)
        self.n_runs = data["simulation"]["n_runs"]
        self.enable_logging = data["simulation"]["enable_logging"]
        self.enable_visuals = data["project"].get("enable_visuals", False)

        # reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # ----- grid -----
        self.grid_width = data["grid"]["width"]
        self.grid_height = data["grid"]["height"]
        self.grid_spacing = data["grid"]["spacing"]
        self.zero_prob = data["grid"]["zero_prob"]
        self.lambda_ = data["grid"]["lambda"]

        # ----- uav -----
        self.num_uavs = data["uav"]["num_uavs"]
        self.speed = data["uav"]["speed"]
        self.max_flight_time = data["uav"]["max_flight_time"]

        # ----- revenue -----
        self.random_revenue = data["revenue"]["random"]
        self.fixed_revenue = data["revenue"]["fixed_value"]
        self.revenue_min = data["revenue"]["min"]
        self.revenue_max = data["revenue"]["max"]

        # ----- algorithms -----
        for algo, enabled in data["algorithms"].items():
            setattr(self, algo, enabled)

        # ----- derived -----
        self.C_margin = 0.15 * self.max_flight_time * self.speed
        self.gamma = -math.log(10**-6) / self.C_margin

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

def find_latest_waypoints(results_root: Path, num_uavs: int, grid_w: int) -> Path:
    """
    Return the newest waypoint file under:
       Results/NonOverlap/waypoints/<YYYY-MM-DD>/simulation_<n>/
    matching:
       UAVs{num_uavs}_GRID{grid_w}_waypoints.xlsx
    """

    # Force search inside NON-OVERLAP tree
    root = results_root / "NonOverlap" / "waypoints"
    wanted_name = f"UAVs{num_uavs}_GRID{grid_w}_waypoints.xlsx"

    if not root.exists():
        raise FileNotFoundError(
            f"[IRADA] Expected NonOverlap waypoint folder does NOT exist: {root}"
        )

    # newest <DATE> folders first
    for date_dir in sorted(root.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue

        # newest simulation_* first
        def sim_key(p: Path) -> int:
            m = re.search(r"simulation_(\d+)", p.name)
            return int(m.group(1)) if m else -1

        sim_dirs = sorted(
            (p for p in date_dir.iterdir()
             if p.is_dir() and p.name.startswith("simulation_")),
            key=sim_key,
            reverse=True
        )

        for sim_dir in sim_dirs:
            candidate = sim_dir / wanted_name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"[IRADA] Could not find waypoint file '{wanted_name}' "
        f"under {root}. Check that SimulationRunner wrote outputs correctly."
    )

def get_max_rounds_from_algorithms(output_dir: str, date_str: str, sim_dir: str) -> int:
    """
    Scan Results/NonOverlap/revenue/date_str/sim_dir/*.xlsx and
    Results/NonOverlap/sequences/date_str/sim_dir/*_sequences.xlsx.
    """
    output_dir = os.path.join("Results", "NonOverlap")

    paths = []
    rev_glob = os.path.join(output_dir, "revenue", date_str, sim_dir, "*.xlsx")
    #seq_glob = os.path.join(output_dir, "sequences", date_str, sim_dir, "*_sequences.xlsx")
    paths += glob.glob(rev_glob)
    #paths += glob.glob(seq_glob)

    max_rounds = 0
    for p in paths:
        try:
            wb = pd.read_excel(p, sheet_name=None, index_col=None)
            for df in wb.values():
                max_rounds = max(max_rounds, len(df))
        except Exception:
            continue

    return max_rounds


# 2) utility to read the “master” waypoint revenue file
def load_waypoint_revenues(path_or_outputdir,
                           date_str: str=None,
                           sim_dir: str=None,
                           cfg=None,
                           run_idx: int=1):
    """
    Flexible loader:
      • Case A (new): path_or_outputdir is a Path/str to the exact workbook
                      '.../UAVs{n}_GRID{d}_waypoints.xlsx'
      • Case B (legacy): give (output_dir, date_str, sim_dir, cfg, run_idx)
                         and it will construct the path.

    Returns (revenues: list[int], coords: list[(x,y)]) for sheet SimRun{run_idx}.
    """
    if date_str is None and sim_dir is None and cfg is None:
        # Case A: direct path provided
        wp_path = Path(path_or_outputdir)
        if not wp_path.exists():
            raise FileNotFoundError(f"Waypoint workbook not found: {wp_path}")
    else:
        # Case B: build from pieces
        output_dir = str(path_or_outputdir) if path_or_outputdir else "Results"
        wp_path = Path(output_dir) / "waypoints" / date_str / sim_dir / \
                  f"UAVs{cfg.num_uavs}_GRID{cfg.grid_width}_waypoints.xlsx"
        if not wp_path.exists():
            raise FileNotFoundError(f"Waypoint workbook not found: {wp_path}")

    sheet = f"SimRun{run_idx}"
    df = pd.read_excel(wp_path, sheet_name=sheet)
    df = df.sort_values("Waypoint")
    revenues = df["Revenue"].tolist()
    coords   = list(zip(df["X"], df["Y"]))
    return revenues, coords

# === LOGGER ===
class Logger:
    def __init__(self, output_dir, filename="negotiation_log.txt", enabled: bool = True):
        self.log_path = os.path.join(output_dir, filename)
        self.enabled = enabled
        if not enabled:
            return
        os.makedirs(output_dir, exist_ok=True)
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
    def __init__(self, config):
        self.cfg = config
        d, s = config.grid_width, config.grid_spacing

        # depot fixed at (0,0)
        self.depot = (0.0, 0.0)

        # build grid of REAL POIs only, indices 0…d^2-2
        self.waypoints = [
            (((i+1) % d)*s, ((i+1)//d)*s)
            for i in range(d*d - 1)
        ]

    def shared_pool(self) -> List[int]:
        """Return real‐POI indices whose revenue>0."""
        return [i for i,v in enumerate(self.values) if v>0]

    def estimate_info(self, poi_idx: int, future_t: float) -> float:
        return float(self.values[poi_idx])

    def reset_values(self):
        self.values = self._initial_values.copy()


# --- 2) travel_time which handles depot = None ---
def travel_time(agent, poi_idx: Optional[int], t: float) -> float:
    """If poi_idx is None ⇒ travel to depot; else to waypoint[poi_idx]."""
    x_v, y_v = agent.position(t)
    if poi_idx is None:
        x_i, y_i = agent.manager.depot
    else:
        x_i, y_i = agent.manager.waypoints[poi_idx]
    dist = math.hypot(x_v - x_i, y_v - y_i)
    return dist / agent.cfg.speed if agent.cfg.speed > 0 else float("inf")

DEPOT_IDX= -1
# === PATH OPTIMIZER ===
class PathOptimizer:
    @staticmethod
    def euclidean(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    # --- helper: travel time from agent to poi at time t ---

# === UAV AGENT ===
class UAVAgent:
    def __init__(self, uid, manager: WaypointManager, optimizer: PathOptimizer, config: Config, logger: Logger ):
        self.uid     = uid
        self.manager = manager
        self.opt     = optimizer
        self.cfg     = config
        self.log     = logger
        self.path    = []  # current sequence of waypoint indices
        self.last_comm: Dict[int, float] = {}
        self.clock = 0.0

    def remaining_capacity(self, t: float) -> float:
        """
        How much flight‐time budget remains at absolute time t?
        Since max_flight_time is the total before having to return
        to the depot and recharge, we simply subtract elapsed time.
        """
        return max(self.cfg.max_flight_time - self.clock, 0.0)


    def current_POIs(self) -> List[int]:
        """
        Which POIs does this UAV “own” for the purposes of the
        communication coefficient η?  You said that ownership
        only matters for the *first* pick (initial optimized set),
        after which we pick from the full shared pool.  We’ve
        arranged in ChronoAllocator to pass in a `restrict_pool`
        for that first pick, so here we can just return:
        """
        # if ChronoAllocator has put a _chrono_restrict set on us, use that:
        if hasattr(self, "_chrono_restrict"):
            return list(self._chrono_restrict)
        # otherwise, default to “all waypoints I have visited so far”
        return list(self.path)


    def weighted_center(self, t: float) -> Tuple[float,float]:
        """
        cᵥ(t) = ( ∑_{i in ownership} ŝᵢ(t)·pᵢ ) / ( ∑_{i in ownership} ŝᵢ(t) )
        If no owned POIs, fall back to the depot.
        """
        owned = getattr(self, "ownership", set())
        if not owned:
            return self.manager.depot

        num_x = 0.0
        num_y = 0.0
        denom = 0.0

        for i in owned:
            x_i, y_i = self.manager.waypoints[i]
            s_hat    = self.manager.estimate_info(i, t)
            # skip non‐positive estimates
            if s_hat <= 0:
                continue
            num_x   += s_hat * x_i
            num_y   += s_hat * y_i
            denom   += s_hat

        if denom <= 0.0:
            # no positive weights: just return depot
            return self.manager.depot

        return (num_x/denom, num_y/denom)


    
    def _set_path(self, path: List[int]) -> "UAVAgent":
        """
        Convenience so you can do
           UAVAgent(...)._set_path([0,3,5]).revenue_rate()
        """
        self.path = list(path)
        return self
    
    def revenue_rate(self) -> float:
        """
        Compute the average revenue per unit time for this UAV’s last trip.
        total_rev = sum of the ORIGINAL revenues of all waypoints visited in self.path
        total_time = how long it spent (its personal clock)
        """
        # pull the *original* revenues out of the manager’s snapshot
        # (we assume WaypointManager stashed these in _initial_values)
        total_rev = sum(self.manager._initial_values[i] for i in self.path)
        total_time = getattr(self, "clock", 0.0)

        if total_time <= 0.0:
            return 0.0

        return total_rev / total_time
    
    def position(self, t: float = None) -> tuple[float,float]:
        """
        Return the agent's current x,y.  If it has visited at least one WP,
        we assume it's at the last one in self.path; otherwise it's at the depot.
        """
        if self.path:
            last_wp = self.path[-1]
            return self.manager.waypoints[last_wp]
        else:
            return self.manager.depot

# === INITIAL ASSIGNER ===
class InitialAssigner:
    def __init__(self, config: Config): self.cfg=config
    def uniform(self, W):
        paths = [[] for _ in range(self.cfg.num_uavs)]
        rng   = random.Random()     # seeded from OS by default
        tasks = list(W)
        rng.shuffle(tasks)
        for i, wp in enumerate(tasks):
            paths[i % self.cfg.num_uavs].append(wp)
        return paths

class AngleAssigner:
    def __init__(self, num_uavs: int):
        self.num_uavs = num_uavs

    def assign(self, waypoints: List[Tuple[float, float]]) -> List[List[int]]:
        """
        Partition each waypoint idx into one of num_uavs bins
        by its angle theta = atan2(y,x) in [0,90] degrees.
        """
        # divide the first quadrant into equal slices
        sector = 90.0 / self.num_uavs
        paths = [[] for _ in range(self.num_uavs)]

        for idx, (x, y) in enumerate(waypoints):
            # compute angle in degrees
            theta = math.degrees(math.atan2(y, x))  # yields [0,90] if x,y>=0

            # figure out which sector [0..num_uavs-1] it belongs to
            u = int(theta // sector)
            # if theta == 90 exactly, floor division gives u == num_uavs
            if u >= self.num_uavs:
                u = self.num_uavs - 1
            paths[u].append(idx)

        return paths

# === TASK ALLOCATOR BASE & STRATEGIES ===
class TaskAllocator:
    """
    Base class for task allocation strategies.
    """
    def __init__(self, manager: WaypointManager, config: Config, logger: Logger, optimizer: Type[PathOptimizer] = PathOptimizer):
        self.manager = manager
        self.cfg = config
        self.log = logger
        self.opt = optimizer

    def _setup_agents(self, initial_paths=None):
        """
        Helper to build UAVAgent instances from a list of paths (or
        fall back to a uniform assignment if None).
        """
        if initial_paths is None:
            pool = self.manager.shared_pool()
            initial_paths = InitialAssigner(self.cfg).uniform(pool)
        agents = [
            UAVAgent(u, self.manager, self.opt, self.cfg, self.log)
            for u in range(self.cfg.num_uavs)
        ]
        for u, path in enumerate(initial_paths):
            agents[u].path = list(path)
        return agents
    def allocate(self, task_pool):
        """
        Assign tasks from task_pool.
        Returns (rates, history).
        """
        raise NotImplementedError

import heapq, math
from typing import List, Optional, Tuple

class IRADAAllocator(TaskAllocator):
    """
    Chronological IRADA allocator, per‐UAV rounds:
      - Depot represented by None.
      - Each UAV’s “round” = depot→trip→depot.
      - current_trip cleared on each depot visit.
      - Stops scheduling once each UAV has done max_rounds returns.
      - Logs φ, ε, η, scores, picks, zeroing & resets, and exactly what
        sequence is passed to revenue_rate(), plus ownership & comm times.
    """
    def __init__(self,
                 manager: WaypointManager,
                 config: Config,
                 log: Logger,
                 max_rounds: int = 1000):
        super().__init__(manager, config, log)
        self.max_rounds = max_rounds
        self.kappa: float = 0.0
        self.log.info(f"[IRADA init] max_rounds = {self.max_rounds}")

    def allocate(self,
                 initial_paths: List[List[int]]
                ) -> Tuple[List[List[float]], List[List[List[int]]]]:
        agents = self._setup_agents(initial_paths)
        depot_x, depot_y = self.manager.depot

        # --- 0) init ownership & last_comm on each agent ---
        for v,agent in enumerate(agents):
            agent.ownership = set(initial_paths[v])
            agent.last_comm = {u: 0.0 for u in range(len(agents)) if u != v}
            agent.global_clock = 0.0
            agent.round_start = 0.0
            agent.current_trip = []
            agent.path     = []
            self.log.info(f"UAV{v}: ownership={sorted(agent.ownership)}, clock=0.0, trip=[]")

        # 1) per‐UAV state & first restricted pick φ’s for κ
        first_phis = []
        events: List[Tuple[float,int,Optional[int]]] = []

        for v, agent in enumerate(agents):
            agent._chrono_restrict = set(initial_paths[v])

            # first pick (restricted, no depot)
            first = select_next_target_IRADA(
                agent, 0.0, agents,
                include_depot=False,
                restrict_pool=agent._chrono_restrict
            )
            τ0 = travel_time(agent, first, 0.0)
            Ihat = agent.manager.estimate_info(first, τ0)
            φ1   = (Ihat/τ0) if τ0 > 0 else 0.0
            first_phis.append(φ1)
            self.log.info(f"[first pick] UAV{v}→WP{first} at t=0 (τ={τ0:.2f}s, φ₁={φ1:.4f})")
            heapq.heappush(events, (τ0, v, first))

        # --- Compute κ as the average φ over all waypoints ---
        all_phis = []
        for wp in range(len(self.manager.waypoints)):
            # τ from depot (0.0 start time) to wp
            τ = math.hypot(self.manager.waypoints[wp][0] - depot_x,
                        self.manager.waypoints[wp][1] - depot_y)
            if τ > 0:
                Ĩ = self.manager.estimate_info(wp, τ)  # info at arrival time
                φ = Ĩ / τ
            else:
                φ = 0.0
            all_phis.append(φ)
        self.kappa = (sum(all_phis) / len(all_phis)) if all_phis else 1.0
        self.log.info(f"[IRADA] κ (avg φ over all WPs) = {self.kappa:.4f}")

        # 2) containers for per‐UAV rounds & rates
        trips = [[] for _ in agents]   # List[List[List[int]]]
        rates = [[] for _ in agents]   # List[List[float]]

        # 3) main loop
        while events:
            t, v, poi = heapq.heappop(events)
            agent = agents[v]
            agent.global_clock = t

            # log ownership & comm history
            self.log.info(f"\n[Event] t={t:.7f}s, UAV{v}, poi={'Depot' if poi is None else poi}")
            self.log.info(f"   → UAV{v} ownership = {sorted(agent.ownership)}")
            self.log.info(f"   → UAV{v} last_comm   = {agent.last_comm}")

            # 3a) depot arrival → close a round
            if poi is None:
                round_time = agent.global_clock - agent.round_start
                trip = agent.current_trip.copy()
                trips[v].append(trip)
                batt_left = self.cfg.max_flight_time - round_time
                self.log.info(f"UAV{v}→Depot; Battery left {batt_left:.4f}, recording trip={trip}")
                
                total_rev = sum(self.manager._initial_values[i] for i in trip)
                revenues  = [self.manager._initial_values[i] for i in trip]
                self.log.info(
                    f"[UAV{v} Round{len(trips[v])-1}] "
                    f"Total revenue = {revenues},{total_rev} / Time travelled = {round_time:.4f}"
                )
                rate = (total_rev/round_time) if round_time > 0 else 0.0
                rates[v].append(rate)
                self.log.info(f"[UAV{v} Round{len(trips[v])-1}] revenue_rate({trip}) = {rate:.4f}")

                # reset for next round
                agent.current_trip.clear()
                agent.round_start = agent.global_clock

                if len(trips[v]) >= self.max_rounds:
                    self.log.info(f"UAV{v} reached max_rounds; halting further events")
                    continue

            # 3b) normal POI visit
            if poi is not None:
                # append to this UAV’s current trip
                agent.current_trip.append(poi)
                agent.path.append(poi)
                self.log.info(f"UAV{v}.current_trip append {poi}")
                # we’re going to “talk” to everybody, so no more empty list

                # --- Check and transfer ownership / record comm time ---
                for other in agents:
                    if other.uid != v and poi in other.ownership:
                        # transfer ownership
                        other.ownership.discard(poi)
                        agent.ownership.add(poi)
                        self.log.info(f"→ Ownership will transfer: UAV{other.uid}→UAV{v} for WP{poi} at t={t:.7f}")
                        break

                old = self.manager.values[poi]
                self.manager.values[poi] = 0
                self.log.info(f"Waypoint{poi} revenue {old}→0")
                if not self.manager.shared_pool():
                    self.manager.reset_values()
                    self.log.info("All POIs exhausted; reset_values()")

            # 4) battery check: compute “time since last depot” directly
            current_t = agent.global_clock
            elapsed   = current_t - agent.round_start
            cap       = max(0.0, agent.cfg.max_flight_time - elapsed)
            speed     = getattr(agent.cfg, "speed", 1.0)
            C_remain  = cap * speed  # remaining capacity in metres
            x_v, y_v  = agent.position(current_t)
            dist_home = math.hypot(x_v - depot_x, y_v - depot_y)
            need_home = C_remain < (dist_home + agent.cfg.C_margin)
            self.log.debug(
                f"UAV{v} @ t={current_t:.2f}s → "
                f"C_remain={C_remain:.2f}m (Battery left: {cap:.2f}s), @ speed={speed:.2f}m/s"
            )
            # 5) build candidate pool (all POIs + Depot)
            pool = self.manager.shared_pool() + [None]
            best_i, best_score = None, -math.inf

            for i in pool:
                # φ
                if i is None:
                    R_safe_depot  = C_remain - dist_home - agent.cfg.C_margin
                    φ   = self.kappa * math.exp(-agent.cfg.gamma * R_safe_depot)
                    self.log.debug(
                    f"[DEBUG φ] UAV{v} → Depot: κ={self.kappa:.4f}, R_safe={R_safe_depot:.4f}, γ={agent.cfg.gamma:.4f} (φ = κ * exp(-γ * R_safe)) → φ={φ:.7f}"
                    )
                else:
                    τ   = travel_time(agent, i, current_t)
                    Ĩ   = agent.manager.estimate_info(i, current_t + τ)
                    φ   = (Ĩ/τ) if τ > 0 else 0.0
                    self.log.debug(
                        f"[DEBUG φ] UAV{v} → WP{i}: (φ = Ĩ / τ) τ={τ:.4f}, Ĩ={Ĩ:.4f}"
                    )
                    

                # ε
                if i is None:
                    ε      = 1.0  # depot has no ε
                else:
                    xi, yi = self.manager.waypoints[i]
                    d1 = math.hypot(x_v - xi, y_v - yi)
                    d2 = math.hypot(xi - depot_x, yi - depot_y)
                    R_safe = C_remain - d1 - d2 - agent.cfg.C_margin
                    ε      = math.exp(agent.cfg.gamma * min(0.0, R_safe))
                    self.log.debug(
                        f"[DEBUG ε] UAV{v}→WP{i}: WP(Current)->WP(Scored)={d1:.4f}, WP(Scored)->Depot={d2:.4f}, C_remain={C_remain:.4f}, "
                        f"R_safe (R_safe = C_remain - WP(Current)->WP(Scored) - WP(Scored)->Depot - C_margin)={R_safe:.4f}, γ={agent.cfg.gamma:.4f} (ε = exp(γ * min(0, R_safe)) → ε={ε:.4f}"
                    )

                # η
                if i is None:
                    η = 1.0  # depot has no η
                else:
                    η = compute_eta(agent, i, current_t, agents)

                score = φ * ε * η
                self.log.info(
                    f"UAV{v} eval {'Depot' if i is None else 'WP'+str(i)}: "
                    f"φ={φ:.4f}, ε={ε:.4f}, η={η:.4f} → Score={score:.4f}"
                )
                if score > best_score:
                    best_score, best_i = score, i

            next_poi = best_i
            self.log.info(
                f"UAV{v} next → {'Depot' if next_poi is None else 'WP'+str(next_poi)} "
                f"(score={best_score:.4f})"
            )

            τ_next = travel_time(agent, next_poi, agent.global_clock)
            heapq.heappush(events, (agent.global_clock + τ_next, v, next_poi))
            self.log.info(
                f"Re-queued event at t={agent.global_clock+τ_next:.2f}: "
                f"UAV{v}, poi={'Depot' if next_poi is None else next_poi}"
            )

            # — Now stamp last_comm with *all* the other UAVs —
            for other in agents:
                if other.uid != v:
                    agent.last_comm[other.uid]    = t
                    other.last_comm[v]            = t
                    self.log.info(
                        f"→ Communicated at t={t:.7f} between UAV{v} ↔ UAV{other.uid}")

        self.log.info("=== IRADA complete ===")
        return rates, trips


def travel_time(agent, poi: Optional[int], t: float) -> float:
    """If poi is None, travel back to depot; else to waypoint[poi]."""
    x_v, y_v = agent.position(t)
    if poi is None:
        x_i, y_i = agent.manager.depot
    else:
        x_i, y_i = agent.manager.waypoints[poi]
    d = math.hypot(x_v - x_i, y_v - y_i)
    return d/agent.cfg.speed if agent.cfg.speed>0 else float("inf")

import math
from typing import List

def compute_phi(agent, poi_idx: int, t: float) -> float:
    """
    φᵥⁱ(t) = Ĩ_i(t + τ_{v,i}) / τ_{v,i},
    where τ_{v,i} = dist(q_v(t), p_i) / V.
    Here q_v(t) ≈ last waypoint in agent.path (or the depot if t=0).
    """
    # --- 1) figure out UAV’s current (x_v,y_v) ---
    if hasattr(agent, "path") and len(agent.path) > 0:
        # last visited waypoint
        last_wp = agent.path[-1]
        x_v, y_v = agent.manager.waypoints[last_wp]
    else:
        # at time zero, they’re still at the depot:
        x_v, y_v = agent.manager.depot

    # --- 2) get the POI coordinates ---
    x_i, y_i = agent.manager.waypoints[poi_idx]

    # --- 3) travel time τ_{v,i} ---
    dist = math.hypot(x_v - x_i, y_v - y_i)
    V    = agent.cfg.speed
    τ    = dist / V if V > 0 else float("inf")
    if τ <= 0:
        return 0.0

    # --- 4) your manager must give you Ĩ_i(·) at a future time t+τ ---
    I_hat = agent.manager.estimate_info(poi_idx, t + τ)
    agent.log.debug(f"[DEBUG φ] UAV{agent.uid}→WP{poi_idx}: τ={τ:.4f}, Ĩ={I_hat:.4f}")
    return I_hat / τ



def compute_epsilon(agent, poi_idx: int, t: float) -> float:
    """
    εᵥ(i,t) = exp( γ * min(0, Rᵥᵇᵃᶠᵉ(i,t)) ),
    with Rᵥᵇᵃᶠᵉ(i,t) = Cᵥᵣᵉᵐᵃᶦⁿ(t)
                     - dist(q_v,p_i)
                     - dist(p_i,p_0)
                     - C_margin
    """
    # positions
    x_v, y_v = agent.position(t)
    x_i, y_i = agent.manager.waypoints[poi_idx]
    x_0, y_0 = agent.manager.depot

    # distances
    d1 = math.hypot(x_v - x_i, y_v - y_i)
    dist_home = math.hypot(x_i - x_0, y_i - y_0)

    R_safe = (
        agent.remaining_capacity(t)
        - d1
        - dist_home
        - agent.cfg.C_margin
    )
    agent.log.debug(f"[DEBUG ε] UAV{agent.uid}→WP{poi_idx}: R_safe={R_safe:.4f}, WP_dist={d1:.4f}, Depot_dist={dist_home:.4f}, C_margin={agent.cfg.C_margin:.4f}"),
    return math.exp(agent.cfg.gamma * min(0.0, R_safe))

def compute_eta(agent, poi_idx: int, t: float, all_agents: List) -> float:
    """
    ηᵥ,ᵤ(i,t) =
      [1 - exp(-λ (t - t_comm_{v,u}))] · exp(-||p_i - c_u(t)||² / ||c_u(t)-c_v(t)||²)
      if ∃ u≠v s.t. i ∈ other.ownership
      else 1.
    """
    eps = 1e-6
    for other in all_agents:
        if other.uid == agent.uid:
            continue
        # now check the dynamic ownership set:
        if poi_idx in other.ownership:
            last = agent.last_comm.get(other.uid, -float('inf'))
            raw_dt = t-last
            Δt = raw_dt if raw_dt > eps else eps
            term1 = 1.0 - math.exp(-agent.cfg.lambda_ * Δt)

            cx_u, cy_u = other.weighted_center(t)
            cx_v, cy_v = agent.weighted_center(t)
            x_i, y_i   = agent.manager.waypoints[poi_idx]

            num   = math.hypot(x_i - cx_v, y_i - cy_v)**2
            denom = math.hypot(cx_u - cx_v, cy_u - cy_v)**2
            term2 = math.exp(-num/denom) if denom>0 else 1.0
            η = term1 * term2
            agent.log.debug(
                f"[DEBUG η] UAV{agent.uid}→WP{poi_idx}@ {x_i,y_i} against UAV{other.uid}: "
                f"Δt={Δt:.7f}, λ={agent.cfg.lambda_:.7f} → term1={term1:.7f};  "
                f"centers: c_u=({cx_u:.4f},{cy_u:.4f}), c_v=({cx_v:.4f},{cy_v:.4f});  "
                f"num-term2 (||{(x_i,y_i)}-{cx_v,cy_v}||^2)={num:.7f}, denom-term2 (||{(cx_u,cy_u)}-{cx_v,cy_v}||^2)={denom:.7f} → term2={term2:.7f};  η={η:.4f}"
            )
            return η
        
    agent.log.debug(f"[DEBUG η] UAV{agent.uid} WP{poi_idx} unowned by others → η=1.0000")
    return 1.0

def select_next_target_IRADA(
    agent,
    t: float,
    all_agents: List,
    include_depot: bool = True
) -> int:
    """
    IRADA’s Lines 16–20 (minus Gaussian) in one call.
    Returns the index i* ∈ pool ∪ {0}.
    """
    mgr = agent.manager
    pool = mgr.shared_pool()
    if include_depot:
        pool = [0] + pool

    best_i = 0
    best_score = -math.inf

    for i in pool:
        φ = compute_phi(agent,    i, t)
        ε = compute_epsilon(agent, i, t)
        η = compute_eta(agent,    i, t, all_agents)
        score = φ * ε * η
        if score > best_score:
            best_score, best_i = score, i

    agent.log.info(
        f"[IRADA pick] UAV{agent.uid} → WP{best_i} "
        f"(φ={φ:.3f},ε={ε:.3f},η={η:.3f}) at t={t:.2f}"
    )
    return best_i


# --- patched select_next_target that allows an optional restrict_pool ---
def select_next_target_IRADA(agent,
                             t: float,
                             all_agents: List,
                             include_depot: bool = True,
                             restrict_pool: set[int] = None) -> int:
    """
    Exactly as before, but if `restrict_pool` is non‐None,
    only consider POIs in that set on first pick.
    """
    mgr = agent.manager
    pool = list(restrict_pool) if restrict_pool is not None else mgr.shared_pool()
    if include_depot:
        pool = [0] + pool

    best_i, best_score = 0, -math.inf
    for i in pool:
        φ = compute_phi(agent,    i, t)
        ε = compute_epsilon(agent, i, t)
        η = compute_eta(agent,    i, t, all_agents)
        score = φ*ε*η
        if score > best_score:
            best_score, best_i = score, i

    agent.log.info(
        f"[IRADA chrono] UAV{agent.uid}→WP{best_i}  φ={φ:.3f},ε={ε:.3f},η={η:.3f}"
    )
    return best_i

class ChronoSimulationRunner:
    """
    Runs the IRADAAllocator for n_runs, writing two workbooks:
      • revenue_IRADA.xlsx    (per‑UAV revenue per round)
      • sequences_IRADA.xlsx  (per‑UAV tour per round)
    """
    def __init__(self, cfg: Config, log: Logger):
        self.cfg = cfg
        self.log = log
        self.manager = WaypointManager(cfg)
        self.opt     = PathOptimizer
        self.log.info("[IRADA] Using NonOverlap base grid (no clones)")

    def run(self):
        rev_runs  = defaultdict(list)   # { "IRADA": [df, df, ...] }
        path_runs = defaultdict(list)

        # ---- 0) locate the most recent waypoints workbook that matches this config ----
        RESULTS = Path(__file__).parent / "Results"
        print(f"Results found in: {RESULTS}")
        waypoints_path = find_latest_waypoints(
            results_root=RESULTS,
            num_uavs=self.cfg.num_uavs,
            grid_w=self.cfg.grid_width,
        )
        print(f"[INFO] Using waypoints: {waypoints_path}")
        self.log.info(f"[INFO] Using waypoints: {waypoints_path}")

        # recover date/sim folders for downstream functions/logs
        date_str = waypoints_path.parents[1].name        # e.g., '2025-08-12'
        sim_dir  = waypoints_path.parents[0].name        # e.g., 'simulation_2'

        for run_idx in range(1, self.cfg.n_runs + 1):
            print(f"\n=== IRADARun {run_idx}/{self.cfg.n_runs} ===")
            self.log.info(f"\n=== IRADARun {run_idx}/{self.cfg.n_runs} ===")

            # 1) load this run's waypoints & revenues
            revenues, coords = load_waypoint_revenues(waypoints_path, run_idx=run_idx)
            print("Success! File found at", waypoints_path)
            self.log.info(f"[LOADED] from {waypoints_path}")

            # 2) override the manager with this run’s data
            self.manager.waypoints        = coords
            self.manager.values           = revenues
            self.manager._initial_values  = list(revenues)
            self.log.info(f"[LOADED] manager.values = {revenues}")
            self.log.info(f"[LOADED] manager._initial_values = {self.manager._initial_values}")

            # 3) angle‑based initial assignment
            initial_paths = AngleAssigner(self.cfg.num_uavs).assign(self.manager.waypoints)
            self.log.info(f"Initial paths (angle assign): {initial_paths}")
            self.log.info(f"       coords   = {coords}")
            self.log.info(f"       revenues = {revenues}")

            # 5) figure out how many rounds to run (based on what your analysis uses)
            max_rounds = get_max_rounds_from_algorithms(self.cfg.results_dir, date_str, sim_dir)
            print(f"→ Using max_rounds = {max_rounds}")
            self.log.info(f"→ Using max_rounds = {max_rounds}")

            # 6) run the allocator
            alloc = IRADAAllocator(self.manager, self.cfg, self.log, max_rounds=max_rounds)
            start = time.time()
            per_uav_rates, per_uav_trips = alloc.allocate(initial_paths)
            elapsed = time.time() - start
            print(f"→ IRADA run took {elapsed:.2f}s")
            self.log.info(f"→ IRADA run took {elapsed:.2f}s")

            # 7a) per‑UAV revenue‑rate curve (rows = rounds)
            df_rev = pd.DataFrame(
                {f"UAV{u}": rates for u, rates in enumerate(per_uav_rates)},
                index=[f"{r}" for r in range(max_rounds)]
            )
            rev_runs["IRADA"].append(df_rev)

            # 7b) per‑UAV trip sequences (rows = rounds)
            df_seq = pd.DataFrame(
                {
                    f"UAV{u}": ["-".join(map(str, per_uav_trips[u][r])) for r in range(max_rounds)]
                    for u in range(self.cfg.num_uavs)
                },
                index=[f"{r}" for r in range(max_rounds)]
            )
            path_runs["IRADA"].append(df_seq)

        # 8) write out results (dated / simulation_N folders)
        rev_dir, path_dir = self._prepare_output_dirs()
        self._dump_excel_data(rev_runs, path_runs, rev_dir, path_dir)

        print("\n✅ All IRADA runs done.")
        self.log.info("✅ All IRADA runs done.")

    def _prepare_output_dirs(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        # revenue
        base_rev = os.path.join(self.cfg.IRADA_benchmarking_dir, "revenue", date_str)
        os.makedirs(base_rev, exist_ok=True)
        sims = [
            d for d in os.listdir(base_rev)
            if os.path.isdir(os.path.join(base_rev, d)) and d.startswith("simulation_")
        ]
        idx = len(sims) + 1
        rev_dir = os.path.join(base_rev, f"simulation_{idx}")
        os.makedirs(rev_dir, exist_ok=True)
        # sequences
        base_paths = os.path.join(self.cfg.IRADA_benchmarking_dir, "sequences", date_str)
        os.makedirs(base_paths, exist_ok=True)
        path_dir = os.path.join(base_paths, f"simulation_{idx}")
        os.makedirs(path_dir, exist_ok=True)
        return rev_dir, path_dir

    def _dump_excel_data(self, rev_data, path_data, rev_dir, path_dir):
        # revenue files
        for _, dfs in rev_data.items():
            fname = f"UAVs{self.cfg.num_uavs}_GRID{self.cfg.grid_width}_IRADA.xlsx"
            out = os.path.join(rev_dir, fname)
            with pd.ExcelWriter(out) as writer:
                for i, df in enumerate(dfs, start=1):
                    df.to_excel(writer, sheet_name=f"SimRun{i}", index_label="Round")
        # sequence files
        for _, dfs in path_data.items():
            fname = (
                f"UAVs{self.cfg.num_uavs}_GRID{self.cfg.grid_width}_"
                f"{self.cfg.max_flight_time}_{self.cfg.speed}_IRADA_sequences.xlsx"
            )
            out = os.path.join(path_dir, fname)
            with pd.ExcelWriter(out) as writer:
                for i, df in enumerate(dfs, start=1):
                    df.to_excel(writer, sheet_name=f"SimRun{i}", index_label="Round")

def parse_cli_overrides():
    p = argparse.ArgumentParser(description="Run IRADA")
    # grid & UAV/time overrides
    p.add_argument('--grid_width', type=int, help="Override grid width (columns)")
    p.add_argument('--grid_height', type=int, help="Override grid height (rows)")
    p.add_argument('--grid_spacing', type=float, help="Override grid spacing (metres)")
    p.add_argument('--num_uavs', type=int, help="Override number of UAVs")
    p.add_argument('--speed', type=float, help="Override UAV speed (m/s)")
    p.add_argument('--max_flight_time', type=float, help="Override max flight time (s)")
    p.add_argument('--n_runs', type=int, help="Override number of runs")
    p.add_argument('--seed', type=int, help="Override random seed")
    p.add_argument('--output_dir', '--output-dir',
                   dest='output_dir',
                   type=str,
                   help='Override base results directory')

    g = p.add_mutually_exclusive_group()
    g.add_argument('--enable-logging',  dest='enable_logging', action='store_true',
                   help="Turn logging ON for this run")
    g.add_argument('--no-enable-logging', dest='enable_logging', action='store_false',
                   help="Turn logging OFF for this run")
    p.set_defaults(enable_logging=None)

    return p.parse_args()


if __name__ == "__main__":

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

    log = Logger(cfg.IRADA_benchmarking_dir, enabled=cfg.enable_logging)
    runner = ChronoSimulationRunner(cfg, log)
    runner.run()
