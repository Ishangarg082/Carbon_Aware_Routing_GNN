"""
Multi-Seed Experiment Runner
============================
Runs the full 48-hour carbon-routing simulation N times with different random
seeds to produce statistically valid, independent per-run totals.

Each seed controls:
  - traffic-matrix generation (Poisson demands, source/destination pairs)
  - node-to-carbon-profile assignment (which nodes are coal vs solar)
  - any other stochastic elements in the simulation pipeline

All five controllers share the same seed within each run, so the design is
*paired* — ideal for a paired t-test / Wilcoxon signed-rank test.

Controllers
-----------
  1. GNN (ours)                   -- Carbon-Aware GAT routing
  2. OSPF Baseline                -- Shortest-path (hop count)
  3. ThresholdCarbonController    -- Dirty-node avoidance (config-driven)
  4. LinearFlowCarbonController   -- El-Zahr & Zilberman (ACM SIGMETRICS 2025)
                                     Linear power model: e_f = alpha'*bytes + beta'*pkts
  5. NashGameController           -- Hogade et al. (IEEE)
                                     Nash equilibrium Best-Reply game-theoretic routing

Usage
-----
    python run_multi_seed_experiment.py                 # 15 seeds, 48 h, 200 nodes
    python run_multi_seed_experiment.py --seeds 10
    python run_multi_seed_experiment.py --seeds 15 --hours 48 --nodes 200
    python run_multi_seed_experiment.py --seeds 5 --hours 12 --nodes 50  # quick test
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.network_topology import create_network
from simulation.carbon_profiles import create_realistic_profiles
from simulation.energy_model import NetworkEnergyManager
from simulation.gnn_routing_controller import (
    RoutingController,
    BaselineController,
    ThresholdCarbonController,
    LinearFlowCarbonController,
    NashGameController,
)
from enhanced_gnn_model import CarbonAwareGAT
from visualization.multi_seed_analyzer import MultiSeedAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(device="cpu"):
    """Load pre-trained GNN weights (or fall back to random init)."""
    model = CarbonAwareGAT(
        node_features=13,
        edge_features=3,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    )
    weights_path = os.path.join(os.path.dirname(__file__), "best_carbon_gat.pth")
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            return model, True
        except Exception as e:
            print(f"  [WARN] Could not load weights ({e}), using random init.")
    return model, False


def _run_one_seed(
    seed: int,
    num_nodes: int,
    duration_hours: int,
    model,
    verbose: bool = False,
) -> dict:
    """
    Run one complete simulation under a given seed with a **unique topology
    and unique network parameters** so that each run exercises a genuinely
    different network scenario.

    Per-seed variation:
      - Topology type  : hierarchical / scale_free / small_world / random
      - Node count     : Â±20 % of the base value (minimum 10)
      - Base bandwidth : 500 â€“ 2000 Mbps
      - Delay scale    : 50 â€“ 200 ms/unit-distance
      - Carbon profile : clustered / geographic / random

    All three controllers (GNN, OSPF, Threshold) share the same seed &
    topology within a run, keeping the design *paired*.
    """
    duration_s = duration_hours * 3600

    # â”€â”€ Seed-controlled stochastic elements â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Randomise topology type per seed
    topology_choices = ["hierarchical", "scale_free", "small_world", "random"]
    topology_type = topology_choices[seed % len(topology_choices)]

    # 2. Randomise node count Â±20 % of the base (minimum 10)
    node_variation = np.random.randint(-max(0, num_nodes // 5), num_nodes // 5 + 1)
    actual_nodes = max(10, num_nodes + node_variation)

    # 3. Randomise link parameters
    base_bw    = int(np.random.choice([500, 1000, 1500, 2000]))  # Mbps
    delay_scale = int(np.random.uniform(50, 200))               # ms/unit

    # Build topology with seed-specific parameters
    topology = create_network(
        num_nodes=actual_nodes,
        topology=topology_type,
        base_bw=base_bw,
        delay_scale=delay_scale,
    )

    # 4. Carbon-profile assignment (reseed to keep it independent of topology RNG)
    np.random.seed(seed + 10000)
    profile_type = np.random.choice(["clustered", "geographic", "random"])
    carbon_mgr = create_realistic_profiles(actual_nodes, topology_type=profile_type)

    # Energy manager (deterministic given topology)
    energy_mgr = NetworkEnergyManager(actual_nodes)
    energy_mgr.initialize_nodes()

    model.eval()

    if verbose:
        print(f"    seed={seed}, topo={topology_type}, "
              f"nodes={actual_nodes}, bw={base_bw}Mbps, "
              f"delay_scale={delay_scale}, profile={profile_type}")

    # â”€â”€ GNN controller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    gnn_ctrl = RoutingController(
        model, topology, carbon_mgr, energy_mgr,
        control_interval=3600, use_ns3=False, seed=seed,
    )
    gnn_res = gnn_ctrl.run_control_loop(duration_s)

    # â”€â”€ Baseline (OSPF shortest path) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bl_ctrl = BaselineController(
        topology, carbon_mgr, energy_mgr,
        control_interval=3600, seed=seed,
    )
    bl_res = bl_ctrl.run_control_loop(duration_s)

    # â”€â”€ Threshold carbon avoidance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    thr_ctrl = ThresholdCarbonController(
        topology, carbon_mgr, energy_mgr,
        control_interval=3600, seed=seed,
    )
    thr_res = thr_ctrl.run_control_loop(duration_s)

    # â”€â”€ Paper 1 baseline: El-Zahr & Zilberman (ACM SIGMETRICS 2025) â”€â”€â”€
    # "From Measurement to Emissions: Assessing the Carbon Footprint
    #  of Traffic Flows" â€” linear switch power model routing
    p1_ctrl = LinearFlowCarbonController(
        topology, carbon_mgr, energy_mgr,
        control_interval=3600, seed=seed,
    )
    p1_res = p1_ctrl.run_control_loop(duration_s)

    # â”€â”€ Paper 2 baseline: Hogade et al. (IEEE) â€” Nash Equilibrium â”€â”€â”€â”€â”€
    # "Reducing Carbon Footprint of AI Inference Workloads for
    #  Geographically Distributed Data Centers" â€” Best-Reply Nash game
    nash_ctrl = NashGameController(
        topology, carbon_mgr, energy_mgr,
        control_interval=3600, seed=seed,
    )
    nash_res = nash_ctrl.run_control_loop(duration_s)

    return {
        "seed": seed,
        "topology": topology_type,
        "num_nodes": actual_nodes,
        "base_bw_mbps": base_bw,
        "delay_scale": delay_scale,
        "carbon_profile": profile_type,
        "gnn": {
            "total_carbon": float(gnn_res["total_carbon"]),
            "carbon_history": [float(v) for v in gnn_res["carbon_history"]],
            "mlp_predicted_carbon": [float(h.get('mlp_predicted_carbon', 0.0)) for h in gnn_res.get('routing_history', [])],
        },
        "ospf": {
            "total_carbon": float(bl_res["total_carbon"]),
            "carbon_history": [float(v) for v in bl_res["carbon_history"]],
        },
        "threshold": {
            "total_carbon": float(thr_res["total_carbon"]),
            "carbon_history": [float(v) for v in thr_res["carbon_history"]],
        },
        "linear_flow_carbon": {
            "total_carbon": float(p1_res["total_carbon"]),
            "carbon_history": [float(v) for v in p1_res["carbon_history"]],
        },
        "nash_game": {
            "total_carbon": float(nash_res["total_carbon"]),
            "carbon_history": [float(v) for v in nash_res["carbon_history"]],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_multi_seed_experiment(
    n_seeds: int = 15,
    num_nodes: int = 200,
    duration_hours: int = 48,
    results_dir: str = "results",
    seed_start: int = 1,
    verbose: bool = False,
):
    """
    Orchestrate N independent simulation runs and produce statistical outputs.

    Parameters
    ----------
    n_seeds        : number of independent seeds (â‰¥10 recommended)
    num_nodes      : nodes in each run's topology
    duration_hours : simulation duration per run in hours
    results_dir    : where to write output files
    seed_start     : first seed value (seeds = seed_start â€¦ seed_start+n_seeds-1)
    verbose        : print per-seed details
    """
    os.makedirs(results_dir, exist_ok=True)
    seeds = list(range(seed_start, seed_start + n_seeds))

    print("=" * 70)
    print("  MULTI-SEED EXPERIMENT â€” CARBON-AWARE ROUTING")
    print("=" * 70)
    print(f"  Seeds:         {seeds}")
    print(f"  Nodes:         {num_nodes}")
    print(f"  Duration/run:  {duration_hours} h")
    print(f"  Methods:       GNN, OSPF Baseline, Threshold Carbon Avoidance")
    print(f"  Output dir:    {results_dir}/")
    print()

    # Load model once, share across all runs (weights are frozen)
    print("Loading GNN model...")
    model, loaded = _build_model()
    status = "pre-trained weights" if loaded else "random initialization"
    print(f"  Model ready ({status})\n")

    # â”€â”€ Run each seed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    per_seed_results = {}
    wall_times = []

    for i, seed in enumerate(seeds, start=1):
        print(f"[{i:2d}/{n_seeds}] seed={seed} ...", end=" ", flush=True)
        t0 = time.time()
        per_seed_results[seed] = _run_one_seed(
            seed, num_nodes, duration_hours, model, verbose=verbose
        )
        elapsed = time.time() - t0
        wall_times.append(elapsed)

        g = per_seed_results[seed]["gnn"]["total_carbon"]
        o = per_seed_results[seed]["ospf"]["total_carbon"]
        pct = (o - g) / o * 100 if o else 0
        print(f"  GNN={g:>12,.1f}  OSPF={o:>12,.1f}  reduction={pct:+.1f}%  "
              f"({elapsed:.1f}s)")

    total_wall = sum(wall_times)
    print(f"\nAll {n_seeds} runs complete.  Total wall time: {total_wall:.1f}s  "
          f"(avg {total_wall/n_seeds:.1f}s/run)\n")

    # â”€â”€ Statistical analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Computing multi-seed statistics...")
    analyzer = MultiSeedAnalyzer(per_seed_results, duration_hours=duration_hours, num_nodes=num_nodes)

    # Text report
    report_text = analyzer.generate_report()
    print(report_text)

    report_path = os.path.join(results_dir, "multi_seed_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Multi-Seed Statistical Analysis\n\n")
        f.write(f"**N runs:** {n_seeds}  |  "
                f"**Nodes:** {num_nodes}  |  "
                f"**Duration/run:** {duration_hours} h\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n\n")

        # Append per-seed table
        f.write("## Per-Seed Details\n\n")
        f.write("| Seed | Topology | Nodes | BW (Mbps) | Carbon Profile | GNN (gCO2) | OSPF (gCO2) | Threshold (gCO2) | Reduction (%) |\n")
        f.write("|------|----------|-------|-----------|----------------|------------|-------------|-------------------|---------------|\n")
        for s in seeds:
            r = per_seed_results[s]
            g_t = r["gnn"]["total_carbon"]
            o_t = r["ospf"]["total_carbon"]
            h_t = r["threshold"]["total_carbon"]
            red = (o_t - g_t) / o_t * 100 if o_t else 0
            topo   = r.get("topology", "hierarchical")
            nodes  = r.get("num_nodes", num_nodes)
            bw     = r.get("base_bw_mbps", 1000)
            profile = r.get("carbon_profile", "?")
            f.write(f"| {s} | {topo} | {nodes} | {bw} | {profile} | {g_t:,.1f} | {o_t:,.1f} | {h_t:,.1f} | {red:+.2f}% |\n")

    print(f"\nReport saved: {report_path}")

    # JSON raw data
    raw_json_path = os.path.join(results_dir, "multi_seed_raw.json")
    with open(raw_json_path, "w") as f:
        json.dump(per_seed_results, f, indent=2)
    print(f"Raw data saved: {raw_json_path}")

    # Stats JSON
    stats_json_path = os.path.join(results_dir, "multi_seed_stats.json")
    analyzer.save_json(stats_json_path)

    # Plots
    print("\nGenerating plots...")
    saved_plots = analyzer.save_plots(results_dir)
    for p in saved_plots:
        print(f"  Saved: {p}")

    # â”€â”€ Quick sanity-check printout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tt = analyzer.paired_ttest()
    eff = analyzer.cohens_d_between_runs()
    wr = analyzer.win_rate()
    rd = analyzer.reduction_stats()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Mean reduction (GNN vs OSPF): {rd['gnn_vs_ospf']['mean_pct']:.2f}% "
          f"Â± {rd['gnn_vs_ospf']['sd_pct']:.2f}% SD")
    print(f"  95% CI: [{rd['gnn_vs_ospf']['ci_lower']:.2f}%, {rd['gnn_vs_ospf']['ci_upper']:.2f}%]")
    print(f"  Paired t-test: t({tt['gnn_vs_ospf']['df']}) = {tt['gnn_vs_ospf']['t_statistic']:.3f}, "
          f"p = {tt['gnn_vs_ospf']['p_value']:.3e}")
    print(f"  Cohen's d (between-run): {eff['gnn_vs_ospf']['d']:+.4f}  "
          f"[{eff['gnn_vs_ospf']['label']}]")
    print(f"  Win rate: GNN < OSPF in {wr['gnn_vs_ospf']['wins']}/{n_seeds} runs "
          f"({wr['gnn_vs_ospf']['win_rate_pct']:.0f}%)")
    print()
    print(f"  Output files:")
    print(f"    {report_path}")
    print(f"    {raw_json_path}")
    print(f"    {stats_json_path}")
    for p in saved_plots:
        print(f"    {p}")
    print("=" * 70)

    return analyzer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run carbon-routing simulation under N independent seeds for valid statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full paper-quality run (15 seeds, 48 h, 200 nodes) â€” ~5 min on a laptop
  python run_multi_seed_experiment.py

  # Quick smoke-test (5 seeds, 12 h, 50 nodes)
  python run_multi_seed_experiment.py --seeds 5 --hours 12 --nodes 50

  # High-confidence (20 seeds)
  python run_multi_seed_experiment.py --seeds 20
        """,
    )
    p.add_argument("--seeds",  type=int, default=15,
                   help="Number of independent seeds (default: 15)")
    p.add_argument("--hours",  type=int, default=48,
                   help="Simulation duration per run in hours (default: 48)")
    p.add_argument("--nodes",  type=int, default=200,
                   help="Number of network nodes (default: 200)")
    p.add_argument("--seed-start", type=int, default=1,
                   help="First seed value (default: 1)")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Output directory (default: results/)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-seed topology details")
    p.add_argument(
        "--ns3", action="store_true", default=False,
        help=(
            "Route all 5 controllers through real ns-3 (must run in WSL with "
            "ns-3.48 installed). Default: pure-Python simulation."
        ),
    )
    p.add_argument(
        "--netanim", action="store_true", default=False,
        help="Generate NetAnim XML for seed 1 (only used with --ns3).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.seeds < 3:
        print("ERROR: At least 3 seeds required for meaningful statistics.")
        sys.exit(1)
    if args.seeds < 10:
        print(f"WARNING: {args.seeds} seeds is below the recommended minimum (10).")
        print("         Results will have low statistical power.  Use --seeds 15 for paper.")

    if args.ns3:
        # Route through real ns-3 -- delegates to run_ns3_demo.run_ns3_experiment()
        from run_ns3_demo import run_ns3_experiment
        run_ns3_experiment(
            n_seeds=args.seeds,
            num_nodes=args.nodes,
            duration_hours=args.hours,
            enable_netanim=args.netanim,
            seed_start=args.seed_start,
            verbose=args.verbose,
        )
    else:
        # Pure-Python simulation (default -- fast, no ns-3 required)
        run_multi_seed_experiment(
            n_seeds=args.seeds,
            num_nodes=args.nodes,
            duration_hours=args.hours,
            results_dir=args.results_dir,
            seed_start=args.seed_start,
            verbose=args.verbose,
        )
