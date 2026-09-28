"""
Full NS-3 Simulation — All 5 Routing Controllers (ns-3.48+)
============================================================

All five routing controllers run through the SAME real ns-3 topology so
results are directly comparable on a physics-accurate simulator:

  1. RoutingController            -- Carbon-Aware GAT (ours)
  2. BaselineController           -- OSPF shortest-path
  3. ThresholdCarbonController    -- Dirty-node avoidance
  4. LinearFlowCarbonController   -- El-Zahr & Zilberman (ACM SIGMETRICS 2025)
  5. NashGameController           -- Hogade et al. (IEEE) Nash equilibrium

Design (Option C -- metric-swap):
  One NS-3 topology per seed. At each hourly interval each controller
  computes its link weights, those weights are applied to NS-3 routing
  tables, NS-3 runs for that interval, and carbon is measured. This gives
  full NS-3 realism for every controller without rebuilding the topology 5x.

Prerequisites:
  - ns-3 installed in WSL with Python bindings (cppyy)
  - Run from WSL with the environment variables set by run_ns3_wsl.sh

Quick start (in WSL):
    bash run_ns3_wsl.sh
    # or manually:
    source ~/ns-allinone-3.48/ns-3.48/ns3-venv/bin/activate
    export PYTHONPATH=~/ns-allinone-3.48/ns-3.48/build/bindings/python:$PYTHONPATH
    export LD_LIBRARY_PATH=~/ns-allinone-3.48/ns-3.48/build/lib:$LD_LIBRARY_PATH
    cd /mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model
    python3 run_ns3_demo.py --hours 24 --nodes 20 --seeds 15
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.network_topology import create_network
from simulation.carbon_profiles import create_realistic_profiles
from simulation.energy_model import NetworkEnergyManager
from enhanced_gnn_model import CarbonAwareGAT
from integration.ns3_bindings import run_all_controllers_ns3, NS3_AVAILABLE
from visualization.multi_seed_analyzer import MultiSeedAnalyzer

# NOTE: NS3_AVAILABLE check is deferred to run_ns3_experiment() at call time,
# so this module can be safely imported without ns-3 being present.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _results_dir():
    """Return the correct results directory (WSL or native)."""
    if os.path.exists("/mnt/e/nnd_implementation"):
        return "/mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model/results"
    return "results"


def _build_model(device="cpu"):
    """Load pre-trained GNN weights, or fall back to random initialisation."""
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
            print("  Loaded pre-trained GNN weights from best_carbon_gat.pth")
            return model, True
        except Exception as e:
            print(f"  Could not load weights ({e}), using random init.")
    else:
        print("  best_carbon_gat.pth not found, using random init.")
    return model, False


def _run_one_seed_subprocess(
    seed: int,
    num_nodes: int,
    duration_hours: int,
    results_dir: str,
    enable_netanim: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run one NS-3 seed in a FRESH SUBPROCESS.

    Why a subprocess?
    -----------------
    ns-3 uses several process-wide C++ singletons that are NOT fully reset by
    Simulator::Destroy():

      * NodeList     -- global container of all ns-3 nodes.
      * RoutingList  -- global routing protocol registry.
      * ChannelList  -- global channel registry.

    When seeds run in the same process, NodeContainer::Create(N) appends N
    nodes to NodeList instead of replacing it.  By Seed 2 the list has
    Seed1.nodes + Seed2.nodes.  Ipv4GlobalRoutingHelper::PopulateRoutingTables()
    runs Dijkstra on the combined list (O(4x) work), and by Seed 10 it hangs
    indefinitely.

    Spawning a subprocess gives each seed a pristine process -- all singletons
    start empty -- and eliminates the issue completely.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_one_seed_ns3.py")

    cmd = [
        sys.executable, script,
        "--seed",        str(seed),
        "--nodes",       str(num_nodes),
        "--hours",       str(duration_hours),
        "--results-dir", results_dir,
    ]
    if enable_netanim:
        cmd.append("--netanim")
    if verbose:
        cmd.append("--verbose")

    # Stream stdout/stderr from the child directly to the terminal so the
    # user sees the time-step progress lines in real time.
    proc = subprocess.run(cmd)  # inherits parent's stdout/stderr

    if proc.returncode != 0:
        raise RuntimeError(
            f"Seed {seed} subprocess exited with code {proc.returncode}")

    result_path = os.path.join(results_dir, f"seed_{seed}_result.json")
    if not os.path.exists(result_path):
        raise RuntimeError(
            f"Seed {seed} subprocess finished but result file not found: {result_path}")

    with open(result_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_ns3_experiment(
    n_seeds: int = 15,
    num_nodes: int = 20,
    duration_hours: int = 24,
    enable_netanim: bool = True,
    seed_start: int = 1,
    verbose: bool = False,
):
    """
    Run N independent NS-3 seeds -- all 5 controllers per seed -- and produce
    statistical outputs via MultiSeedAnalyzer.
    """
    if not NS3_AVAILABLE:
        print("ERROR: ns-3 not available. Run this in WSL with ns-3.48 installed:")
        print("  source ~/ns-allinone-3.48/ns-3.48/ns3-venv/bin/activate")
        print("  export PYTHONPATH=~/ns-allinone-3.48/ns-3.48/build/bindings/python:$PYTHONPATH")
        print("  export LD_LIBRARY_PATH=~/ns-allinone-3.48/ns-3.48/build/lib:$LD_LIBRARY_PATH")
        sys.exit(1)

    rdir = _results_dir()
    os.makedirs(rdir, exist_ok=True)

    print("=" * 70)
    print("  CARBON-AWARE ROUTING -- 5-CONTROLLER NS-3 SIMULATION")
    print("=" * 70)
    print(f"  Seeds    : {n_seeds}  (seeds {seed_start} .. {seed_start + n_seeds - 1})")
    print(f"  Nodes    : ~{num_nodes} per seed (+/-20%)")
    print(f"  Duration : {duration_hours}h per seed")
    print(f"  NetAnim  : {'ENABLED (seed 1 only)' if enable_netanim else 'DISABLED'}")
    print(f"  Results  : {rdir}/")
    print(f"  Runner   : subprocess per seed (clean ns-3 singletons)")
    print("=" * 70)
    print()

    # NOTE: The GNN model is loaded inside each subprocess (run_one_seed_ns3.py)
    # so we don't need to build it here.  We still print a one-line note.
    print("  GNN weights: best_carbon_gat.pth (loaded per subprocess)")
    print()

    per_seed_results = {}
    seeds = list(range(seed_start, seed_start + n_seeds))
    t_total_start = time.time()

    for i, seed in enumerate(seeds):
        print()
        print(f"[Seed {i+1}/{n_seeds}]  seed={seed}")
        print("-" * 50)
        t0 = time.time()
        try:
            result = _run_one_seed_subprocess(
                seed=seed,
                num_nodes=num_nodes,
                duration_hours=duration_hours,
                results_dir=rdir,
                enable_netanim=(enable_netanim and i == 0),
                verbose=verbose,
            )
            per_seed_results[seed] = result
            elapsed = time.time() - t0
            gnn_tot  = result["gnn"]["total_carbon"]
            ospf_tot = result["ospf"]["total_carbon"]
            pct      = (ospf_tot - gnn_tot) / ospf_tot * 100 if ospf_tot else 0
            print(f"\n  Seed {seed} done in {elapsed:.0f}s  |  GNN vs OSPF: {pct:+.1f}%")
        except Exception as e:
            print(f"  Seed {seed} FAILED: {e}")
            import traceback; traceback.print_exc()

    if not per_seed_results:
        print("No successful seeds. Exiting.")
        return

    # ── Save raw JSON ──────────────────────────────────────────────────────
    json_path = os.path.join(rdir, "ns3_multi_seed_results.json")
    with open(json_path, "w") as f:
        json.dump(per_seed_results, f, indent=2)
    print(f"\n  Raw results saved: {json_path}")

    # ── Statistical analysis ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  STATISTICAL ANALYSIS")
    print("=" * 70)
    analyzer = MultiSeedAnalyzer(
        per_seed_results,
        duration_hours=duration_hours,
        num_nodes=num_nodes,
    )
    report = analyzer.generate_report()
    print(report)

    report_path = os.path.join(rdir, "ns3_multi_seed_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Carbon-Aware Routing -- NS-3 5-Controller Multi-Seed Report\n\n")
        f.write(f"**Simulator:** ns-3.48 (real packet-level)  \n")
        f.write(f"**Seeds:** {len(per_seed_results)}  |  ")
        f.write(f"**Nodes:** ~{num_nodes}  |  **Duration:** {duration_hours}h per seed\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")
    print(f"\n  Report saved: {report_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    try:
        plot_paths = analyzer.save_plots(rdir)
        print(f"  Plots saved: {len(plot_paths)} files in {rdir}/")
    except Exception as e:
        print(f"  Plot generation failed: {e}")

    # ── Summary table ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total_start
    print()
    print("=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Total time: {elapsed_total/60:.1f} minutes")
    print(f"  Successful seeds: {len(per_seed_results)}/{n_seeds}")
    stats = analyzer.between_run_stats()
    print(f"\n  {'Controller':<32} {'Mean CO2 (gCO2)':>18}  {'vs OSPF':>10}")
    print("  " + "-" * 62)
    ospf_mean = stats.get("ospf", {}).get("mean", 0) or 1
    ctrl_labels = {
        "gnn":                "GNN (Carbon-Aware GAT) [OURS]",
        "ospf":               "OSPF Baseline",
        "threshold":          "ThresholdCarbonController",
        "linear_flow_carbon": "LinearFlowCarbon (El-Zahr 2025)",
        "nash_game":          "NashGame (Hogade et al.)",
    }
    for key, label in ctrl_labels.items():
        if key in stats:
            mean = stats[key]["mean"]
            pct  = (ospf_mean - mean) / ospf_mean * 100
            ci_lo = stats[key]["ci_lower"]
            ci_hi = stats[key]["ci_upper"]
            print(f"  {label:<32} {mean:>18.2f}  {pct:>+9.1f}%"
                  f"  [95% CI: {ci_lo:.2f}..{ci_hi:.2f}]")
    print()
    print(f"  Results: {rdir}/")
    print(f"    ns3_multi_seed_results.json")
    print(f"    ns3_multi_seed_report.md")
    if enable_netanim:
        print(f"    carbon-routing-seed{seed_start}.xml  (NetAnim animation)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run all 5 routing controllers through real ns-3 simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick 6-hour test, 10 nodes, 3 seeds
  python3 run_ns3_demo.py --hours 6 --nodes 10 --seeds 3

  # Full 24-hour, 20 nodes, 15 seeds + NetAnim for seed 1
  python3 run_ns3_demo.py --hours 24 --nodes 20 --seeds 15 --netanim

  # Extended 48-hour experiment
  python3 run_ns3_demo.py --hours 48 --nodes 30 --seeds 15
        """
    )
    parser.add_argument("--nodes",   type=int, default=20,
                        help="Base number of network nodes (default: 20)")
    parser.add_argument("--hours",   type=int, default=24,
                        help="Simulation duration per seed in hours (default: 24)")
    parser.add_argument("--seeds",   type=int, default=15,
                        help="Number of independent seeds (default: 15)")
    parser.add_argument("--seed-start", type=int, default=1,
                        help="First seed value (default: 1)")
    parser.add_argument("--netanim", action="store_true", default=True,
                        help="Generate NetAnim XML for seed 1 (default: on)")
    parser.add_argument("--no-netanim", dest="netanim", action="store_false",
                        help="Disable NetAnim generation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-seed topology details")

    args = parser.parse_args()

    if args.nodes < 5:
        print("Error: minimum 5 nodes required")
        sys.exit(1)
    if args.hours < 1:
        print("Error: minimum 1 hour required")
        sys.exit(1)
    if args.seeds < 1:
        print("Error: minimum 1 seed required")
        sys.exit(1)

    run_ns3_experiment(
        n_seeds=args.seeds,
        num_nodes=args.nodes,
        duration_hours=args.hours,
        enable_netanim=args.netanim,
        seed_start=args.seed_start,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
