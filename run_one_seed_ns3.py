#!/usr/bin/env python3
"""
Single-seed NS-3 subprocess runner.
=====================================
Called as a fresh subprocess by run_ns3_demo.py for EACH seed so that the
ns-3 process-wide singletons (NodeList, Simulator, GlobalRouteManager,
Ipv4GlobalRoutingHelper) are fully reset between seeds.

Problem solved
--------------
ns-3's NodeList is a global/static container that is NOT cleared by
Simulator::Destroy().  When run_ns3_demo.py loops over seeds in the SAME
process, each seed's NodeContainer.Create(N) appends N nodes to the existing
list.  By Seed 2, Ipv4GlobalRoutingHelper::PopulateRoutingTables() runs
Dijkstra on Seed1.nodes + Seed2.nodes, making it O(4x) slower; by Seed 10
it runs on ~10x the nodes and hangs indefinitely.

Running each seed as a subprocess gives a clean process -> clean singletons.

Usage (internal -- called by run_ns3_demo.py):
    python3 run_one_seed_ns3.py \\
        --seed 1 --nodes 200 --hours 24 \\
        --results-dir /path/to/results \\
        [--netanim] [--verbose]

Output:
    Writes  {results_dir}/seed_{seed}_result.json
    Prints  RESULT_FILE:{path}   on stdout (parsed by parent)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Ensure project root is on the path regardless of where this file sits
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from simulation.network_topology import create_network
from simulation.carbon_profiles import create_realistic_profiles
from simulation.energy_model import NetworkEnergyManager
from enhanced_gnn_model import CarbonAwareGAT
from integration.ns3_bindings import run_all_controllers_ns3, NS3_AVAILABLE


def _build_model(device="cpu"):
    model = CarbonAwareGAT(
        node_features=13,
        edge_features=3,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
    )
    weights_path = os.path.join(_HERE, "best_carbon_gat.pth")
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"  [seed subprocess] Loaded pre-trained GNN weights")
            return model
        except Exception as e:
            print(f"  [seed subprocess] Could not load weights ({e}), using random init.")
    else:
        print(f"  [seed subprocess] best_carbon_gat.pth not found, using random init.")
    return model


def run_seed(seed, num_nodes, duration_hours, results_dir,
             enable_netanim=False, verbose=False):
    if not NS3_AVAILABLE:
        print("ERROR: ns-3 not available in subprocess.", file=sys.stderr)
        sys.exit(1)

    duration_s = duration_hours * 3600

    # -- Reproducible per-seed variation -------------------------------------
    np.random.seed(seed)
    torch.manual_seed(seed)

    topology_choices = ["hierarchical", "scale_free", "small_world", "random"]
    topology_type    = topology_choices[seed % len(topology_choices)]

    node_variation = np.random.randint(-max(0, num_nodes // 5), num_nodes // 5 + 1)
    actual_nodes   = max(10, num_nodes + node_variation)

    base_bw     = int(np.random.choice([500, 1000, 1500, 2000]))
    delay_scale = int(np.random.uniform(50, 200))

    topology = create_network(
        num_nodes=actual_nodes,
        topology=topology_type,
        base_bw=base_bw,
        delay_scale=delay_scale,
    )

    np.random.seed(seed + 10000)
    profile_type = np.random.choice(["clustered", "geographic", "random"])
    carbon_mgr   = create_realistic_profiles(actual_nodes, topology_type=profile_type)

    energy_mgr = NetworkEnergyManager(actual_nodes)
    energy_mgr.initialize_nodes()

    model = _build_model()
    model.eval()

    if verbose:
        print(f"  seed={seed}, topo={topology_type}, nodes={actual_nodes}, "
              f"bw={base_bw}Mbps, delay_scale={delay_scale}, profile={profile_type}")

    netanim_path = None
    if enable_netanim:
        os.makedirs(results_dir, exist_ok=True)
        netanim_path = os.path.join(results_dir, f"carbon-routing-seed{seed}.xml")

    ns3_results = run_all_controllers_ns3(
        gnn_model=model,
        topology=topology,
        carbon_mgr=carbon_mgr,
        energy_mgr=energy_mgr,
        duration_seconds=duration_s,
        control_interval=3600,
        enable_netanim=enable_netanim,
        netanim_output=netanim_path or os.path.join(results_dir, "carbon-routing-animation.xml"),
        seed=seed,
    )

    result = {
        "seed":           seed,
        "topology":       topology_type,
        "num_nodes":      actual_nodes,
        "base_bw_mbps":   base_bw,
        "delay_scale":    delay_scale,
        "carbon_profile": profile_type,
        "gnn": {
            "total_carbon":         float(ns3_results["gnn"]["total_carbon"]),
            "carbon_history":       [float(v) for v in ns3_results["gnn"]["carbon_history"]],
            "mlp_predicted_carbon": [],
        },
        "ospf": {
            "total_carbon":   float(ns3_results["ospf"]["total_carbon"]),
            "carbon_history": [float(v) for v in ns3_results["ospf"]["carbon_history"]],
        },
        "threshold": {
            "total_carbon":   float(ns3_results["threshold"]["total_carbon"]),
            "carbon_history": [float(v) for v in ns3_results["threshold"]["carbon_history"]],
        },
        "linear_flow_carbon": {
            "total_carbon":   float(ns3_results["linear_flow_carbon"]["total_carbon"]),
            "carbon_history": [float(v) for v in ns3_results["linear_flow_carbon"]["carbon_history"]],
        },
        "nash_game": {
            "total_carbon":   float(ns3_results["nash_game"]["total_carbon"]),
            "carbon_history": [float(v) for v in ns3_results["nash_game"]["carbon_history"]],
        },
    }

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"seed_{seed}_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    # (parent reads from the fixed path: seed_{seed}_result.json)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run one NS-3 seed as a subprocess (called by run_ns3_demo.py)"
    )
    parser.add_argument("--seed",        type=int, required=True)
    parser.add_argument("--nodes",       type=int, required=True)
    parser.add_argument("--hours",       type=int, required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--netanim",     action="store_true", default=False)
    parser.add_argument("--verbose",     action="store_true", default=False)
    args = parser.parse_args()

    t0 = time.time()
    run_seed(
        seed=args.seed,
        num_nodes=args.nodes,
        duration_hours=args.hours,
        results_dir=args.results_dir,
        enable_netanim=args.netanim,
        verbose=args.verbose,
    )
    # Do NOT print 'Seed X done' here -- the parent orchestrator (run_ns3_demo.py)
    # prints that line after reading the JSON result, avoiding duplicate output.


if __name__ == "__main__":
    main()
