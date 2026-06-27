"""
Full NS-3 Simulation with NetAnim (ns-3.41+)

This script runs the carbon-aware routing simulation using REAL ns-3
and generates NetAnim animation files.

Prerequisites:
- ns-3 installed in WSL with Python bindings (cppyy)
- Run from WSL environment with proper environment setup

Quick start (in WSL):
    source ~/ns3-venv/bin/activate
    export PATH=~/ns3-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export PYTHONPATH=~/ns-allinone-3.41/ns-3.41/build/bindings/python:$PYTHONPATH
    export LD_LIBRARY_PATH=~/ns-allinone-3.41/ns-3.41/build/lib:$LD_LIBRARY_PATH
    cd /mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model
    python3 run_ns3_demo.py

Or use the launch script:
    bash run_ns3_wsl.sh
"""

import os
import sys
import torch

# Import our modules first (ns3_bindings handles ns-3 loading)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation.network_topology import create_network
from simulation.carbon_profiles import create_realistic_profiles
from enhanced_gnn_model import CarbonAwareGAT
from integration.ns3_bindings import run_ns3_simulation, NS3_AVAILABLE
from visualization.metrics_analyzer import MetricsAnalyzer

if not NS3_AVAILABLE:
    print("ns-3 not available!")
    print("\nMake sure you've set up the environment:")
    print("  source ~/ns3-venv/bin/activate")
    print("  export PATH=~/ns3-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    print("  export PYTHONPATH=~/ns-allinone-3.41/ns-3.41/build/bindings/python:$PYTHONPATH")
    print("  export LD_LIBRARY_PATH=~/ns-allinone-3.41/ns-3.41/build/lib:$LD_LIBRARY_PATH")
    sys.exit(1)


def run_full_ns3_simulation(num_nodes=20, duration_hours=24, enable_netanim=True):
    """
    Run complete ns-3 simulation with NetAnim visualization
    
    Args:
        num_nodes: Number of network nodes
        duration_hours: Simulation duration in hours
        enable_netanim: Whether to generate NetAnim trace file
    """
    
    print("=" * 70)
    print(" CARBON-AWARE ROUTING - FULL NS-3 SIMULATION")
    print("=" * 70)
    print("")
    
    # Setup
    print(f"Creating {num_nodes}-node hierarchical topology...")
    topology = create_network(num_nodes=num_nodes, topology='hierarchical')
    print(f"   {topology['graph'].number_of_edges()} links created")
    
    print(f"\nConfiguring carbon intensity profiles...")
    carbon_mgr = create_realistic_profiles(num_nodes, topology_type='clustered')
    
    # Show sample profiles
    sample_time = 43200  # Noon
    print(f"   Sample intensities at noon:")
    for i in range(min(5, num_nodes)):
        intensity = carbon_mgr.get_node_intensity(i, sample_time)
        profile = carbon_mgr.node_assignments[i]
        print(f"     Node {i} ({profile:12s}): {intensity:6.1f} gCO2/kWh")
    
    print(f"\n🧠 Loading enhanced GAT model...")
    model = CarbonAwareGAT(
        node_features=7,
        edge_features=3,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1
    )
    
    # Try to load pre-trained weights
    try:
        model.load_state_dict(torch.load('best_carbon_gat.pth'))
        print("   Loaded pre-trained weights")
    except:
        print("   Using random initialization")
    
    model.eval()
    
    # Run ns-3 simulation
    print(f"\n🚀 Starting ns-3 simulation...")
    print(f"   Duration: {duration_hours} hours")
    print(f"   Control interval: 1 hour")
    if enable_netanim:
        print(f"   NetAnim: ENABLED")
        
        # Smart path selection - save to Windows drive when in WSL
        if os.path.exists('/mnt/e/nnd_implementation'):
            # Running in WSL - save to Windows drive for easy access
            netanim_output = "/mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model/results/carbon-routing-animation.xml"
            print(f"   Output (WSL): {netanim_output}")
            print(f"   Windows path: e:\\nnd_implementation\\Carbon_aware_routing_via_GNN_model\\results\\carbon-routing-animation.xml")
        else:
            # Running natively on Linux or Windows
            netanim_output = "results/carbon-routing-animation.xml"
            print(f"   Output: {netanim_output}")
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(netanim_output), exist_ok=True)
    else:
        print(f"   NetAnim: DISABLED")
        netanim_output = None
    
    print("")
    print("-" * 70)
    
    duration_seconds = duration_hours * 3600
    
    results = run_ns3_simulation(
        gnn_model=model,
        topology=topology,
        carbon_mgr=carbon_mgr,
        duration_seconds=duration_seconds,
        control_interval=3600,
        enable_netanim=enable_netanim,
        netanim_output=netanim_output
    )
    
    print("-" * 70)
    print("")
    
    # --- Run baseline comparison (simulated mode) ---
    print("\nRunning evaluation of 3 routing strategies...\n")
    
    from simulation.gnn_routing_controller import RoutingController, BaselineController, ThresholdCarbonController
    from simulation.energy_model import NetworkEnergyManager
    import numpy as np
    
    energy_mgr = NetworkEnergyManager(num_nodes)
    energy_mgr.initialize_nodes()
    
    # Phase 1: Carbon-aware controller (GNN — includes PyTorch inference)
    print("[1/3] Carbon-Aware GNN Routing...")
    ca_controller = RoutingController(
        model, topology, carbon_mgr, energy_mgr,
        control_interval=3600, use_ns3=False
    )
    ca_results = ca_controller.run_control_loop(duration_seconds)
    
    # Phase 2: Baseline controller (OSPF shortest path — fast)
    print("[2/3] Baseline OSPF Routing...")
    bl_controller = BaselineController(
        topology, carbon_mgr, energy_mgr,
        control_interval=3600
    )
    bl_results = bl_controller.run_control_loop(duration_seconds)
    
    # Phase 3: Threshold-Based Carbon Avoidance (fast)
    print("[3/3] Threshold Carbon Avoidance...")
    hr_controller = ThresholdCarbonController(
        topology, carbon_mgr, energy_mgr,
        control_interval=3600
    )
    hr_results = hr_controller.run_control_loop(duration_seconds)
    
    # --- Generate metrics report ---
    print("\n📈 Generating metrics report...")
    
    # Determine results directory
    if os.path.exists('/mnt/e/nnd_implementation'):
        results_dir = "/mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model/results"
    else:
        results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    analyzer = MetricsAnalyzer(ca_results, bl_results, hr_results)
    
    # Print report
    report_text = analyzer.generate_report()
    print(report_text)
    
    # Export CSV
    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    analyzer.export_csv(csv_path)
    
    # Save final report
    report_path = os.path.join(results_dir, "final_report.md")
    with open(report_path, 'w') as f:
        f.write("# Carbon-Aware Routing - NS-3 Simulation Results\n\n")
        f.write(f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Nodes:** {num_nodes} | **Duration:** {duration_hours}h\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n\n")
        
        # Add ns-3 simulation log
        if results.get('log'):
            f.write("## NS-3 Simulation Log\n\n")
            f.write("| Time (s) | Hours | Carbon (gCO2) |\n")
            f.write("|----------|-------|---------------|\n")
            for entry in results['log']:
                f.write(f"| {entry['time']:>8d} | {entry['time']/3600:>5.1f} | {entry['carbon_pred']:>13.4f} |\n")
            f.write(f"\n**Average:** {np.mean([e['carbon_pred'] for e in results['log']]):.4f} gCO2\n")
    
    print(f"Report saved: {report_path}")
    
    # Generate comprehensive results plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        reduction = analyzer.calculate_carbon_reduction()
        energy = analyzer.calculate_energy_metrics()
        peak_offpeak = analyzer.calculate_peak_offpeak()
        
        ca_carbon = np.array(ca_results['carbon_history'])
        bl_carbon = np.array(bl_results['carbon_history'])
        hr_carbon = np.array(hr_results['carbon_history'])
        timestamps_hr = np.array(ca_results['timestamps']) / 3600
        min_len = min(len(ca_carbon), len(bl_carbon), len(hr_carbon))
        
        bl_total = bl_results['total_carbon']
        hr_total = hr_results['total_carbon']
        hr_pct = ((bl_total - hr_total) / bl_total * 100) if bl_total else 0
        
        # --- Consistent plot styling ---
        BG = '#ffffff'
        CARD = '#ffffff'
        BORDER = '#d4d4d4'
        TXT = '#1a1a1a'
        TXT2 = '#525252'
        TXT_M = '#a3a3a3'
        GRN = '#16a34a'
        RED = '#dc2626'
        BLU = '#2563eb'
        AMB = '#d97706'

        def _style(ax, title):
            ax.set_facecolor(CARD)
            ax.tick_params(colors=TXT_M, labelsize=8)
            ax.spines['bottom'].set_color(BORDER)
            ax.spines['left'].set_color(BORDER)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(title, fontsize=11, fontweight='600', color=TXT, pad=10, loc='left')
        
        # ── Plot 1: Hourly Emissions ──
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
        _style(ax, 'Hourly Carbon Emissions')
        ax.plot(timestamps_hr[:min_len], bl_carbon[:min_len], color=RED, linewidth=1.8, label='Baseline (OSPF)', linestyle='--', alpha=0.8)
        ax.plot(timestamps_hr[:min_len], hr_carbon[:min_len], color=AMB, linewidth=1.8, label='Threshold Carbon', linestyle='-.', alpha=0.8)
        ax.plot(timestamps_hr[:min_len], ca_carbon[:min_len], color=GRN, linewidth=2, label='Carbon-Aware (GNN)')
        ax.fill_between(timestamps_hr[:min_len], ca_carbon[:min_len], bl_carbon[:min_len],
                        alpha=0.10, color=GRN, where=bl_carbon[:min_len] >= ca_carbon[:min_len])
        ax.set_xlabel('Time (hours)', fontsize=9, color=TXT2)
        ax.set_ylabel('Carbon (gCO2)', fontsize=9, color=TXT2)
        ax.legend(fontsize=8, framealpha=0.9, edgecolor=BORDER, facecolor=CARD, labelcolor=TXT2)
        ax.grid(True, alpha=0.15, color=BORDER)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '01_hourly_emissions.png'), dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        
        # ── Plot 2: Total Comparison Bars ──
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
        _style(ax, f'Total Carbon Comparison ({reduction["percentage"]:.1f}% GNN Reduction)')
        labels = ['Baseline\n(OSPF)', 'Threshold\n(Top-25%)', 'GNN\n(Ours)']
        totals = [bl_total, hr_total, reduction['carbon_aware_total']]
        bar_colors = [RED, AMB, GRN]
        bars = ax.bar(labels, totals, color=bar_colors, alpha=0.75, edgecolor='none', width=0.5)
        for bar, val in zip(bars, totals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='600', color=TXT)
        ax.set_ylabel('Total Carbon (gCO2)', fontsize=9, color=TXT2)
        ax.grid(axis='y', alpha=0.15, color=BORDER)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '02_total_comparison.png'), dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        
        # ── Plot 3: Cumulative Savings ──
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
        _style(ax, 'Cumulative Carbon Savings vs Baseline')
        cum_gnn = np.cumsum(bl_carbon[:min_len] - ca_carbon[:min_len])
        cum_hr  = np.cumsum(bl_carbon[:min_len] - hr_carbon[:min_len])
        ax.plot(timestamps_hr[:min_len], cum_gnn, color=GRN, linewidth=2, label='GNN vs Baseline')
        ax.plot(timestamps_hr[:min_len], cum_hr,  color=AMB, linewidth=1.8, linestyle='--', label='Threshold vs Baseline')
        ax.fill_between(timestamps_hr[:min_len], 0, cum_gnn, alpha=0.10, color=GRN, where=np.array(cum_gnn) >= 0)
        ax.axhline(y=0, color=TXT_M, linestyle='-', linewidth=0.5)
        ax.set_xlabel('Time (hours)', fontsize=9, color=TXT2)
        ax.set_ylabel('Cumulative Savings (gCO2)', fontsize=9, color=TXT2)
        ax.legend(fontsize=8, framealpha=0.9, edgecolor=BORDER, facecolor=CARD, labelcolor=TXT2)
        ax.grid(True, alpha=0.15, color=BORDER)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '03_cumulative_savings.png'), dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        
        # ── Plot 4: Hourly Reduction % ──
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
        _style(ax, 'Hourly Carbon Reduction (%)')
        gnn_pct = [(bl_carbon[i] - ca_carbon[i]) / bl_carbon[i] * 100 if bl_carbon[i] != 0 else 0 for i in range(min_len)]
        hr_pct_arr = [(bl_carbon[i] - hr_carbon[i]) / bl_carbon[i] * 100 if bl_carbon[i] != 0 else 0 for i in range(min_len)]
        w = 0.35
        x = timestamps_hr[:min_len]
        ax.bar(x - w/2, gnn_pct, width=w, color=GRN, alpha=0.7, label=f'GNN (avg {np.mean(gnn_pct):.1f}%)')
        ax.bar(x + w/2, hr_pct_arr, width=w, color=AMB, alpha=0.7, label=f'Threshold (avg {np.mean(hr_pct_arr):.1f}%)')
        ax.axhline(y=0, color=TXT_M, linewidth=0.5)
        ax.set_xlabel('Time (hours)', fontsize=9, color=TXT2)
        ax.set_ylabel('Reduction (%)', fontsize=9, color=TXT2)
        ax.legend(fontsize=8, framealpha=0.9, edgecolor=BORDER, facecolor=CARD, labelcolor=TXT2)
        ax.grid(True, alpha=0.15, color=BORDER)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '04_hourly_reduction_pct.png'), dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        
        # ── Plot 5: Peak vs Off-Peak ──
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
        _style(ax, 'Peak vs Off-Peak Performance')
        p_labels, p_bl, p_hr, p_ca = [], [], [], []
        # Compute peak/off-peak for heuristic
        hr_carbon_arr = hr_carbon[:min_len]
        ca_carbon_arr = ca_carbon[:min_len]
        bl_carbon_arr = bl_carbon[:min_len]
        ts_arr = np.array(ca_results['timestamps'][:min_len])
        for period, lo, hi in [('Peak (9AM-5PM)', 9, 17), ('Off-Peak', 0, 9)]:
            mask = np.zeros(min_len, dtype=bool)
            for idx in range(min_len):
                hour = (ts_arr[idx] / 3600) % 24
                if period.startswith('Peak'):
                    mask[idx] = (9 <= hour <= 17)
                else:
                    mask[idx] = (hour < 9 or hour > 17)
            if mask.any():
                p_labels.append(period)
                p_bl.append(float(bl_carbon_arr[mask].sum()))
                p_hr.append(float(hr_carbon_arr[mask].sum()))
                p_ca.append(float(ca_carbon_arr[mask].sum()))
        if p_labels:
            x = np.arange(len(p_labels))
            bw = 0.22
            ax.bar(x - bw, p_bl, bw, label='Baseline', color=RED, alpha=0.8)
            ax.bar(x,      p_hr, bw, label='Threshold', color=AMB, alpha=0.8)
            ax.bar(x + bw, p_ca, bw, label='GNN', color=GRN, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(p_labels, fontsize=9)
            for i in range(len(p_labels)):
                gnn_red = (p_bl[i] - p_ca[i]) / p_bl[i] * 100 if p_bl[i] else 0
                ax.text(x[i] + bw, p_ca[i], f'{gnn_red:.1f}%', ha='center', va='bottom',
                        fontsize=8, fontweight='600', color=GRN)
        ax.set_ylabel('Carbon (gCO2)', fontsize=9, color=TXT2)
        ax.legend(fontsize=8, framealpha=0.9, edgecolor=BORDER, facecolor=CARD, labelcolor=TXT2)
        ax.grid(axis='y', alpha=0.15, color=BORDER)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '05_peak_offpeak.png'), dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        
        # ── Plot 6: Consistency Pie ──
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax.set_facecolor(CARD)
        gnn_better = int(np.sum(ca_carbon[:min_len] < bl_carbon[:min_len]))
        hr_better  = int(np.sum((hr_carbon[:min_len] < bl_carbon[:min_len]) & (ca_carbon[:min_len] >= bl_carbon[:min_len])))
        bl_better  = min_len - gnn_better - hr_better
        sizes = [gnn_better, hr_better, bl_better]
        labels_pie = [f'GNN Best\n{gnn_better}/{min_len}', f'Threshold Best\n{hr_better}/{min_len}', f'Baseline Best\n{bl_better}/{min_len}']
        colors_pie = [GRN, AMB, RED]
        # Remove zero-size slices
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors_pie) if s > 0]
        if non_zero:
            sizes_nz, labels_nz, colors_nz = zip(*non_zero)
            ax.pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.0f%%',
                   startangle=90, textprops={'fontsize': 10, 'color': TXT})
        ax.set_title(f'Routing Consistency ({energy["consistency"]:.0f}% GNN effective)',
                     fontsize=11, fontweight='600', color=TXT, pad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, '06_consistency_pie.png'), dpi=150, bbox_inches='tight', facecolor=BG)
        plt.close()
        
        print(f"Saved 6 individual plots to {results_dir}/")
    except Exception as e:
        print(f"Plot generation failed: {e}")
        import traceback; traceback.print_exc()
    
    # Generate interactive HTML animation
    try:
        from visualization.generate_animation import generate_animation
        anim_path = os.path.join(results_dir, "simulation_animation.html")
        generate_animation(topology, carbon_mgr, ca_results, bl_results, anim_path)
    except Exception as e:
        print(f"Animation generation failed: {e}")
    
    # Final summary
    print("")
    print("=" * 70)
    print(" SIMULATION COMPLETE")
    print("=" * 70)
    print("")
    print(f"Carbon Reduction: {reduction['percentage']:.1f}%")
    print(f"   Baseline:      {reduction['baseline_total']:.2f} gCO2")
    hr_total_final = hr_results['total_carbon']
    hr_pct_final = ((reduction['baseline_total'] - hr_total_final) / reduction['baseline_total'] * 100) if reduction['baseline_total'] else 0
    print(f"   Threshold:     {hr_total_final:.2f} gCO2  ({hr_pct_final:+.1f}%)")
    print(f"   Carbon-Aware:  {reduction['carbon_aware_total']:.2f} gCO2  ({reduction['percentage']:+.1f}%)")
    print(f"   GNN Saved:     {reduction['absolute']:.2f} gCO2")
    print("")
    print(f"Results saved to: {results_dir}/")
    print(f"   - metrics_summary.csv")
    print(f"   - final_report.md")
    print(f"   - 01_hourly_emissions.png  through  06_consistency_pie.png")
    print(f"   - simulation_animation.html")
    if enable_netanim:
        print(f"   - carbon-routing-animation.xml")
    print("")
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run full ns-3 simulation with carbon-aware routing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick 6-hour simulation
  python3 run_ns3_demo.py --hours 6 --nodes 10
  
  # Full 24-hour simulation with NetAnim
  python3 run_ns3_demo.py --hours 24 --nodes 20 --netanim
  
  # Extended 48-hour simulation
  python3 run_ns3_demo.py --hours 48 --nodes 30
        """
    )
    
    parser.add_argument('--nodes', type=int, default=20,
                       help='Number of network nodes (default: 20)')
    parser.add_argument('--hours', type=int, default=24,
                       help='Simulation duration in hours (default: 24)')
    parser.add_argument('--netanim', action='store_true',
                       help='Enable NetAnim trace generation')
    parser.add_argument('--no-netanim', dest='netanim', action='store_false',
                       help='Disable NetAnim (faster)')
    parser.set_defaults(netanim=True)
    
    args = parser.parse_args()
    
    # Validate
    if args.nodes < 5:
        print("Error: Minimum 5 nodes required")
        sys.exit(1)
    
    if args.hours < 1:
        print("Error: Minimum 1 hour required")
        sys.exit(1)
    
    # Run
    results = run_full_ns3_simulation(
        num_nodes=args.nodes,
        duration_hours=args.hours,
        enable_netanim=args.netanim
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
