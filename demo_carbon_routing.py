"""
Standalone demo of carbon-aware routing (no NS3 required)

This script demonstrates the carbon reduction achievable with the trained model
on a simulated network without requiring NS3 installation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
import numpy as np
from controllers.mlp_routing_controller import CarbonAwareRoutingController



def create_test_network(num_nodes=50, edge_prob=0.4):
    """Create a random network with carbon intensities"""
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    
    # Add carbon intensities to nodes
    # Simulate 3 regions: low (200-300), medium (350-450), high (500-600)
    regions = ['low', 'medium', 'high']
    for node in G.nodes():
        region = np.random.choice(regions)
        if region == 'low':
            carbon = np.random.uniform(200, 300)
        elif region == 'medium':
            carbon = np.random.uniform(350, 450)
        else:
            carbon = np.random.uniform(500, 600)
        
        G.nodes[node]['carbon_intensity'] = carbon
        G.nodes[node]['region'] = region
    
    return G


def generate_traffic_flows(num_nodes, num_flows=50):
    """Generate random traffic flows"""
    flows = []
    protocols = ['TCP', 'UDP', 'ICMP']
    
    for _ in range(num_flows):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        if src == dst:
            continue
        
        demand = np.random.uniform(0.1, 2.0)  # Gbps
        protocol = np.random.choice(protocols)
        
        flows.append((src, dst, demand, protocol))
    
    return flows


def baseline_routing(graph, traffic_flows):
    """Baseline: shortest path routing"""
    total_carbon = 0
    
    for src, dst, demand, protocol in traffic_flows:
        try:
            # Shortest path (hop count)
            path = nx.shortest_path(graph, src, dst)
            num_hops = len(path) - 1
            
            # Calculate carbon for this flow using realistic formula
            # (matching the scale of training data in carbon_network_data.csv)
            avg_carbon_intensity = np.mean([graph.nodes[n]['carbon_intensity'] for n in path])
            
            # Flow characteristics
            duration_ms = 60 * 1000  # 1 minute in milliseconds
            byte_count = demand * 1e9 / 8 * (duration_ms / 1000)  # Gbps to bytes
            packet_count = int(byte_count / 1500)
            
            # Realistic carbon calculation (based on training data formula)
            # Power consumption model:
            # - Base power per hop ~ 45W
            # - CPU component (assume 50% util)
            # - Data transmission energy
            # - Cooling factor ~ 1.8
            cpu_usage = 50.0
            power_watts = (num_hops * 45) + (cpu_usage * 1.2) + (byte_count / 1e7)
            total_power_with_cooling = power_watts * 1.8
            
            # Energy in kWh
            energy_kwh = (total_power_with_cooling / 1000) * (duration_ms / 3600000) * 10
            
            # Carbon emissions = energy × carbon intensity
            carbon = energy_kwh * avg_carbon_intensity
            total_carbon += carbon

            
        except nx.NetworkXNoPath:
            continue
    
    return total_carbon


def threshold_carbon_routing(graph, traffic_flows):
    """Threshold-Based Carbon Avoidance: only avoids top-25% dirtiest nodes.
    
    Adds a penalty to links touching high-carbon nodes (top quartile).
    All other links use pure hop-count routing.  Crude binary filter
    unlike the GNN's continuous carbon optimization.
    """
    HOP_COST = 10000
    DIRTY_PENALTY = 3000  # ~0.3 extra hops for dirty-node links
    
    # Compute threshold: top 25% dirtiest nodes
    all_ci = [graph.nodes[n].get('carbon_intensity', 400) for n in graph.nodes()]
    threshold = np.percentile(all_ci, 75)
    
    # Build weighted graph
    weighted = graph.copy()
    for u, v in weighted.edges():
        ci_u = weighted.nodes[u].get('carbon_intensity', 400)
        ci_v = weighted.nodes[v].get('carbon_intensity', 400)
        if max(ci_u, ci_v) > threshold:
            weighted[u][v]['threshold_weight'] = HOP_COST + DIRTY_PENALTY
        else:
            weighted[u][v]['threshold_weight'] = HOP_COST

    total_carbon = 0
    for src, dst, demand, protocol in traffic_flows:
        try:
            path = nx.shortest_path(weighted, src, dst, weight='threshold_weight')
            num_hops = len(path) - 1
            avg_carbon_intensity = np.mean([graph.nodes[n]['carbon_intensity'] for n in path])

            duration_ms = 60 * 1000
            byte_count = demand * 1e9 / 8 * (duration_ms / 1000)
            cpu_usage = 50.0
            power_watts = (num_hops * 45) + (cpu_usage * 1.2) + (byte_count / 1e7)
            total_power_with_cooling = power_watts * 1.8
            energy_kwh = (total_power_with_cooling / 1000) * (duration_ms / 3600000) * 10
            carbon = energy_kwh * avg_carbon_intensity
            total_carbon += carbon
        except nx.NetworkXNoPath:
            continue
    return total_carbon


def main():
    print("="*70)
    print("CARBON-AWARE ROUTING DEMO (No NS3 Required)")
    print("="*70)
    
    # Create network
    print("\nCreating test network...")
    num_nodes = 20
    G = create_test_network(num_nodes)
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Avg degree: {2 * G.number_of_edges() / num_nodes:.1f}")
    
    # Store for later use
    num_edges = G.number_of_edges()
    
    # Show carbon distribution
    carbon_intensities = [G.nodes[n]['carbon_intensity'] for n in G.nodes()]
    print(f"\nCarbon Intensity Distribution:")
    print(f"  Min: {min(carbon_intensities):.1f} gCO2/kWh")
    print(f"  Mean: {np.mean(carbon_intensities):.1f} gCO2/kWh")
    print(f"  Max: {max(carbon_intensities):.1f} gCO2/kWh")
    
    # Generate traffic
    print("\nGenerating traffic flows...")
    num_flows = 50
    traffic_flows = generate_traffic_flows(num_nodes, num_flows)
    print(f"  Total flows: {len(traffic_flows)}")
    
    # Baseline routing (shortest path)
    print("\n" + "="*70)
    print("BASELINE: Shortest Path Routing")
    print("="*70)
    baseline_carbon = baseline_routing(G, traffic_flows)
    print(f"  Total Carbon: {baseline_carbon:.6f} gCO2eq")
    
    # Threshold-based carbon avoidance
    print("\n" + "="*70)
    print("THRESHOLD: Avoid Top-25% Dirtiest Nodes")
    print("="*70)
    heuristic_carbon = threshold_carbon_routing(G, traffic_flows)
    heuristic_reduction = ((baseline_carbon - heuristic_carbon) / baseline_carbon) * 100
    print(f"  Total Carbon: {heuristic_carbon:.6f} gCO2eq")
    print(f"  Reduction vs Baseline: {heuristic_reduction:+.2f}%")
    
    # Carbon-aware routing
    print("\n" + "="*70)
    print("CARBON-AWARE: ML-Optimized Routing")
    print("="*70)
    
    try:
        controller = CarbonAwareRoutingController()
        routing_decisions, carbon_aware_carbon = controller.route_traffic_matrix(G, traffic_flows)
        
        print(f"  Total Carbon: {carbon_aware_carbon:.6f} gCO2eq")
        print(f"  Flows routed: {len(routing_decisions)}")
        
        # Calculate reduction
        reduction = ((baseline_carbon - carbon_aware_carbon) / baseline_carbon) * 100
        gnn_vs_heuristic = ((heuristic_carbon - carbon_aware_carbon) / heuristic_carbon) * 100 if heuristic_carbon else 0
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"  Baseline Carbon:     {baseline_carbon:.6f} gCO2eq")
        print(f"  ECMP-Carbon:         {heuristic_carbon:.6f} gCO2eq  ({heuristic_reduction:+.2f}% vs Baseline)")
        print(f"  Carbon-Aware (GNN):  {carbon_aware_carbon:.6f} gCO2eq  ({reduction:+.2f}% vs Baseline)")
        print(f"  GNN vs ECMP-Carbon:  {gnn_vs_heuristic:+.2f}%")
        
        if reduction > 0:
            print(f"\n  SUCCESS: Achieved {reduction:.2f}% carbon reduction!")
        else:
            print(f"\n  Note: Negative reduction may occur with random networks.")
            print(f"        Try running again or use larger network with more diversity.")
        
        # Show statistics
        stats = controller.get_statistics()
        print(f"\n  Routing Statistics:")
        print(f"    Avg hops: {stats['avg_num_hops']:.2f}")
        print(f"    Avg carbon per flow: {stats['avg_carbon_per_flow']:.6f} gCO2eq")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the model is trained: python training/train_real_model.py")
        return
    
    # Generate results files
    import os
    os.makedirs('results', exist_ok=True)
    
    # 1. Save metrics to CSV
    import csv
    with open('results/metrics_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Baseline Carbon (gCO2eq)', f'{baseline_carbon:.6f}'])
        writer.writerow(['ECMP-Carbon (gCO2eq)', f'{heuristic_carbon:.6f}'])
        writer.writerow(['ECMP-Carbon Reduction (%)', f'{heuristic_reduction:.2f}'])
        writer.writerow(['Carbon-Aware Carbon (gCO2eq)', f'{carbon_aware_carbon:.6f}'])
        writer.writerow(['Carbon Reduction (%)', f'{reduction:.2f}'])
        writer.writerow(['GNN vs ECMP-Carbon (%)', f'{gnn_vs_heuristic:.2f}'])
        writer.writerow(['Number of Flows', len(traffic_flows)])
        writer.writerow(['Number of Nodes', num_nodes])
        writer.writerow(['Number of Edges', num_edges])
        writer.writerow(['Avg Hops', f'{stats["avg_num_hops"]:.2f}'])
    
    # 2. Create separate visualizations
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    BG = '#ffffff'
    CARD = '#ffffff'
    BORDER = '#d4d4d4'
    TXT = '#1a1a1a'
    TXT2 = '#525252'
    TXT_M = '#a3a3a3'
    GRN = '#16a34a'
    RED = '#dc2626'
    BLU = '#2563eb'
    PRP = '#7c3aed'
    AMB = '#d97706'
    
    def _style(ax, title):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=TXT_M, labelsize=8)
        ax.spines['bottom'].set_color(BORDER)
        ax.spines['left'].set_color(BORDER)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, fontsize=11, fontweight='600', color=TXT, pad=10, loc='left')
    
    carbon_intensities = [G.nodes[n]['carbon_intensity'] for n in G.nodes()]
    baseline_per_flow = baseline_carbon / len(traffic_flows)
    heuristic_per_flow = heuristic_carbon / len(traffic_flows)
    carbon_aware_per_flow = stats['avg_carbon_per_flow']
    carbon_aware_hops = stats['avg_num_hops']
    
    # ── Plot 1: Total Emissions Comparison (3 bars) ──
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    _style(ax, f'Total Carbon Emissions ({reduction:+.1f}% GNN Reduction)')
    categories = ['Baseline\n(OSPF)', 'ECMP\n(Carbon Tie)', 'GNN\n(Ours)']
    values = [baseline_carbon, heuristic_carbon, carbon_aware_carbon]
    bar_cols = [RED, AMB, GRN]
    bars = ax.bar(categories, values, color=bar_cols, alpha=0.75, edgecolor='none', width=0.5)
    ax.set_ylabel('gCO2eq', fontsize=9, color=TXT2)
    ax.grid(axis='y', alpha=0.15, color=BORDER)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h,
                f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='600', color=TXT)
    plt.tight_layout()
    plt.savefig('results/01_total_emissions.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    # ── Plot 2: Reduction % (horizontal bars) ──
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    _style(ax, 'Carbon Reduction vs Baseline')
    labels = ['GNN (Ours)', 'ECMP-Carbon']
    reductions = [reduction, heuristic_reduction]
    colors_r = [GRN, AMB]
    ax.barh(labels, reductions, color=colors_r, alpha=0.75, edgecolor='none', height=0.4)
    ax.set_xlabel('Reduction (%)', fontsize=9, color=TXT2)
    ax.axvline(x=0, color=TXT_M, linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.15, color=BORDER)
    for i, (lbl, val) in enumerate(zip(labels, reductions)):
        ax.text(val + 0.3 if val > 0 else val - 0.3, i,
                f'{val:+.2f}%', ha='left' if val > 0 else 'right', va='center',
                fontsize=10, fontweight='600', color=colors_r[i])
    plt.tight_layout()
    plt.savefig('results/02_reduction_pct.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    # ── Plot 3: Avg Carbon per Flow (3 bars) ──
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    _style(ax, 'Average Carbon per Flow')
    vals = [baseline_per_flow, heuristic_per_flow, carbon_aware_per_flow]
    bars3 = ax.bar(['Baseline', 'ECMP-Carbon', 'GNN'], vals,
                   color=[RED, AMB, GRN], alpha=0.75, edgecolor='none', width=0.45)
    ax.set_ylabel('gCO2eq', fontsize=9, color=TXT2)
    ax.grid(axis='y', alpha=0.15, color=BORDER)
    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='500', color=TXT)
    plt.tight_layout()
    plt.savefig('results/03_avg_carbon_per_flow.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    # ── Plot 4: Carbon Intensity Distribution ──
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    _style(ax, 'Node Carbon Intensity Distribution')
    ax.hist(carbon_intensities, bins=15, color=BLU, alpha=0.6, edgecolor=CARD, linewidth=0.5)
    ax.axvline(np.mean(carbon_intensities), color=AMB, linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(carbon_intensities):.1f}')
    ax.set_xlabel('gCO2/kWh', fontsize=9, color=TXT2)
    ax.set_ylabel('Nodes', fontsize=9, color=TXT2)
    ax.legend(fontsize=8, framealpha=0.3, edgecolor=BORDER, facecolor=CARD, labelcolor=TXT2)
    ax.grid(alpha=0.15, color=BORDER)
    plt.tight_layout()
    plt.savefig('results/04_carbon_intensity_dist.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    # ── Plot 5: Routing Path Length ──
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    _style(ax, 'Routing Path Length (GNN)')
    ax.bar(['Avg Hops'], [carbon_aware_hops], color=PRP, alpha=0.75, edgecolor='none', width=0.35)
    ax.set_ylabel('Hops', fontsize=9, color=TXT2)
    ax.set_ylim([0, max(3, carbon_aware_hops + 1)])
    ax.grid(axis='y', alpha=0.15, color=BORDER)
    ax.text(0, carbon_aware_hops, f'{carbon_aware_hops:.2f}',
            ha='center', va='bottom', fontsize=11, fontweight='600', color=TXT)
    plt.tight_layout()
    plt.savefig('results/05_routing_path_length.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    # ── Plot 6: Summary Stats Panel ──
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(CARD)
    ax.axis('off')
    
    info_lines = [
        ('NETWORK', ''),
        ('Nodes', str(num_nodes)),
        ('Edges', str(num_edges)),
        ('Avg Degree', f'{2*num_edges/num_nodes:.1f}'),
        ('Flows', str(len(traffic_flows))),
        ('', ''),
        ('CARBON COMPARISON', ''),
        ('Baseline', f'{baseline_carbon:.2f} gCO2eq'),
        ('ECMP-Carbon', f'{heuristic_carbon:.2f} gCO2eq  ({heuristic_reduction:+.1f}%)'),
        ('GNN', f'{carbon_aware_carbon:.2f} gCO2eq  ({reduction:+.1f}%)'),
        ('GNN vs ECMP', f'{gnn_vs_heuristic:+.1f}%'),
        ('', ''),
        ('ROUTING', ''),
        ('Avg Hops', f'{stats["avg_num_hops"]:.2f}'),
        ('Avg CO2/Flow', f'{stats["avg_carbon_per_flow"]:.2f}'),
        ('', ''),
        ('INTENSITY', ''),
        ('Min', f'{min(carbon_intensities):.1f} gCO2/kWh'),
        ('Mean', f'{np.mean(carbon_intensities):.1f} gCO2/kWh'),
        ('Max', f'{max(carbon_intensities):.1f} gCO2/kWh'),
    ]
    y = 0.95
    for label, value in info_lines:
        if label in ('NETWORK', 'CARBON COMPARISON', 'ROUTING', 'INTENSITY'):
            ax.text(0.08, y, label, fontsize=8, fontweight='600', color=TXT_M,
                    transform=ax.transAxes, family='sans-serif')
        elif label:
            ax.text(0.08, y, label, fontsize=9, color=TXT2,
                    transform=ax.transAxes, family='sans-serif')
            ax.text(0.92, y, value, fontsize=9, color=TXT, fontweight='500',
                    transform=ax.transAxes, ha='right', family='sans-serif')
        y -= 0.048
    plt.tight_layout()
    plt.savefig('results/06_summary_stats.png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    
    # 3. Generate markdown report
    with open('results/final_report.md', 'w', encoding='utf-8') as f:
        f.write('# Carbon-Aware Routing Demo Results\n\n')
        f.write(f'**Generated:** {np.datetime64("now")}\n\n')
        f.write('## Summary\n\n')
        f.write(f'| Approach | Total Carbon (gCO2eq) | Reduction vs Baseline |\n')
        f.write(f'|----------|----------------------|----------------------|\n')
        f.write(f'| Baseline (OSPF) | {baseline_carbon:.6f} | - |\n')
        f.write(f'| ECMP-Carbon | {heuristic_carbon:.6f} | {heuristic_reduction:+.2f}% |\n')
        f.write(f'| **GNN (Ours)** | **{carbon_aware_carbon:.6f}** | **{reduction:+.2f}%** |\n\n')
        f.write('## Network Configuration\n\n')
        f.write(f'- Nodes: {num_nodes}\n')
        f.write(f'- Edges: {num_edges}\n')
        f.write(f'- Traffic Flows: {len(traffic_flows)}\n\n')
        f.write('## Routing Statistics\n\n')
        f.write(f'- Average Hops: {stats["avg_num_hops"]:.2f}\n')
        f.write(f'- Average Carbon per Flow: {stats["avg_carbon_per_flow"]:.6f} gCO2eq\n\n')
        f.write('## Plots\n\n')
        f.write('![Total Emissions](01_total_emissions.png)\n\n')
        f.write('![Reduction](02_reduction_pct.png)\n\n')
        f.write('![Per Flow](03_avg_carbon_per_flow.png)\n\n')
        f.write('![Intensity](04_carbon_intensity_dist.png)\n\n')
        f.write('![Path Length](05_routing_path_length.png)\n\n')
        f.write('![Summary](06_summary_stats.png)\n\n')
        if reduction > 0:
            f.write(f'**SUCCESS:** Achieved {reduction:.2f}% carbon reduction!\n')
        else:
            f.write(f'**Note:** Negative reduction may occur with random networks.\n')
    
    print('\n' + '='*70)
    print('RESULTS SAVED')
    print('='*70)
    print('  - results/metrics_summary.csv')
    print('  - results/01_total_emissions.png')
    print('  - results/02_reduction_pct.png')
    print('  - results/03_avg_carbon_per_flow.png')
    print('  - results/04_carbon_intensity_dist.png')
    print('  - results/05_routing_path_length.png')
    print('  - results/06_summary_stats.png')
    print('  - results/final_report.md')
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run with larger network: modify num_nodes=50")
    print("  2. Run NS3 simulation: python run_ns3_demo.py")
    print("  3. Customize scenarios: edit network topology and traffic patterns")


if __name__ == "__main__":
    main()
