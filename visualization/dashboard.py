import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Clean white theme
COLORS = {
    'bg': '#ffffff',
    'card': '#ffffff',
    'surface': '#f5f5f5',
    'border': '#d4d4d4',
    'text': '#1a1a1a',
    'text_secondary': '#525252',
    'text_muted': '#a3a3a3',
    'green': '#16a34a',
    'red': '#dc2626',
    'blue': '#2563eb',
    'purple': '#7c3aed',
    'amber': '#d97706',
    'teal': '#0d9488',
}


def _apply_style(ax, title=None):
    """Apply consistent clean styling to an axis."""
    ax.set_facecolor(COLORS['card'])
    ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
    ax.spines['bottom'].set_color(COLORS['border'])
    ax.spines['left'].set_color(COLORS['border'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color=COLORS['text'], pad=10, loc='left')


class CarbonDashboard:
    def __init__(self, topology, figsize=(16, 10)):
        self.topology = topology
        self.fig = plt.figure(figsize=figsize, facecolor=COLORS['bg'])
        self.setup_layout()
        
    def setup_layout(self):
        gs = self.fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        self.ax_network = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_carbon = self.fig.add_subplot(gs[0, 2])
        self.ax_comparison = self.fig.add_subplot(gs[1, 2])
        self.ax_metrics = self.fig.add_subplot(gs[2, :])
        
        _apply_style(self.ax_network, 'Network Topology - Carbon Intensity')
        _apply_style(self.ax_carbon, 'Current Emission Rate')
        _apply_style(self.ax_comparison, 'Total Emissions')
        _apply_style(self.ax_metrics, 'Carbon Emission Timeline')
        
        self.ax_network.axis('off')
    
    def plot_network(self, carbon_intensities, highlighted_paths=None):
        self.ax_network.clear()
        self.ax_network.set_facecolor(COLORS['card'])
        self.ax_network.axis('off')
        _apply_style(self.ax_network, 'Network Topology - Carbon Intensity')
        
        G = self.topology['graph']
        pos = self.topology['positions']
        
        node_colors = [carbon_intensities.get(node, 350) for node in G.nodes()]
        
        nx.draw_networkx_edges(G, pos, ax=self.ax_network, 
                              edge_color=COLORS['border'], width=1, alpha=0.5)
        
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('carbon',
            ['#16a34a', '#d97706', '#dc2626'])
        
        nodes = nx.draw_networkx_nodes(G, pos, ax=self.ax_network,
                                       node_color=node_colors, 
                                       node_size=400, 
                                       cmap=cmap, 
                                       vmin=200, vmax=600,
                                       edgecolors=COLORS['border'], linewidths=1)
        
        nx.draw_networkx_labels(G, pos, ax=self.ax_network, 
                               font_size=8, font_weight='bold',
                               font_color=COLORS['text'])
        
        if highlighted_paths:
            for path in highlighted_paths:
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                                      ax=self.ax_network, edge_color=COLORS['blue'],
                                      width=2.5, alpha=0.8)
        
        cbar = plt.colorbar(nodes, ax=self.ax_network, shrink=0.6, pad=0.02)
        cbar.set_label('Carbon Intensity (gCO2/kWh)', fontsize=8, color=COLORS['text_secondary'])
        cbar.ax.tick_params(colors=COLORS['text_secondary'], labelsize=7)
        cbar.outline.set_edgecolor(COLORS['border'])
    
    def plot_carbon_timeline(self, carbon_aware_results, baseline_results=None):
        ax = self.ax_metrics
        ax.clear()
        _apply_style(ax, 'Carbon Emission Timeline')
        
        timestamps = np.array(carbon_aware_results['timestamps']) / 3600
        carbon_history = carbon_aware_results['carbon_history']
        
        ax.fill_between(timestamps, carbon_history, alpha=0.1, color=COLORS['blue'])
        ax.plot(timestamps, carbon_history, 
               color=COLORS['blue'], linewidth=1.5, label='Carbon-Aware (GNN)',
               marker='.', markersize=3)
        
        if baseline_results:
            baseline_carbon = baseline_results['carbon_history']
            ax.plot(timestamps[:len(baseline_carbon)], baseline_carbon, 
                   color=COLORS['red'], linewidth=1.5, label='Baseline (Shortest Path)',
                   linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Time (hours)', fontsize=9, color=COLORS['text_secondary'])
        ax.set_ylabel('Carbon (gCO2)', fontsize=9, color=COLORS['text_secondary'])
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9,
                 edgecolor=COLORS['border'], facecolor=COLORS['card'],
                 labelcolor=COLORS['text_secondary'])
        ax.grid(True, alpha=0.15, color=COLORS['border'])
    
    def plot_carbon_gauge(self, current_carbon, max_carbon=100):
        """Horizontal bar showing current emission rate."""
        ax = self.ax_carbon
        ax.clear()
        _apply_style(ax, 'Current Emission Rate')
        
        pct = min(100, (current_carbon / max_carbon) * 100)
        
        # Background bar
        ax.barh(0, 100, height=0.5, color=COLORS['surface'], edgecolor=COLORS['border'], linewidth=0.5)
        
        # Fill bar
        if pct < 40:
            bar_color = COLORS['green']
        elif pct < 70:
            bar_color = COLORS['amber']
        else:
            bar_color = COLORS['red']
            
        ax.barh(0, pct, height=0.5, color=bar_color, edgecolor='none', alpha=0.8)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        
        ax.text(50, -0.45, f'{current_carbon:.2f} gCO2',
               ha='center', va='center', fontsize=13, fontweight='bold',
               color=COLORS['text'])
        ax.text(50, 0.45, f'{pct:.0f}% of maximum',
               ha='center', va='center', fontsize=8,
               color=COLORS['text_muted'])
    
    def plot_comparison(self, carbon_aware_total, baseline_total):
        ax = self.ax_comparison
        ax.clear()
        _apply_style(ax, 'Total Emissions')
        
        reduction = ((baseline_total - carbon_aware_total) / baseline_total) * 100
        
        categories = ['GNN', 'Baseline']
        values = [carbon_aware_total, baseline_total]
        colors = [COLORS['blue'], COLORS['red']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.75,
                     edgecolor=COLORS['border'], linewidth=0.5, width=0.5)
        
        ax.set_ylabel('gCO2', fontsize=8, color=COLORS['text_secondary'])
        ax.grid(axis='y', alpha=0.15, color=COLORS['border'])
        
        ax.text(0.5, max(values) * 1.05, f'{reduction:+.1f}%',
               ha='center', fontsize=10, fontweight='bold',
               color=COLORS['green'] if reduction > 0 else COLORS['red'])
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                   f'{value:.1f}',
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   color='white', alpha=0.9)
    
    def show(self):
        plt.tight_layout()
        plt.show()
    
    def save(self, filename):
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved dashboard to {filename}")


def create_comparison_plot(carbon_aware_results, baseline_results, output_file=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=COLORS['bg'])
    
    timestamps = np.array(carbon_aware_results['timestamps']) / 3600
    ca_carbon = carbon_aware_results['carbon_history']
    bl_carbon = baseline_results['carbon_history']
    
    # 1. Emission timeline
    ax = axes[0, 0]
    _apply_style(ax, 'Carbon Emission Over Time')
    ax.fill_between(timestamps, ca_carbon, alpha=0.08, color=COLORS['blue'])
    ax.plot(timestamps, ca_carbon, color=COLORS['blue'], linewidth=1.5,
           label='Carbon-Aware', marker='.', markersize=2)
    ax.plot(timestamps[:len(bl_carbon)], bl_carbon, color=COLORS['red'],
           linewidth=1.5, label='Baseline', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time (hours)', fontsize=8, color=COLORS['text_secondary'])
    ax.set_ylabel('gCO2', fontsize=8, color=COLORS['text_secondary'])
    ax.legend(fontsize=7, framealpha=0.9, edgecolor=COLORS['border'],
             facecolor=COLORS['card'], labelcolor=COLORS['text_secondary'])
    ax.grid(True, alpha=0.15, color=COLORS['border'])
    
    # 2. Cumulative
    ax = axes[0, 1]
    _apply_style(ax, 'Cumulative Emissions')
    cumulative_ca = np.cumsum(ca_carbon)
    cumulative_bl = np.cumsum(bl_carbon)
    ax.fill_between(timestamps, cumulative_ca, alpha=0.08, color=COLORS['blue'])
    ax.plot(timestamps, cumulative_ca, color=COLORS['blue'], linewidth=1.5, label='Carbon-Aware')
    ax.plot(timestamps[:len(cumulative_bl)], cumulative_bl, color=COLORS['red'],
           linewidth=1.5, label='Baseline', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time (hours)', fontsize=8, color=COLORS['text_secondary'])
    ax.set_ylabel('Cumulative gCO2', fontsize=8, color=COLORS['text_secondary'])
    ax.legend(fontsize=7, framealpha=0.9, edgecolor=COLORS['border'],
             facecolor=COLORS['card'], labelcolor=COLORS['text_secondary'])
    ax.grid(True, alpha=0.15, color=COLORS['border'])
    
    # 3. Reduction percentage
    ax = axes[1, 0]
    _apply_style(ax, 'Hourly Carbon Reduction')
    min_len = min(len(ca_carbon), len(bl_carbon))
    reduction = ((np.array(bl_carbon[:min_len]) - np.array(ca_carbon[:min_len])) 
                / np.array(bl_carbon[:min_len])) * 100
    ax.fill_between(timestamps[:min_len], reduction, alpha=0.1, color=COLORS['green'])
    ax.plot(timestamps[:min_len], reduction, color=COLORS['green'], linewidth=1.5, marker='.', markersize=2)
    ax.axhline(y=0, color=COLORS['text_muted'], linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Time (hours)', fontsize=8, color=COLORS['text_secondary'])
    ax.set_ylabel('Reduction (%)', fontsize=8, color=COLORS['text_secondary'])
    ax.grid(True, alpha=0.15, color=COLORS['border'])
    
    # 4. Total comparison bar
    ax = axes[1, 1]
    _apply_style(ax, 'Total Carbon Comparison')
    categories = ['GNN\nRouting', 'Baseline\nShortest Path']
    totals = [carbon_aware_results['total_carbon'], baseline_results['total_carbon']]
    bar_colors = [COLORS['blue'], COLORS['red']]
    bars = ax.bar(categories, totals, color=bar_colors, alpha=0.75,
                 edgecolor=COLORS['border'], linewidth=0.5, width=0.45)
    ax.set_ylabel('Total gCO2', fontsize=8, color=COLORS['text_secondary'])
    ax.grid(axis='y', alpha=0.15, color=COLORS['border'])
    
    pct = ((totals[1] - totals[0]) / totals[1] * 100)
    ax.text(0.5, max(totals) * 1.08,
           f'Reduction: {pct:.1f}%',
           ha='center', fontsize=10, fontweight='bold',
           color=COLORS['green'] if pct > 0 else COLORS['red'],
           transform=ax.get_xaxis_transform())
    
    for bar, value in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold',
               color=COLORS['text'])
    
    fig.suptitle('Carbon-Aware Routing - Performance Analysis',
                fontsize=13, fontweight='bold', color=COLORS['text'], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved comparison plot to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    print("Dashboard Test")
    print("=" * 50)
    
    from simulation.network_topology import create_network
    
    topology = create_network(12, 'hierarchical')
    
    carbon_intensities = {i: 300 + np.random.randint(-50, 100) for i in range(12)}
    
    dashboard = CarbonDashboard(topology)
    dashboard.plot_network(carbon_intensities)
    
    mock_results = {
        'timestamps': list(range(0, 86400, 3600)),
        'carbon_history': np.random.uniform(10, 30, 24),
        'total_carbon': 500
    }
    
    baseline_results = {
        'carbon_history': np.random.uniform(15, 40, 24),
        'total_carbon': 650
    }
    
    dashboard.plot_carbon_timeline(mock_results, baseline_results)
    dashboard.plot_carbon_gauge(25)
    dashboard.plot_comparison(500, 650)
    
    print("\nDashboard test passed (close window to continue)")
    dashboard.show()
