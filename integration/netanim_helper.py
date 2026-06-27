"""
NetAnim Integration - ns-3 Network Animation Visualization (ns-3.41+)

NetAnim is ns-3's official network animator that provides:
- Packet-level animation (visualize packet flows)
- Node movement tracking
- Queue occupancy visualization
- Link utilization graphs
- Real-time statistics

This module integrates NetAnim with our carbon-aware routing simulation
to provide visual validation of routing decisions.
"""

try:
    from ns import ns
    NS3_AVAILABLE = True
except ImportError:
    NS3_AVAILABLE = False
    ns = None

import os


class NetAnimConfig:
    """Configuration for NetAnim visualization"""
    
    def __init__(self, output_file="carbon-routing-animation.xml"):
        self.output_file = output_file
        self.enable_packet_metadata = True
        self.enable_ip_addresses = True
        self.update_interval = 1.0
        self.max_packets_per_trace = 10000
        
        self.node_colors = {
            'solar_heavy': (0, 255, 0),      # Green
            'wind_heavy': (100, 200, 255),   # Light blue
            'coal_heavy': (255, 50, 50),     # Red
            'nuclear': (255, 200, 0),        # Yellow
            'hydro': (0, 150, 255),          # Blue
            'mixed_grid': (200, 200, 200)    # Gray
        }
        
        self.node_sizes = {
            'core': 50,
            'aggregation': 40,
            'access': 30
        }


class NetAnimHelper:
    """Helper class for NetAnim integration with carbon-aware routing"""
    
    def __init__(self, config=None):
        if not NS3_AVAILABLE:
            raise RuntimeError("ns-3 not available - cannot use NetAnim")
        
        self.config = config or NetAnimConfig()
        self.anim = None
        self.node_descriptions = {}
        
    def initialize(self, output_file=None):
        """Initialize NetAnim animation helper"""
        if output_file:
            self.config.output_file = output_file
        
        self.anim = ns.AnimationInterface(self.config.output_file)
        
        if self.config.enable_packet_metadata:
            try:
                self.anim.EnablePacketMetadata(True)
            except Exception:
                pass
        
        if self.config.enable_ip_addresses:
            try:
                self.anim.EnableIpv4RouteTracking(
                    self.config.output_file.replace('.xml', '-routing.xml'),
                    ns.TimeStep(0),
                    ns.TimeStep(int(self.config.update_interval * 1e9)),
                    ns.TimeStep(int(3600 * 1e9))
                )
            except Exception as e:
                print(f"  Route tracking disabled: {e}")
        
        print(f"NetAnim initialized: {self.config.output_file}")
        return self.anim
    
    def set_node_positions(self, topology):
        """Set node positions for NetAnim visualization"""
        if not self.anim:
            raise RuntimeError("NetAnim not initialized")
        
        positions = topology['positions']
        
        scale_x = 1000
        scale_y = 800
        
        for node_id, (x, y) in positions.items():
            scaled_x = x * scale_x
            scaled_y = y * scale_y
            
            self.anim.SetConstantPosition(
                ns.NodeContainer.GetGlobal().Get(node_id),
                scaled_x,
                scaled_y
            )
    
    def set_carbon_aware_colors(self, carbon_assignments):
        """Color nodes based on their carbon profile"""
        if not self.anim:
            raise RuntimeError("NetAnim not initialized")
        
        for node_id, profile_name in carbon_assignments.items():
            color = self.config.node_colors.get(profile_name, (128, 128, 128))
            
            self.anim.UpdateNodeColor(
                ns.NodeContainer.GetGlobal().Get(node_id),
                color[0], color[1], color[2]
            )
            
            self.anim.UpdateNodeDescription(
                ns.NodeContainer.GetGlobal().Get(node_id),
                f"Node {node_id} ({profile_name})"
            )
    
    def set_node_sizes(self, node_types):
        """Set node sizes based on their role (core/aggregation/access)"""
        if not self.anim:
            raise RuntimeError("NetAnim not initialized")
        
        for node_id, node_type in node_types.items():
            size = self.config.node_sizes.get(node_type, 35)
            
            self.anim.UpdateNodeSize(
                ns.NodeContainer.GetGlobal().Get(node_id),
                size, size
            )
    
    def add_route_update_marker(self, timestamp, num_routes_changed):
        """Add a text annotation when routes are updated by GNN"""
        if not self.anim:
            return
        
        text = f"GNN Update: {num_routes_changed} routes changed"
        
        self.anim.AddResource(
            f"RouteUpdate_{timestamp}",
            text
        )
    
    def update_link_utilization(self, edge_index, link_weights, timestamp):
        """Update link colors based on routing weights (carbon optimization)"""
        if not self.anim:
            return
        
        for idx in range(edge_index.shape[1]):
            weight = link_weights[idx]
            
            if weight < 5:
                color = (0, 255, 0)
            elif weight < 20:
                color = (100, 200, 100)
            elif weight < 50:
                color = (255, 255, 0)
            else:
                color = (255, 100, 100)
            
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            
            desc = f"Link {src}->{dst}: weight={weight} (t={timestamp}s)"
            self.anim.UpdateLinkDescription(src, dst, desc)


class CarbonAwareNetAnimSimulation:
    """Complete simulation with NetAnim visualization"""
    
    def __init__(self, gnn_model, topology, carbon_manager, config=None):
        self.gnn_model = gnn_model
        self.topology = topology
        self.carbon_manager = carbon_manager
        self.netanim_helper = NetAnimHelper(config)
        
    def setup_visualization(self, output_file="results/carbon-routing.xml"):
        """Setup NetAnim with carbon-aware coloring"""
        
        self.netanim_helper.initialize(output_file)
        
        self.netanim_helper.set_node_positions(self.topology)
        
        carbon_assignments = {
            i: self.carbon_manager.node_assignments.get(i, 'mixed_grid')
            for i in range(self.topology['graph'].number_of_nodes())
        }
        self.netanim_helper.set_carbon_aware_colors(carbon_assignments)
        
        node_types = self._classify_node_types()
        self.netanim_helper.set_node_sizes(node_types)
        
        print(f"NetAnim visualization configured")
        print(f"  Output: {output_file}")
        print(f"  Open with: netanim {output_file}")
    
    def _classify_node_types(self):
        """Classify nodes as core/aggregation/access based on degree"""
        num_nodes = self.topology['graph'].number_of_nodes()
        degrees = [self.topology['graph'].degree(i) for i in range(num_nodes)]
        
        avg_degree = sum(degrees) / len(degrees)
        
        node_types = {}
        for i in range(num_nodes):
            if degrees[i] >= avg_degree * 1.5:
                node_types[i] = 'core'
            elif degrees[i] >= avg_degree * 0.7:
                node_types[i] = 'aggregation'
            else:
                node_types[i] = 'access'
        
        return node_types
    
    def run_with_animation(self, duration_seconds=3600):
        """Run simulation and generate NetAnim trace"""
        
        from integration.ns3_bindings import run_ns3_simulation
        
        self.setup_visualization()
        
        results = run_ns3_simulation(
            self.gnn_model,
            self.topology,
            self.carbon_manager,
            duration_seconds,
            control_interval=600
        )
        
        print(f"\nSimulation complete with NetAnim trace")
        print(f"  View animation: netanim {self.netanim_helper.config.output_file}")
        
        return results


def export_netanim_compatible_trace(simulation_results, topology, output_file="results/sim-trace.xml"):
    """Export simulation results in NetAnim-compatible XML format (fallback for non-ns3 mode)"""
    
    import xml.etree.ElementTree as ET
    
    root = ET.Element("animations")
    
    info = ET.SubElement(root, "information")
    ET.SubElement(info, "attribute", {"name": "name", "value": "Carbon-Aware Routing"})
    ET.SubElement(info, "attribute", {"name": "version", "value": "1.0"})
    
    topology_elem = ET.SubElement(root, "topology")
    
    for node_id in range(topology['graph'].number_of_nodes()):
        x, y = topology['positions'].get(node_id, (0.5, 0.5))
        
        node = ET.SubElement(topology_elem, "node")
        node.set("id", str(node_id))
        node.set("x", str(x * 1000))
        node.set("y", str(y * 800))
        node.set("r", "30")
        node.set("g", "30")
        node.set("b", "200")
    
    for u, v in topology['graph'].edges():
        link = ET.SubElement(topology_elem, "link")
        link.set("from", str(u))
        link.set("to", str(v))
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    print(f"Exported NetAnim-compatible trace: {output_file}")
    print(f"  Note: For full NetAnim features, use ns-3 mode")


if __name__ == "__main__":
    print("NetAnim Integration Module")
    print("=" * 60)
    
    if NS3_AVAILABLE:
        print("ns-3 with NetAnim available")
        print("\nUsage:")
        print("  from integration.netanim_helper import CarbonAwareNetAnimSimulation")
        print("  sim = CarbonAwareNetAnimSimulation(model, topology, carbon_mgr)")
        print("  sim.run_with_animation(duration_seconds=3600)")
        print("\nView animation:")
        print("  netanim results/carbon-routing.xml")
    else:
        print("ns-3 not available")
        print("\nFallback mode:")
        print("  Can export static topology trace")
        print("  Use Python dashboard for visualization instead")
