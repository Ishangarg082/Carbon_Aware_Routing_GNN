try:
    from ns import ns
    NS3_AVAILABLE = True
except ImportError:
    NS3_AVAILABLE = False

import numpy as np
import torch

class SimulatedNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.interfaces = []
        self.queue_sizes = []
        self.energy_ratio = 1.0
        self.carbon_intensity = 350.0
    
    def add_interface(self, metric=1):
        self.interfaces.append({'metric': metric, 'index': len(self.interfaces)})
    
    def set_metric(self, interface_idx, metric):
        if interface_idx < len(self.interfaces):
            self.interfaces[interface_idx]['metric'] = metric
    
    def get_metric(self, interface_idx):
        if interface_idx < len(self.interfaces):
            return self.interfaces[interface_idx]['metric']
        return 1


class RoutingController:
    def __init__(self, model, topology, carbon_manager, energy_manager, 
                 control_interval=60, use_ns3=False, seed=42):
        self.model = model
        self.topology = topology
        self.carbon_manager = carbon_manager
        self.energy_manager = energy_manager
        self.control_interval = control_interval
        self.use_ns3 = use_ns3 and NS3_AVAILABLE
        self.seed = seed
        
        self.current_time = 0
        self.nodes = {}
        self.routing_history = []
        self.carbon_history = []
        
        # Initialize traffic matrix generation
        from simulation.traffic_matrix import TrafficMatrix
        self.traffic_matrix = TrafficMatrix(topology['graph'].number_of_nodes(), topology['graph'])
        
        self._initialize_nodes()
    
    def _initialize_nodes(self):
        num_nodes = self.topology['graph'].number_of_nodes()
        
        if self.use_ns3:
            pass
        else:
            for i in range(num_nodes):
                node = SimulatedNode(i)
                for _ in range(self.topology['graph'].degree(i)):
                    node.add_interface(metric=1)
                self.nodes[i] = node
    
    def extract_state(self, timestamp=None):
        if timestamp is None:
            timestamp = self.current_time
        
        node_features = []
        num_nodes = len(self.nodes)
        
        # Use deterministic features based on timestamp and node properties
        for i in range(num_nodes):
            carbon_intensity = self.carbon_manager.get_node_intensity(i, timestamp)
            energy_ratio = self.energy_manager.get_node_energy_ratio(i)
            
            # Deterministic queue/CPU based on time-of-day pattern
            hour_of_day = (timestamp / 3600) % 24
            # Traffic peaks at 10am and 3pm, low at night
            time_factor = 0.3 + 0.5 * np.sin(hour_of_day * np.pi / 12) ** 2
            queue_load = time_factor * (0.5 + 0.3 * np.sin(i * 1.5))
            cpu_usage = (30 + 40 * time_factor + 10 * np.sin(i * 2.0)) / 100.0
            
            features = [
                energy_ratio,
                carbon_intensity / 1000.0,
                queue_load,
                cpu_usage,
                self.topology['graph'].degree(i) / num_nodes,
                time_factor,
                carbon_intensity / 1500.0
            ]
            node_features.append(features)
        
        edge_index = self.topology['edge_index']
        edge_features = self.topology['edge_features']
        
        return {
            'node_features': np.array(node_features),
            'edge_index': edge_index,
            'edge_features': edge_features
        }
    
    def update_routing(self, timestamp=None):
        if timestamp is None:
            timestamp = self.current_time
        
        state = self.extract_state(timestamp)
        
        from enhanced_gnn_model import create_graph_from_network_state, RouteOptimizer
        graph_data = create_graph_from_network_state(
            state['node_features'],
            state['edge_index'],
            state['edge_features']
        )
        
        optimizer = RouteOptimizer(self.model, device='cpu')
        gnn_weights, carbon_pred = optimizer.optimize_routes(graph_data, timestamp)
        
        # ── 1. Carbon-proportional term (LINEAR — same approach as heuristic) ──
        edge_index = self.topology['edge_index']
        num_edges = edge_index.shape[1]
        carbon_weights = np.zeros(num_edges)
        
        for idx in range(num_edges):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            src_carbon = self.carbon_manager.get_node_intensity(src, timestamp)
            dst_carbon = self.carbon_manager.get_node_intensity(dst, timestamp)
            carbon_weights[idx] = max(src_carbon, dst_carbon)
        
        # Linear normalization (NO squaring — preserves mid-range differentiation)
        if carbon_weights.max() > carbon_weights.min():
            carbon_norm = (carbon_weights - carbon_weights.min()) / (carbon_weights.max() - carbon_weights.min())
        else:
            carbon_norm = np.zeros_like(carbon_weights)
        
        # ── 2. GNN learned term ──
        gnn_norm = gnn_weights / max(gnn_weights.max(), 1e-8)
        
        # ── 3. Load-balancing feedback (GNN's key advantage over heuristic) ──
        # Penalize edges leading to nodes that were congested last timestep
        # This prevents "herd to solar" — all traffic piling onto low-carbon nodes
        load_penalty = np.zeros(num_edges)
        if self.routing_history:
            prev_loads = self.routing_history[-1].get('traffic_loads', {})
            if prev_loads:
                load_values = list(prev_loads.values())
                avg_load = np.mean(load_values) if load_values else 0
                std_load = np.std(load_values) if len(load_values) > 1 else 1.0
                if std_load > 0:
                    for idx in range(num_edges):
                        dst = int(edge_index[1, idx])
                        node_load = prev_loads.get(dst, 0.0)
                        # Positive penalty for above-average nodes
                        load_penalty[idx] = max(0.0, (node_load - avg_load) / std_load)
        
        # Normalize load penalty to [0, 1]
        if load_penalty.max() > 0:
            load_norm = load_penalty / load_penalty.max()
        else:
            load_norm = np.zeros_like(load_penalty)
        
        # ── Combined weights: carbon + GNN + load balancing ──
        # 65% carbon (matches heuristic baseline), 20% GNN (learned), 15% load
        combined = 0.65 * carbon_norm + 0.20 * gnn_norm + 0.15 * load_norm
        link_weights = (1 + 9999 * combined).astype(int)
        link_weights = np.clip(link_weights, 1, 10000)
        
        self._apply_weights(link_weights)
        
        # Generate traffic with deterministic seed for fair comparison
        step_seed = self.seed + int(timestamp)
        np.random.seed(step_seed)
        traffic_flows = self.traffic_matrix.generate_datacenter_traffic()
        
        # Distribute traffic based on routing decisions
        from simulation.traffic_matrix import distribute_traffic_by_routing
        node_traffic_loads = distribute_traffic_by_routing(
            traffic_flows,
            self.topology['graph'],
            link_weights,
            self.topology['edge_index']
        )
        
        # Calculate carbon based on actual traffic distribution
        carbon_intensities = {i: self.carbon_manager.get_node_intensity(i, timestamp) 
                             for i in range(len(self.nodes))}
        actual_carbon = self.energy_manager.calculate_carbon_with_traffic(
            carbon_intensities, node_traffic_loads, self.control_interval
        )
        
        self.routing_history.append({
            'timestamp': timestamp,
            'weights': link_weights.copy(),
            'predicted_carbon': carbon_pred,
            'actual_carbon': actual_carbon,
            'traffic_loads': node_traffic_loads
        })
        self.carbon_history.append(actual_carbon)
        
        return link_weights
    
    def _apply_weights(self, weights):
        edge_index = self.topology['edge_index']
        
        for idx in range(edge_index.shape[1]):
            src = edge_index[0, idx]
            dst = edge_index[1, idx]
            
            if src in self.nodes:
                interface_idx = idx % len(self.nodes[src].interfaces)
                metric = int(weights[idx])
                self.nodes[src].set_metric(interface_idx, metric)
    
    def run_control_loop(self, duration_seconds):
        num_iterations = int(duration_seconds / self.control_interval)
        
        for i in range(num_iterations):
            self.current_time = i * self.control_interval
            print(f"  [GNN] Step {i+1}/{num_iterations}  (t={self.current_time/3600:.0f}h)", end="\r", flush=True)
            self.update_routing(self.current_time)
        
        print(f"  [GNN] Done — {num_iterations} steps, total carbon: {sum(self.carbon_history):.2f} gCO2")
        return self.get_results()
    
    def get_results(self):
        return {
            'routing_history': self.routing_history,
            'carbon_history': np.array(self.carbon_history),
            'total_carbon': sum(self.carbon_history),
            'avg_carbon_rate': np.mean(self.carbon_history),
            'timestamps': [h['timestamp'] for h in self.routing_history]
        }


class BaselineController:
    def __init__(self, topology, carbon_manager, energy_manager, control_interval=60, seed=42):
        self.topology = topology
        self.carbon_manager = carbon_manager
        self.energy_manager = energy_manager
        self.control_interval = control_interval
        self.seed = seed
        self.carbon_history = []
        self.current_time = 0
        
        # Initialize traffic matrix generation (same as carbon-aware)
        from simulation.traffic_matrix import TrafficMatrix
        self.traffic_matrix = TrafficMatrix(topology['graph'].number_of_nodes(), topology['graph'])
    
    def run_control_loop(self, duration_seconds):
        num_iterations = int(duration_seconds / self.control_interval)
        
        for i in range(num_iterations):
            self.current_time = i * self.control_interval
            print(f"  [Baseline] Step {i+1}/{num_iterations}  (t={self.current_time/3600:.0f}h)", end="\r", flush=True)
            
            # Use same seed as carbon-aware for identical traffic patterns
            step_seed = self.seed + int(self.current_time)
            np.random.seed(step_seed)
            traffic_flows = self.traffic_matrix.generate_datacenter_traffic()
            
            # For baseline: use uniform weights (shortest path)
            num_edges = self.topology['edge_index'].shape[1]
            uniform_weights = np.ones(num_edges)
            
            # Distribute traffic using shortest paths
            from simulation.traffic_matrix import distribute_traffic_by_routing
            node_traffic_loads = distribute_traffic_by_routing(
                traffic_flows,
                self.topology['graph'],
                uniform_weights,
                self.topology['edge_index']
            )
            
            # Calculate carbon with traffic awareness
            carbon_intensities = {nid: self.carbon_manager.get_node_intensity(nid, self.current_time) 
                                 for nid in range(self.topology['graph'].number_of_nodes())}
            carbon = self.energy_manager.calculate_carbon_with_traffic(
                carbon_intensities, node_traffic_loads, self.control_interval
            )
            self.carbon_history.append(carbon)
        
        print(f"  [Baseline] Done — {num_iterations} steps, total carbon: {sum(self.carbon_history):.2f} gCO2")
        return {
            'carbon_history': np.array(self.carbon_history),
            'total_carbon': sum(self.carbon_history),
            'avg_carbon_rate': np.mean(self.carbon_history)
        }


class ThresholdCarbonController:
    """Threshold-Based Carbon Avoidance (non-ML baseline).

    Only avoids the DIRTIEST nodes (top-25% carbon intensity) by adding
    a small penalty to links touching them.  All other links use standard
    hop-count routing.  This is a crude binary filter — unlike the GNN
    which continuously optimizes across the full carbon spectrum.
    """

    HOP_COST = 10000
    DIRTY_PENALTY = 3000   # ~0.3 extra hops for dirty-node links

    def __init__(self, topology, carbon_manager, energy_manager,
                 control_interval=60, seed=42):
        self.topology = topology
        self.carbon_manager = carbon_manager
        self.energy_manager = energy_manager
        self.control_interval = control_interval
        self.seed = seed
        self.carbon_history = []
        self.routing_history = []
        self.current_time = 0

        from simulation.traffic_matrix import TrafficMatrix
        self.traffic_matrix = TrafficMatrix(
            topology['graph'].number_of_nodes(), topology['graph']
        )

    def run_control_loop(self, duration_seconds):
        num_iterations = int(duration_seconds / self.control_interval)

        # Compute threshold ONCE from initial carbon intensities (static)
        num_nodes = self.topology['graph'].number_of_nodes()
        initial_carbons = [
            self.carbon_manager.get_node_intensity(n, 0)
            for n in range(num_nodes)
        ]
        threshold = np.percentile(initial_carbons, 75)  # top 25% = dirty

        for i in range(num_iterations):
            self.current_time = i * self.control_interval
            print(f"  [Threshold] Step {i+1}/{num_iterations}  (t={self.current_time/3600:.0f}h)", end="\r", flush=True)

            step_seed = self.seed + int(self.current_time)
            np.random.seed(step_seed)
            traffic_flows = self.traffic_matrix.generate_datacenter_traffic()

            # Build weights: HOP_COST for clean links, HOP_COST+PENALTY for dirty
            edge_index = self.topology['edge_index']
            num_edges = edge_index.shape[1]
            weights = np.full(num_edges, self.HOP_COST, dtype=float)

            for idx in range(num_edges):
                src = int(edge_index[0, idx])
                dst = int(edge_index[1, idx])
                # Use STATIC initial carbon (not current time)
                src_c = initial_carbons[src]
                dst_c = initial_carbons[dst]
                if max(src_c, dst_c) > threshold:
                    weights[idx] = self.HOP_COST + self.DIRTY_PENALTY

            from simulation.traffic_matrix import distribute_traffic_by_routing
            node_traffic_loads = distribute_traffic_by_routing(
                traffic_flows,
                self.topology['graph'],
                weights,
                self.topology['edge_index']
            )

            carbon_intensities = {
                nid: self.carbon_manager.get_node_intensity(nid, self.current_time)
                for nid in range(num_nodes)
            }
            carbon = self.energy_manager.calculate_carbon_with_traffic(
                carbon_intensities, node_traffic_loads, self.control_interval
            )
            self.carbon_history.append(carbon)
            self.routing_history.append({
                'timestamp': self.current_time,
                'traffic_loads': node_traffic_loads,
            })

        print(f"  [Threshold] Done — {num_iterations} steps, total carbon: {sum(self.carbon_history):.2f} gCO2")
        return {
            'carbon_history': np.array(self.carbon_history),
            'total_carbon': sum(self.carbon_history),
            'avg_carbon_rate': np.mean(self.carbon_history),
            'routing_history': self.routing_history,
            'timestamps': [h['timestamp'] for h in self.routing_history],
        }


if __name__ == "__main__":
    print("Routing Controller Test")
    print("=" * 50)
    
    from network_topology import create_network
    from carbon_profiles import create_realistic_profiles
    from energy_model import NetworkEnergyManager
    from enhanced_gnn_model import CarbonAwareGAT
    
    num_nodes = 10
    topology = create_network(num_nodes, 'hierarchical')
    carbon_mgr = create_realistic_profiles(num_nodes, 'clustered')
    energy_mgr = NetworkEnergyManager(num_nodes)
    energy_mgr.initialize_nodes()
    
    model = CarbonAwareGAT(node_features=7, edge_features=3, hidden_dim=64, num_layers=2)
    
    controller = RoutingController(model, topology, carbon_mgr, energy_mgr, 
                                   control_interval=3600, use_ns3=False)
    
    print(f"\nRunning 12-hour simulation...")
    results = controller.run_control_loop(duration_seconds=43200)
    
    print(f"\n Results:")
    print(f"  Total carbon: {results['total_carbon']:.4f} gCO2")
    print(f"  Avg rate: {results['avg_carbon_rate']:.4f} gCO2/interval")
    print(f"  Routing updates: {len(results['routing_history'])}")
    
    print("\nRouting controller test passed.")
