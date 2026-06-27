"""
NS3-Compatible Carbon-Aware Routing Controller

Integrates trained CarbonPredictor model with NS3 for real-time
carbon-optimized routing decisions.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
from models.carbon_predictor import CarbonRoutingOptimizer


class CarbonAwareRoutingController:
    """
    Carbon-aware routing controller for NS3 integration
    
    Uses trained machine learning model to predict carbon emissions
    and select routes that minimize environmental impact.
    """
    
    def __init__(self, model_path='models/best_carbon_predictor.pth',
                 scaler_path='models/feature_scaler.pkl'):
        """
        Initialize controller with trained model
        
        Args:
            model_path: Path to trained PyTorch model
            scaler_path: Path to feature scaler
        """
        self.optimizer = CarbonRoutingOptimizer(model_path)
        self.optimizer.load_scaler(scaler_path)
        
        self.routing_history = []
        self.carbon_saved = 0
        
        print(f"Carbon-aware routing controller initialized")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
    
    def find_candidate_routes(self, graph, src, dst, k=3):
        """
        Find k shortest paths between src and dst
        
        Args:
            graph: NetworkX graph with 'carbon_intensity' node attribute
            src: Source node ID
            dst: Destination node ID
            k: Number of candidate paths
        
        Returns:
            List of candidate routes with carbon_intensity
        """
        try:
            # Use faster simple approach: find shortest path, then try variations
            candidates = []
            
            # 1. Shortest path (hop count)
            try:
                path = nx.shortest_path(graph, src, dst)
                carbon_intensities = [graph.nodes[node].get('carbon_intensity', 400) 
                                     for node in path]
                avg_carbon_intensity = np.mean(carbon_intensities)
                
                candidates.append({
                    'num_hops': len(path) - 1,
                    'path_nodes': path,
                    'avg_carbon_intensity': avg_carbon_intensity
                })
            except nx.NetworkXNoPath:
                return []
            
            # 2. Try removing highest-carbon node and reroute (if path has >2 nodes)
            if len(path) > 2 and len(candidates) < k:
                for intermediate in path[1:-1]:  # Exclude src and dst
                    try:
                        # Create temp graph without this node
                        temp_graph = graph.copy()
                        temp_graph.remove_node(intermediate)
                        alt_path = nx.shortest_path(temp_graph, src, dst)
                        
                        # Avoid duplicate paths
                        if alt_path not in [c['path_nodes'] for c in candidates]:
                            carbon_intensities = [graph.nodes[node].get('carbon_intensity', 400) 
                                                for node in alt_path]
                            avg_carbon_intensity = np.mean(carbon_intensities)
                            
                            candidates.append({
                                'num_hops': len(alt_path) - 1,
                                'path_nodes': alt_path,
                                'avg_carbon_intensity': avg_carbon_intensity
                            })
                            
                            if len(candidates) >= k:
                                break
                    except (nx.NetworkXNoPath, nx.NetworkXError):
                        continue
            
            # 3. If still need more, add shortest path with weighted edges
            if len(candidates) < k:
                try:
                    # Weight by inverse of remaining capacity or random
                    weighted_path = nx.shortest_path(graph, src, dst, weight=lambda u, v, d: 1.0)
                    if weighted_path not in [c['path_nodes'] for c in candidates]:
                        carbon_intensities = [graph.nodes[node].get('carbon_intensity', 400) 
                                             for node in weighted_path]
                        avg_carbon_intensity = np.mean(carbon_intensities)
                        
                        candidates.append({
                            'num_hops': len(weighted_path) - 1,
                            'path_nodes': weighted_path,
                            'avg_carbon_intensity': avg_carbon_intensity
                        })
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    pass
            
            return candidates
        
        except Exception as e:
            print(f"Error finding routes between {src} and {dst}: {e}")
            return []

    
    def select_route(self, graph, src, dst, flow_info):
        """
        Select carbon-optimal route for a flow
        
        Args:
            graph: NetworkX network graph
            src: Source node
            dst: Destination node
            flow_info: dict with:
                - packet_count: int
                - byte_count: int
                - protocol: str ('TCP', 'UDP', 'ICMP')
                - cpu_usage: float (0-100)
        
        Returns:
            Selected route dict with 'path_nodes' and 'predicted_carbon'
        """
        # Find candidate routes
        candidates = self.find_candidate_routes(graph, src, dst, k=5)
        
        if not candidates:
            return None
        
        # Use model to select best route
        best_route, predicted_carbon = self.optimizer.select_best_route(
            candidates, flow_info
        )
        
        # Record decision
        self.routing_history.append({
            'src': src,
            'dst': dst,
            'selected_path': best_route['path_nodes'],
            'num_hops': best_route['num_hops'],
            'predicted_carbon': predicted_carbon,
            'num_candidates': len(candidates)
        })
        
        return {
            'path_nodes': best_route['path_nodes'],
            'predicted_carbon': predicted_carbon
        }
    
    def route_traffic_matrix(self, graph, traffic_flows):
        """
        Route entire traffic matrix using carbon-aware decisions
        
        Args:
            graph: NetworkX network graph
            traffic_flows: List of (src, dst, demand_gbps, protocol) tuples
        
        Returns:
            routing_decisions: List of route assignments
            total_predicted_carbon: Sum of predicted carbon for all flows
        """
        routing_decisions = []
        total_carbon = 0
        
        for src, dst, demand, protocol in traffic_flows:
            if src == dst:
                continue
            
            # Estimate flow characteristics
            # Simple heuristic: demand → bytes/packets
            duration_sec = 60  # Assume 1-minute flow
            byte_count = demand * 1e9 / 8 * duration_sec  # Gbps to bytes
            packet_count = int(byte_count / 1500)  # Assume 1500 byte packets
            
            flow_info = {
                'packet_count': packet_count,
                'byte_count': byte_count,
                'protocol': protocol,
                'cpu_usage': 50.0  # Default assumption
            }
            
            route = self.select_route(graph, src, dst, flow_info)
            
            if route:
                routing_decisions.append({
                    'src': src,
                    'dst': dst,
                    'demand': demand,
                    'path': route['path_nodes'],
                    'carbon': route['predicted_carbon']
                })
                total_carbon += route['predicted_carbon']
        
        return routing_decisions, total_carbon
    
    def get_statistics(self):
        """Return routing statistics"""
        if not self.routing_history:
            return {}
        
        num_hops = [r['num_hops'] for r in self.routing_history]
        predicted_carbon = [r['predicted_carbon'] for r in self.routing_history]
        
        return {
            'total_routes': len(self.routing_history),
            'avg_num_hops': np.mean(num_hops),
            'total_predicted_carbon': np.sum(predicted_carbon),
            'avg_carbon_per_flow': np.mean(predicted_carbon)
        }


# Example usage for NS3 integration
if __name__ == "__main__":
    print("="*70)
    print("CARBON-AWARE ROUTING CONTROLLER TEST")
    print("="*70)
    
    # Create test network
    G = nx.Graph()
    for i in range(10):
        # Assign random carbon intensities
        G.add_node(i, carbon_intensity=np.random.uniform(200, 600))
    
    # Create edges (full mesh for testing)
    for i in range(10):
        for j in range(i+1, 10):
            if np.random.random() > 0.3:  # 70% edge probability
                G.add_edge(i, j)
    
    print(f"\nTest network:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Initialize controller
    controller = CarbonAwareRoutingController()
    
    # Test routing decision
    flow_info = {
        'packet_count': 5000,
        'byte_count': 7500000,
        'protocol': 'TCP',
        'cpu_usage': 60.0
    }
    
    route = controller.select_route(G, src=0, dst=9, flow_info=flow_info)
    
    if route:
        print(f"\nRouting decision for flow (0 → 9):")
        print(f"  Path: {' → '.join(map(str, route['path_nodes']))}")
        print(f"  Predicted carbon: {route['predicted_carbon']:.6f} gCO2eq")
    
    # Get statistics
    stats = controller.get_statistics()
    print(f"\nController statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Controller ready for NS3 integration!")
    print("="*70)
