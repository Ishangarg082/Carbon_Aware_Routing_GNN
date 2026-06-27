"""
Traffic Matrix Generation for Network Simulation

Generates realistic traffic patterns between network nodes based on
topology, time of day, and network characteristics.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple


class TrafficMatrix:
    """Generates and manages traffic flows between network nodes"""
    
    def __init__(self, num_nodes: int, topology_graph: nx.Graph):
        self.num_nodes = num_nodes
        self.graph = topology_graph
        self.base_traffic_gbps = 5.0  # Increased base traffic for more visible carbon impact
        
        # Create traffic patterns based on node importance
        self.node_importance = self._calculate_node_importance()
        
    def _calculate_node_importance(self) -> Dict[int, float]:
        """Calculate node importance based on centrality"""
        centrality = nx.betweenness_centrality(self.graph)
        # Normalize to 0-1 range
        max_centrality = max(centrality.values()) if centrality else 1.0
        return {node: centrality.get(node, 0) / max_centrality 
                for node in range(self.num_nodes)}
    
    def generate_traffic_matrix(self, timestamp: int = 0) -> List[Tuple[int, int, float]]:
        """
        Generate traffic flows for the current timestamp
        
        Returns:
            List of (source, destination, traffic_gbps) tuples
        """
        flows = []
        
        # Time-of-day variation (normalized to 24-hour cycle)
        hour = (timestamp / 3600) % 24
        time_factor = self._get_time_factor(hour)
        
        # Generate flows between node pairs
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    # Skip if no path exists
                    if not nx.has_path(self.graph, src, dst):
                        continue
                    
                    # Traffic volume depends on node importance and time
                    src_importance = self.node_importance.get(src, 0.5)
                    dst_importance = self.node_importance.get(dst, 0.5)
                    
                    # Higher importance nodes generate/receive more traffic
                    traffic = self.base_traffic_gbps * (src_importance + dst_importance) / 2
                    traffic *= time_factor
                    
                    # Add some randomness
                    traffic *= np.random.uniform(0.7, 1.3)
                    
                    if traffic > 0.01:  # Only add non-trivial flows
                        flows.append((src, dst, traffic))
        
        return flows
    
    def _get_time_factor(self, hour: float) -> float:
        """Get traffic multiplier based on time of day"""
        # Peak hours: 9-11 AM and 2-5 PM
        # Low hours: midnight to 6 AM
        if 9 <= hour <= 11 or 14 <= hour <= 17:
            return 1.5  # Peak times
        elif 0 <= hour <= 6:
            return 0.3  # Low traffic
        else:
            return 1.0  # Normal traffic
    
    def get_total_traffic_volume(self, flows: List[Tuple[int, int, float]]) -> float:
        """Calculate total traffic volume in Gbps"""
        return sum(traffic for _, _, traffic in flows)
    
    def generate_datacenter_traffic(self, dc_nodes=None) -> List[Tuple[int, int, float]]:
        """
        Generate realistic data center to client traffic patterns.
        
        Pattern:
        - 10-20% of nodes are data centers (heavy traffic sources)
        - 80-90% are clients (traffic destinations)
        - Heavy traffic: DC → clients (0.5-2.0 Gbps)
        - Light traffic: client ↔ client (0.01-0.1 Gbps)
        
        Args:
            dc_nodes: List of data center node IDs (default: first 20% of nodes)
            
        Returns:
            List of (source, destination, traffic_gbps) tuples
        """
        if dc_nodes is None:
            # First 20% of nodes are data centers
            num_dc = max(2, self.num_nodes // 5)
            dc_nodes = list(range(num_dc))
        
        client_nodes = [n for n in range(self.num_nodes) if n not in dc_nodes]
        flows = []
        
        # Heavy DC → client traffic (primary workload)
        for dc in dc_nodes:
            for client in client_nodes:
                if nx.has_path(self.graph, dc, client):
                    # Heavy traffic from data centers to clients
                    traffic = np.random.uniform(0.5, 2.0)  # Gbps
                    flows.append((dc, client, traffic))
        
        # Light client ↔ client traffic (peer-to-peer, much less)
        num_client_flows = len(client_nodes) // 2
        for _ in range(num_client_flows):
            src = np.random.choice(client_nodes)
            dst = np.random.choice(client_nodes)
            if src != dst and nx.has_path(self.graph, src, dst):
                traffic = np.random.uniform(0.01, 0.1)  # Much lighter
                flows.append((src, dst, traffic))
        
        return flows


def distribute_traffic_by_routing(
    traffic_flows: List[Tuple[int, int, float]],
    topology_graph: nx.Graph,
    link_weights: np.ndarray,
    edge_index: np.ndarray
) -> Dict[int, float]:
    """
    Distribute traffic across nodes based on routing decisions
    
    Args:
        traffic_flows: List of (src, dst, traffic_gbps) tuples
        topology_graph: Network topology
        link_weights: Current routing weights for each edge
        edge_index: Edge connectivity (2 x num_edges)
    
    Returns:
        Dictionary mapping node_id to total traffic load (Gbps)
    """
    node_traffic = {i: 0.0 for i in topology_graph.nodes()}
    
    # Create weighted graph for routing
    weighted_graph = topology_graph.copy()
    
    # Apply link weights to graph
    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        weight = float(link_weights[idx])
        
        if weighted_graph.has_edge(src, dst):
            weighted_graph[src][dst]['weight'] = weight
    
    # Route each traffic flow
    for src, dst, traffic_gbps in traffic_flows:
        try:
            # Find shortest path with current weights
            path = nx.shortest_path(weighted_graph, src, dst, weight='weight')
            
            # Add traffic load to all nodes in the path
            for node in path:
                node_traffic[node] += traffic_gbps
                
        except nx.NetworkXNoPath:
            # If no path exists, skip this flow
            continue
    
    return node_traffic


if __name__ == "__main__":
    # Test traffic matrix generation
    print("Traffic Matrix Test")
    print("=" * 50)
    
    # Create simple test topology
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)])
    
    tm = TrafficMatrix(5, G)
    
    print("\nNode Importance:")
    for node, importance in tm.node_importance.items():
        print(f"  Node {node}: {importance:.3f}")
    
    print("\nTraffic at different times:")
    for hour in [3, 10, 15, 20]:
        timestamp = hour * 3600
        flows = tm.generate_traffic_matrix(timestamp)
        total_traffic = tm.get_total_traffic_volume(flows)
        print(f"  {hour:02d}:00 - Total: {total_traffic:.2f} Gbps ({len(flows)} flows)")
    
    # Test traffic distribution
    print("\nTraffic Distribution Test:")
    flows = [(0, 4, 1.0), (1, 3, 0.5)]
    edge_index = np.array([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]])
    weights = np.ones(5)
    
    node_loads = distribute_traffic_by_routing(flows, G, weights, edge_index)
    print("  Node loads (Gbps):")
    for node, load in node_loads.items():
        print(f"    Node {node}: {load:.2f} Gbps")
    
    print("\nTraffic matrix test passed.")
