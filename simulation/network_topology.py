import numpy as np
import networkx as nx

class NetworkTopology:
    def __init__(self, num_nodes, topology_type='hierarchical'):
        self.num_nodes = num_nodes
        self.topology_type = topology_type
        self.graph = None
        self.positions = {}
        self.node_properties = {}
        self.edge_properties = {}
        
    def generate(self):
        if self.topology_type == 'grid':
            self._create_grid()
        elif self.topology_type == 'hierarchical':
            self._create_hierarchical()
        elif self.topology_type == 'scale_free':
            self._create_scale_free()
        elif self.topology_type == 'small_world':
            self._create_small_world()
        else:
            self._create_random()
        
        self._assign_link_properties()
        return self.graph
    
    def _create_grid(self):
        side = int(np.ceil(np.sqrt(self.num_nodes)))
        self.graph = nx.grid_2d_graph(side, side)
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        self.graph = self.graph.subgraph(range(self.num_nodes)).copy()
        
        for i, node in enumerate(self.graph.nodes()):
            x = i % side
            y = i // side
            self.positions[node] = (x / side, y / side)
    
    def _create_hierarchical(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))
        
        core_nodes = max(2, self.num_nodes // 10)
        agg_nodes = max(3, self.num_nodes // 4)
        
        # Core layer: fully meshed
        for i in range(core_nodes):
            for j in range(i + 1, core_nodes):
                self.graph.add_edge(i, j)
            self.positions[i] = (i / core_nodes, 0.9)
        
        # Aggregation layer: connect to core + cross-links between agg nodes
        for i in range(core_nodes, agg_nodes):
            # Primary uplink to core
            core = i % core_nodes
            self.graph.add_edge(i, core)
            # Secondary uplink to another core (redundancy)
            core2 = (i + 1) % core_nodes
            if core2 != core:
                self.graph.add_edge(i, core2)
            self.positions[i] = ((i - core_nodes) / max(1, agg_nodes - core_nodes), 0.5)
        
        # Cross-links between adjacent aggregation nodes
        for i in range(core_nodes, agg_nodes - 1):
            self.graph.add_edge(i, i + 1)
        
        # Access layer: connect to aggregation + some cross-links
        for i in range(agg_nodes, self.num_nodes):
            # Primary downlink to aggregation
            agg = core_nodes + (i % (agg_nodes - core_nodes))
            self.graph.add_edge(i, agg)
            # Secondary downlink to another aggregation node (multi-homing)
            agg2 = core_nodes + ((i + 1) % (agg_nodes - core_nodes))
            if agg2 != agg:
                self.graph.add_edge(i, agg2)
            self.positions[i] = ((i - agg_nodes) / max(1, self.num_nodes - agg_nodes), 0.1)
        
        # Add a few cross-links between access nodes for extra redundancy
        access_nodes = list(range(agg_nodes, self.num_nodes))
        # NOTE: seed is inherited from the caller (np.random.seed set before create_network())
        num_cross = max(2, len(access_nodes) // 3)
        for _ in range(num_cross):
            a, b = np.random.choice(access_nodes, 2, replace=False)
            if not self.graph.has_edge(a, b):
                self.graph.add_edge(a, b)
    
    def _create_scale_free(self):
        rng_seed = int(np.random.randint(0, 2**31))
        # m = edges added per new node (Barabasi-Albert).
        # Cap m at 4 to keep edge count O(4N) — consistent with small_world k=8
        # (average degree = 2m, so m=4 -> avg degree 8).
        m = np.random.randint(2, 5)  # 2, 3, or 4
        self.graph = nx.barabasi_albert_graph(self.num_nodes, m=m, seed=rng_seed)
        self.positions = nx.spring_layout(self.graph, seed=rng_seed)
    
    def _create_small_world(self):
        rng_seed = int(np.random.randint(0, 2**31))
        # k = neighbour count in the ring lattice before rewiring.
        # Cap at 8 to bound edge count: N=200, k=8 -> 800 edges max.
        # Without the cap: k=max(4, randint(4, N//4)) -> k up to 50 for N=200
        # -> 5000 edges -> PopulateRoutingTables runs Dijkstra for minutes.
        k = min(8, max(4, np.random.randint(4, max(5, self.num_nodes // 10))))
        p_rewire = np.random.uniform(0.1, 0.5)   # rewiring probability varies per seed
        self.graph = nx.watts_strogatz_graph(self.num_nodes, k, p=p_rewire, seed=rng_seed)
        self.positions = nx.circular_layout(self.graph)
    
    def _create_random(self):
        rng_seed = int(np.random.randint(0, 2**31))
        # Derive p from a target AVERAGE DEGREE of 8 (matching small_world k=8 cap).
        # Old formula: p = uniform(0.05, 0.15) + log(N)/N
        #   -> for N=350: p ≈ 0.05 + 0.017 = 0.067 -> 3065 edges -> hangs.
        # New formula: p = target_degree / (N-1) with small variation
        #   -> for N=350: p = 8/349 ≈ 0.023 -> ~1400 edges -> fast.
        target_degree = np.random.uniform(6.0, 8.0)   # average degree 6-8
        p = target_degree / max(self.num_nodes - 1, 1)
        p *= np.random.uniform(0.9, 1.1)              # +/-10% variation
        p = min(p, 0.08)                              # hard cap just in case
        p = max(p, 2.0 / max(self.num_nodes - 1, 1)) # at least 2 edges/node avg
        self.graph = nx.erdos_renyi_graph(self.num_nodes, p, seed=rng_seed)
        attempts = 0
        while not nx.is_connected(self.graph) and attempts < 20:
            p = min(p * 1.3, 0.15)  # bump up gently; hard cap at 0.15
            rng_seed += 1
            self.graph = nx.erdos_renyi_graph(self.num_nodes, p, seed=rng_seed)
            attempts += 1
        self.positions = nx.spring_layout(self.graph, seed=rng_seed)
    
    def _assign_link_properties(self):
        # base_bw and delay_scale are set by caller to vary per seed
        base_bw   = getattr(self, 'base_bw',    1000)
        delay_scale = getattr(self, 'delay_scale', 100)
        for u, v in self.graph.edges():
            if u in self.positions and v in self.positions:
                pos_u = np.array(self.positions[u])
                pos_v = np.array(self.positions[v])
                distance = np.linalg.norm(pos_u - pos_v)
            else:
                distance = np.random.uniform(0.05, 0.3)

            bandwidth = int(base_bw * np.random.uniform(0.5, 2.0))
            delay = max(1, int(distance * delay_scale))

            self.edge_properties[(u, v)] = {
                'bandwidth': bandwidth,
                'delay': delay,
                'distance': distance
            }
    
    def get_adjacency_matrix(self):
        return nx.to_numpy_array(self.graph)
    
    def get_edge_list(self):
        edges = []
        for u, v in self.graph.edges():
            edges.append([u, v])
            edges.append([v, u])
        return np.array(edges).T
    
    def get_node_features(self, default_features=None):
        if default_features is None:
            default_features = {}
        
        features = []
        for node in range(self.num_nodes):
            if node in default_features:
                features.append(default_features[node])
            else:
                degree = self.graph.degree(node) if self.graph.has_node(node) else 0
                features.append([
                    degree / self.num_nodes,
                    np.random.random(),
                    np.random.random(),
                    350.0
                ])
        return np.array(features)
    
    def get_edge_features(self):
        edge_list = self.get_edge_list()
        features = []
        
        for i in range(edge_list.shape[1]):
            u, v = edge_list[0, i], edge_list[1, i]
            
            edge_key = (u, v) if (u, v) in self.edge_properties else (v, u)
            if edge_key in self.edge_properties:
                props = self.edge_properties[edge_key]
                features.append([
                    props['bandwidth'],
                    props['delay'],
                    1
                ])
            else:
                features.append([1000, 10, 1])
        
        return np.array(features)


def create_network(num_nodes=20, topology='hierarchical', base_bw=1000, delay_scale=100):
    topo = NetworkTopology(num_nodes, topology)
    # inject per-seed link parameters
    topo.base_bw = base_bw
    topo.delay_scale = delay_scale
    graph = topo.generate()

    return {
        'graph': graph,
        'positions': topo.positions,
        'edge_properties': topo.edge_properties,
        'node_features': topo.get_node_features(),
        'edge_index': topo.get_edge_list(),
        'edge_features': topo.get_edge_features()
    }


if __name__ == "__main__":
    print("Network Topology Generator Test")
    print("=" * 50)
    
    for topo_type in ['grid', 'hierarchical', 'scale_free']:
        net = create_network(num_nodes=15, topology=topo_type)
        
        print(f"\n{topo_type.capitalize()} Topology:")
        print(f"  Nodes: {net['graph'].number_of_nodes()}")
        print(f"  Edges: {net['graph'].number_of_edges()}")
        print(f"  Connected: {nx.is_connected(net['graph'])}")
        print(f"  Avg degree: {sum(dict(net['graph'].degree()).values()) / net['graph'].number_of_nodes():.2f}")
    
    print("\nTopology generator test passed.")
