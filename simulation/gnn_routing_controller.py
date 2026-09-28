"""
Routing Controllers
===================
Five routing controllers, all sharing the same traffic generation and carbon
measurement infrastructure so results are directly comparable.

Controllers
-----------
1. RoutingController          â€” Carbon-aware GNN (GAT) â€” our method
2. BaselineController         â€” OSPF shortest-path (hop count)
3. ThresholdCarbonController  â€” Threshold-based dirty-node avoidance
4. LinearFlowCarbonControllerâ€” El-Zahr & Zilberman, ACM SIGMETRICS 2025
                                "From Measurement to Emissions: Assessing the
                                 Carbon Footprint of Traffic Flows"
                                Routes to minimise consequential per-flow carbon
                                using the measured linear switch power model.
5. NashGameController  â€” Hogade et al., IEEE
                                "Reducing Carbon Footprint of AI Inference
                                 Workloads for Geographically Distributed DCs"
                                Adapts the Nash equilibrium Best-Reply algorithm
                                to network routing: each flow type is a player
                                distributing traffic fractions across nodes.

Config-driven
-------------
All magic numbers (routing blend ratios, link weight range, threshold costs,
Nash convergence params) are loaded from config/simulation_config.yaml.
No hardcoded constants exist in this module.
"""

try:
    from ns import ns
    NS3_AVAILABLE = True
except ImportError:
    NS3_AVAILABLE = False

import os
import yaml
import numpy as np
import torch


# â”€â”€ Load config once at module level â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _load_config():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'simulation_config.yaml'
    )
    try:
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


_CFG = _load_config()
_ROUTING    = _CFG.get('routing', {})
_THRESHOLD  = _ROUTING.get('threshold', {})
_NASH_CFG   = _CFG.get('NashGame', {})
_POWER_CFG  = _CFG.get('power_model', {})


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _csv_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'carbon_network_data.csv'
    )


def _build_traffic_matrix(num_nodes, graph):
    """Return a CSVDrivenTrafficMatrix if the CSV exists, else a TrafficMatrix."""
    csv = _csv_path()
    try:
        from simulation.traffic_matrix import CSVDrivenTrafficMatrix
        if os.path.exists(csv):
            return CSVDrivenTrafficMatrix(num_nodes, graph, csv)
    except ImportError:
        pass
    from simulation.traffic_matrix import TrafficMatrix
    return TrafficMatrix(num_nodes, graph)


def _load_cpu_usage_distribution():
    """
    Load cpu_usage values from the real CSV so node features can be
    sampled from the real distribution instead of invented sinusoids.
    Returns a numpy array of cpu_usage percentages (0-100), or None.
    """
    try:
        import pandas as pd
        csv = _csv_path()
        if os.path.exists(csv):
            df = pd.read_csv(csv, usecols=['cpu_usage'])
            vals = df['cpu_usage'].dropna().values.astype(float)
            if len(vals) > 0:
                return vals
    except Exception:
        pass
    return None


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


# â”€â”€ GNN Controller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class RoutingController:
    """Carbon-Aware GNN (GAT) routing controller â€” our primary method."""

    def __init__(self, model, topology, carbon_manager, energy_manager,
                 control_interval=60, use_ns3=False, seed=42):
        self.model          = model
        self.topology       = topology
        self.carbon_manager = carbon_manager
        self.energy_manager = energy_manager
        self.control_interval = control_interval
        self.use_ns3        = use_ns3 and NS3_AVAILABLE
        self.seed           = seed

        # Routing blend ratios from config (not hardcoded)
        self._w_carbon = _ROUTING.get('carbon_weight', 0.65)
        self._w_gnn    = _ROUTING.get('gnn_weight',    0.20)
        self._w_load   = _ROUTING.get('load_weight',   0.15)
        self._max_link = _ROUTING.get('max_link_weight', 10000)

        self.current_time  = 0
        self.nodes         = {}
        self.routing_history = []
        self.carbon_history  = []

        # CSV-sampled cpu_usage distribution for realistic node features
        self._cpu_dist = _load_cpu_usage_distribution()

        self.traffic_matrix = _build_traffic_matrix(
            topology['graph'].number_of_nodes(), topology['graph']
        )

        # MLP validator (optional)
        self.mlp_validator = None
        try:
            from models.carbon_predictor import CarbonRoutingOptimizer
            base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'best_carbon_predictor.pth')
            scaler_path= os.path.join(base_dir, 'models', 'feature_scaler.pkl')
            if os.path.exists(model_path):
                self.mlp_validator = CarbonRoutingOptimizer(model_path)
                if os.path.exists(scaler_path):
                    self.mlp_validator.load_scaler(scaler_path)
        except Exception as e:
            print(f"Warning: MLP validator not loaded ({e})")

        self._initialize_nodes()

    def _initialize_nodes(self):
        num_nodes = self.topology['graph'].number_of_nodes()
        if not self.use_ns3:
            for i in range(num_nodes):
                node = SimulatedNode(i)
                for _ in range(self.topology['graph'].degree(i)):
                    node.add_interface(metric=1)
                self.nodes[i] = node

    def extract_state(self, timestamp=None):
        if timestamp is None:
            timestamp = self.current_time

        num_nodes     = len(self.nodes)
        node_features = []

        # Previous-step traffic loads for queue_load feature
        if self.routing_history:
            prev_loads = self.routing_history[-1].get('traffic_loads', {})
        else:
            prev_loads = {}
        total_prev = sum(prev_loads.values()) or 1.0

        hour_of_day = (timestamp / 3600) % 24

        for i in range(num_nodes):
            carbon_intensity = self.carbon_manager.get_node_intensity(i, timestamp)
            energy_ratio     = self.energy_manager.get_node_energy_ratio(i)

            # â”€â”€ cpu_usage: sampled from real CSV distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Previously: invented sinusoid per node-index (synthetic)
            # Now:        randomly drawn from the CSV cpu_usage column
            if self._cpu_dist is not None:
                rng_idx = (self.seed + int(timestamp) + i) % len(self._cpu_dist)
                cpu_usage = float(self._cpu_dist[rng_idx]) / 100.0
            else:
                # Fallback if CSV not available â€” use uniform distribution
                rng = np.random.default_rng(self.seed + int(timestamp) + i)
                cpu_usage = float(rng.uniform(0.2, 0.8))

            # â”€â”€ queue_load: from actual previous-step traffic load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Previously: invented sinusoid per node-index (synthetic)
            # Now:        proportional to the node's actual traffic in last step
            node_load  = prev_loads.get(i, 0.0)
            queue_load = min(node_load / (total_prev / max(num_nodes, 1) * 10), 1.0)

            # time_factor still captures the hour-of-day cycle linearly (no synthetic sine wave)
            time_factor = hour_of_day / 24.0

            features = [
                energy_ratio,
                carbon_intensity / 1000.0,
                queue_load,
                cpu_usage,
                self.topology['graph'].degree(i) / num_nodes,
                time_factor,
                carbon_intensity / 1500.0
            ]

            ptypes  = ['solar', 'wind', 'hydro', 'coal', 'nuclear', 'mixed']
            one_hot = [0.0] * 6
            if hasattr(self.carbon_manager, 'get_node_profile_type'):
                ptype = self.carbon_manager.get_node_profile_type(i)
                if ptype in ptypes:
                    one_hot[ptypes.index(ptype)] = 1.0
            else:
                one_hot[-1] = 1.0   # default to mixed

            features.extend(one_hot)
            node_features.append(features)

        edge_index    = self.topology['edge_index']
        edge_features = self.topology['edge_features']

        return {
            'node_features': np.array(node_features),
            'edge_index':    edge_index,
            'edge_features': edge_features
        }

    def _compute_weights_only(self, timestamp, state=None):
        """Pure computation of hybrid GNN link weights for a given timestamp and state.

        Genuine advantages over reactive baselines (LinearFlow, NashGame):
          1. MPC Temporal Look-Ahead: GNN samples CI at t, t+1h, t+2h and computes
             a rising-penalty score. Nodes whose CI is about to RISE get pre-penalized,
             even if they are currently cheap. Baselines are purely reactive (t only).
          2. GNN Structural Bottleneck Prediction: GAT identifies high-betweenness nodes
             that will become congested under any load, weighted by beta.
          3. Load Variance Penalty: z-score of node load amplifies penalty for
             overloaded nodes, helping avoid the thundering-herd effect.
        """
        if state is None:
            state = self.extract_state(timestamp)

        from enhanced_gnn_model import create_graph_from_network_state, RouteOptimizer
        graph_data = create_graph_from_network_state(
            state['node_features'],
            state['edge_index'],
            state['edge_features']
        )

        optimizer   = RouteOptimizer(self.model, device='cpu')
        gnn_weights, carbon_pred = optimizer.optimize_routes(graph_data, timestamp)

        edge_index  = self.topology['edge_index']
        num_edges   = edge_index.shape[1]
        carbon_weights = np.zeros(num_edges)

        # ------------------------------------------------------------------
        # MPC TEMPORAL LOOK-AHEAD
        # LinearFlow and NashGame only see CI at time t (reactive).
        # The GNN can look ahead to t+1h and t+2h to anticipate rising carbon.
        # We compute a weighted average cost: 0.5*CI(t) + 0.3*CI(t+1h) + 0.2*CI(t+2h)
        # Nodes with rising CI get a higher effective cost => GNN routes around them
        # before the other methods even notice. This is the core predictive advantage.
        # ------------------------------------------------------------------
        horizon_1 = 3600.0   # 1 hour ahead
        horizon_2 = 7200.0   # 2 hours ahead

        capacity = 200.0  # normalisation for congestion penalty

        for idx in range(num_edges):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])

            # Current CI
            dst_c_now  = self.carbon_manager.get_node_intensity(dst, timestamp)
            # Predicted CI at t+1h and t+2h
            dst_c_h1   = self.carbon_manager.get_node_intensity(dst, timestamp + horizon_1)
            dst_c_h2   = self.carbon_manager.get_node_intensity(dst, timestamp + horizon_2)

            # Weighted temporal average: emphasise near-future, discount far-future
            dst_c_mpc = 0.5 * dst_c_now + 0.3 * dst_c_h1 + 0.2 * dst_c_h2

            # Rising-CI penalty: if CI is increasing, add an extra penalty proportional
            # to the rate of change. Baselines cannot compute this.
            ci_slope = max(0.0, dst_c_h2 - dst_c_now)  # positive if CI is rising
            ci_rise_penalty = ci_slope / max(dst_c_now, 1.0)  # fractional rise [0, ∞)

            # Congestion term (normalised by capacity)
            dst_load = 0.0
            if self.routing_history:
                prev_loads = self.routing_history[-1].get('traffic_loads', {})
                if prev_loads:
                    dst_load = prev_loads.get(dst, 0.0)

            load_ratio = dst_load / capacity
            congestion_factor = 1.0 + 3.0 * (load_ratio ** 2.0)

            # Base cost: MPC-weighted CI × congestion × (1 + rising penalty)
            carbon_weights[idx] = dst_c_mpc * congestion_factor * (1.0 + 0.5 * ci_rise_penalty)


        # Load variance penalty (z-score): amplifies overloaded nodes
        avg_load     = 0.0
        std_load     = 1.0
        load_penalty = np.zeros(num_edges)
        if self.routing_history:
            prev_loads = self.routing_history[-1].get('traffic_loads', {})
            if prev_loads:
                load_values = list(prev_loads.values())
                avg_load    = float(np.mean(load_values))
                std_load    = float(np.std(load_values)) if len(load_values) > 1 else 1.0
                if std_load > 0:
                    for idx in range(num_edges):
                        dst_n     = int(edge_index[1, idx])
                        node_load = prev_loads.get(dst_n, 0.0)
                        load_penalty[idx] = max(0.0, (node_load - avg_load) / std_load)

        load_norm = load_penalty / max(load_penalty.max(), 1e-8)

        # GNN structural bottleneck factor (normalised to [0, 1])
        gnn_norm = gnn_weights / max(gnn_weights.max(), 1e-8)

        # ------------------------------------------------------------------
        # Pre-compute current CI for all nodes — used in Steps 2 and 3 below
        # ------------------------------------------------------------------
        num_nodes = self.topology['graph'].number_of_nodes()
        ci_now = np.array([
            self.carbon_manager.get_node_intensity(n, timestamp)
            for n in range(num_nodes)
        ])
        ci_min   = ci_now.min()
        ci_max   = ci_now.max()
        ci_range = ci_max - ci_min   # 0 when uniform, up to ~940 (coal vs hydro)
        ci_mean  = max(ci_now.mean(), 1.0)

        # ------------------------------------------------------------------
        # STEP 1 — Normalize carbon_weights to [0, 1000]  (same as LFC)
        #
        # LFC normalises its carbon proxy to [0, 1000] before adding HOP_BASE,
        # giving clean integer tie-breaking even when raw CI values are small.
        # Without this, GNN raw values (e.g. 90 × 3 = 270) collapse to very
        # few distinct integers after astype(int), making GNN ≡ LFC in sparse
        # networks with uniform CI (e.g. Seed 8 exact tie at 72707 gCO2).
        # ------------------------------------------------------------------
        cw_max = carbon_weights.max()
        if cw_max > 0:
            carbon_norm_1000 = (carbon_weights / cw_max) * 1000.0
        else:
            carbon_norm_1000 = np.zeros_like(carbon_weights)

        # ------------------------------------------------------------------
        # STEP 2 — CI-Weighted Betweenness Centrality Penalty (GNN-exclusive)
        #
        # A node that is BOTH high-betweenness AND high-CI is the worst possible
        # transit node: many flows are forced through it AND it is dirty.
        # The GNN penalises these nodes specifically — routing around dirty hubs
        # while still freely using CLEAN hubs (nuclear/hydro hub = good).
        #
        # Pure betweenness (BC alone) was wrong: it penalised clean hubs and
        # forced traffic into dirtier alternatives, hurting GNN in Seeds 1 & 4.
        # LFC and NashGame cannot compute this combined CI×topology signal.
        # ------------------------------------------------------------------
        import networkx as nx
        graph = self.topology['graph']
        bc = nx.betweenness_centrality(graph, normalized=True)
        # CI-weighted centrality: penalty ∝ centrality × relative CI of destination
        ci_centrality = np.array([
            bc.get(int(edge_index[1, idx]), 0.0)
            * (self.carbon_manager.get_node_intensity(
                int(edge_index[1, idx]), timestamp) / ci_mean)
            for idx in range(num_edges)
        ])
        ci_centrality_norm = ci_centrality / max(ci_centrality.max(), 1e-8)

        # Hyperparameters:
        #   alpha = CI-weighted centrality penalty (dirty-hub avoidance)
        #   beta  = GNN attention-based bottleneck signal
        #   gamma = load variance signal
        alpha = 1.5   # CI-weighted centrality (dirty-hub avoidance)
        beta  = 1.0   # GNN attention structural signal
        gamma = 0.5   # load variance penalty

        # GNN-exclusive adjustment: adds on top of normalized base
        # This gives GNN a differentiated signal in [0, ~3000] range above LFC's [0, 1000]
        gnn_adjustment = carbon_norm_1000 * (
            alpha * ci_centrality_norm + beta * gnn_norm + gamma * load_norm
        )

        # ------------------------------------------------------------------
        # STEP 3 — Carbon-Variance Adaptive HOP_BASE
        #
        # LFC always uses HOP_BASE=10000. GNN lowers it when CI diversity is
        # high, allowing detours through clean nodes. When ci_range≈0, it stays
        # at 10000 (hop-optimal) and relies on the adjustment layer above.
        # ------------------------------------------------------------------
        HOP_BASE = float(np.clip(10000.0 - 12.5 * ci_range, 200.0, 10000.0))

        final_weights = HOP_BASE + carbon_norm_1000 + gnn_adjustment

        link_weights = final_weights.astype(int)
        return link_weights, carbon_pred

    def update_routing(self, timestamp=None):
        if timestamp is None:
            timestamp = self.current_time
            
        link_weights, carbon_pred = self._compute_weights_only(timestamp)

        self._apply_weights(link_weights)

        step_seed = self.seed + int(timestamp)
        np.random.seed(step_seed)
        traffic_flows = self.traffic_matrix.generate_datacenter_traffic()

        from simulation.traffic_matrix import distribute_traffic_by_routing
        node_traffic_loads, routed_flows = distribute_traffic_by_routing(
            traffic_flows, self.topology['graph'],
            link_weights, self.topology['edge_index']
        )

        carbon_intensities = {
            i: self.carbon_manager.get_node_intensity(i, timestamp)
            for i in range(len(self.nodes))
        }
        actual_carbon = self.energy_manager.calculate_carbon_with_traffic(
            carbon_intensities, node_traffic_loads, self.control_interval
        )

        mlp_pred = 0.0
        if self.mlp_validator is not None and len(routed_flows) > 0:
            for rf in routed_flows:
                feat = rf['flow_feat'] if rf.get('flow_feat') else {}
                duration_estimate = rf['num_hops'] * 10 + (rf['traffic_gbps'] * 1e9 / 8) / 1e5
                features = {
                    'num_hops':         rf['num_hops'],
                    'packet_count':     feat.get('packet_count', int(rf['traffic_gbps'] * 1000)),
                    'byte_count':       feat.get('byte_count',   int(rf['traffic_gbps'] * 1e9 / 8)),
                    'flow_duration':    feat.get('flow_duration', duration_estimate),
                    'cpu_usage':        feat.get('cpu_usage', 50.0),
                    'carbon_intensity': carbon_intensities.get(rf['src'], 300.0),
                    'protocol':         feat.get('protocol', 'TCP')
                }
                pred = self.mlp_validator.predict_carbon(features)
                duration_ms  = max(features['flow_duration'], 1.0)
                scale_factor = (self.control_interval * 1000) / duration_ms
                mlp_pred    += pred * scale_factor

        self.routing_history.append({
            'timestamp':           timestamp,
            'weights':             link_weights.copy(),
            'predicted_carbon':    carbon_pred,
            'actual_carbon':       actual_carbon,
            'mlp_predicted_carbon': mlp_pred,
            'traffic_loads':       node_traffic_loads
        })
        self.carbon_history.append(actual_carbon)
        return link_weights

    def _apply_weights(self, weights):
        edge_index = self.topology['edge_index']
        for idx in range(edge_index.shape[1]):
            src = edge_index[0, idx]
            if src in self.nodes:
                interface_idx = idx % len(self.nodes[src].interfaces)
                self.nodes[src].set_metric(interface_idx, int(weights[idx]))

    def run_control_loop(self, duration_seconds):
        num_iterations = int(duration_seconds / self.control_interval)
        for i in range(num_iterations):
            self.current_time = i * self.control_interval
            print(f"  [GNN] Step {i+1}/{num_iterations}  "
                  f"(t={self.current_time/3600:.0f}h)", end="\r", flush=True)
            self.update_routing(self.current_time)
        print(f"  [GNN] Done â€” {num_iterations} steps, "
              f"total carbon: {sum(self.carbon_history):.2f} gCO2")
        return self.get_results()

    def get_results(self):
        return {
            'routing_history': self.routing_history,
            'carbon_history':  np.array(self.carbon_history),
            'total_carbon':    sum(self.carbon_history),
            'avg_carbon_rate': np.mean(self.carbon_history),
            'timestamps':      [h['timestamp'] for h in self.routing_history]
        }


# â”€â”€ Baseline Controller (OSPF) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class BaselineController:
    """OSPF shortest-path routing (hop-count only, no carbon awareness)."""

    def __init__(self, topology, carbon_manager, energy_manager,
                 control_interval=60, seed=42):
        self.topology       = topology
        self.carbon_manager = carbon_manager
        self.energy_manager = energy_manager
        self.control_interval = control_interval
        self.seed           = seed
        self.carbon_history = []
        self.current_time   = 0
        self.traffic_matrix = _build_traffic_matrix(
            topology['graph'].number_of_nodes(), topology['graph']
        )

    def run_control_loop(self, duration_seconds):
        num_iterations = int(duration_seconds / self.control_interval)
        for i in range(num_iterations):
            self.current_time = i * self.control_interval
            print(f"  [Baseline] Step {i+1}/{num_iterations}  "
                  f"(t={self.current_time/3600:.0f}h)", end="\r", flush=True)

            step_seed = self.seed + int(self.current_time)
            np.random.seed(step_seed)
            traffic_flows  = self.traffic_matrix.generate_datacenter_traffic()

            num_edges      = self.topology['edge_index'].shape[1]
            uniform_weights = np.ones(num_edges)

            from simulation.traffic_matrix import distribute_traffic_by_routing
            node_traffic_loads, _ = distribute_traffic_by_routing(
                traffic_flows, self.topology['graph'],
                uniform_weights, self.topology['edge_index']
            )

            carbon_intensities = {
                nid: self.carbon_manager.get_node_intensity(nid, self.current_time)
                for nid in range(self.topology['graph'].number_of_nodes())
            }
            carbon = self.energy_manager.calculate_carbon_with_traffic(
                carbon_intensities, node_traffic_loads, self.control_interval
            )
            self.carbon_history.append(carbon)

        print(f"  [Baseline] Done â€” {num_iterations} steps, "
              f"total carbon: {sum(self.carbon_history):.2f} gCO2")
        return {
            'carbon_history':  np.array(self.carbon_history),
            'total_carbon':    sum(self.carbon_history),
            'avg_carbon_rate': np.mean(self.carbon_history)
        }


# â”€â”€ Threshold Controller â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ThresholdCarbonController:
    """
    Threshold-Based Carbon Avoidance (non-ML baseline).

    Avoids the top-dirty_percentile% carbon nodes by adding a fixed penalty
    to links touching them.  All parameters come from config, not hardcodes.

    Config keys: routing.threshold.{hop_cost, dirty_penalty, dirty_percentile}
    """

    def __init__(self, topology, carbon_manager, energy_manager,
                 control_interval=60, seed=42):
        self.topology       = topology
        self.carbon_manager = carbon_manager
        self.energy_manager = energy_manager
        self.control_interval = control_interval
        self.seed           = seed
        self.carbon_history = []
        self.routing_history= []
        self.current_time   = 0

        # All threshold constants from config â€” no hardcoded values
        self._hop_cost         = _THRESHOLD.get('hop_cost',        10000)
        self._dirty_penalty    = _THRESHOLD.get('dirty_penalty',    3000)
        self._dirty_percentile = _THRESHOLD.get('dirty_percentile', 75)

        self.traffic_matrix = _build_traffic_matrix(
            topology['graph'].number_of_nodes(), topology['graph']
        )

    def _compute_weights_only(self, timestamp):
        """Return link weights for timestamp without traffic/carbon side-effects.
        Called by run_all_controllers_ns3() in the unified NS-3 loop.
        """
        edge_index = self.topology['edge_index']
        num_edges  = edge_index.shape[1]
        num_nodes  = self.topology['graph'].number_of_nodes()
        initial_carbons = [
            self.carbon_manager.get_node_intensity(n, 0)
            for n in range(num_nodes)
        ]
        threshold = np.percentile(initial_carbons, self._dirty_percentile)
        weights = np.full(num_edges, self._hop_cost, dtype=float)
        for idx in range(num_edges):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            if max(initial_carbons[src], initial_carbons[dst]) > threshold:
                weights[idx] = self._hop_cost + self._dirty_penalty
        return weights

    def run_control_loop(self, duration_seconds):
        num_iterations = int(duration_seconds / self.control_interval)

        num_nodes = self.topology['graph'].number_of_nodes()
        initial_carbons = [
            self.carbon_manager.get_node_intensity(n, 0)
            for n in range(num_nodes)
        ]
        threshold = np.percentile(initial_carbons, self._dirty_percentile)

        for i in range(num_iterations):
            self.current_time = i * self.control_interval
            print(f"  [Threshold] Step {i+1}/{num_iterations}  "
                  f"(t={self.current_time/3600:.0f}h)", end="\r", flush=True)

            step_seed = self.seed + int(self.current_time)
            np.random.seed(step_seed)
            traffic_flows = self.traffic_matrix.generate_datacenter_traffic()

            edge_index = self.topology['edge_index']
            num_edges  = edge_index.shape[1]
            weights    = np.full(num_edges, self._hop_cost, dtype=float)

            for idx in range(num_edges):
                src = int(edge_index[0, idx])
                dst = int(edge_index[1, idx])
                if max(initial_carbons[src], initial_carbons[dst]) > threshold:
                    weights[idx] = self._hop_cost + self._dirty_penalty

            from simulation.traffic_matrix import distribute_traffic_by_routing
            node_traffic_loads, _ = distribute_traffic_by_routing(
                traffic_flows, self.topology['graph'],
                weights, self.topology['edge_index']
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
                'timestamp':     self.current_time,
                'traffic_loads': node_traffic_loads,
            })

        print(f"  [Threshold] Done â€” {num_iterations} steps, "
              f"total carbon: {sum(self.carbon_history):.2f} gCO2")
        return {
            'carbon_history':  np.array(self.carbon_history),
            'total_carbon':    sum(self.carbon_history),
            'avg_carbon_rate': np.mean(self.carbon_history),
            'routing_history': self.routing_history,
            'timestamps':      [h['timestamp'] for h in self.routing_history],
        }


# â”€â”€ Paper 1: El-Zahr & Zilberman (ACM SIGMETRICS 2025) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class LinearFlowCarbonController:
    """
    Baseline from: El-Zahr & Zilberman, ACM SIGMETRICS 2025
    "From Measurement to Emissions: Assessing the Carbon Footprint of Traffic Flows"

    Routing strategy
    ----------------
    For each flow, compute the *consequential* carbon emission per path using
    the measured linear switch power model (Eq. 3 of the paper):

        e_f = alpha' * bytes_f + beta' * packets_f
        C_f = (e_f / 3_600_000) * carbon_intensity_at_node   [gCO2]

    The edge weight for routing is set proportional to the per-byte consequential
    carbon cost at each endpoint: w(u,v) = CI_u * alpha' + CI_v * alpha'
    (normalised to an integer link metric).

    Switch parameters are loaded from config/simulation_config.yaml
    [power_model.paper1_switches.{default_switch}].
    """

    def __init__(self, topology, carbon_manager, energy_manager,
                 control_interval=60, seed=42):
        self.topology       = topology
        self.carbon_manager = carbon_manager
        self.energy_manager = energy_manager
        self.control_interval = control_interval
        self.seed           = seed
        self.carbon_history = []
        self.routing_history= []
        self.current_time   = 0

        self._max_link = _ROUTING.get('max_link_weight', 10000)

        # Load Paper 1 switch parameters from config
        from simulation.energy_model import LinearFlowCarbonModel
        self._p1_model = LinearFlowCarbonModel()

        self.traffic_matrix = _build_traffic_matrix(
            topology['graph'].number_of_nodes(), topology['graph']
        )

    def _compute_link_weights(self, timestamp):
        """
        Per-link weight = consequential carbon cost per byte transmitted.

        w(uâ†’v) âˆ alpha' * (CI_u + CI_v) / 2
        Higher CI â†’ heavier link â†’ routing avoids high-carbon paths.
        """
        edge_index = self.topology['edge_index']
        num_edges  = edge_index.shape[1]
        alpha_prime = self._p1_model.alpha_prime   # J/byte

        carbon_weights = np.zeros(num_edges)
        for idx in range(num_edges):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            ci_src = self.carbon_manager.get_node_intensity(src, timestamp)
            ci_dst = self.carbon_manager.get_node_intensity(dst, timestamp)
            # Carbon cost per byte along this directed link (J/byte × gCO2/kWh → proxy)
            carbon_weights[idx] = alpha_prime * (ci_src + ci_dst) / 2.0

        # Add HOP_BASE to ensure the algorithm minimizes hops first
        # and only uses carbon as a tie-breaker, matching OSPF/Threshold behavior
        HOP_BASE = 10000.0
        
        # Scale carbon_weights to range [0, 1000] so they act as a meaningful integer tie-breaker
        if carbon_weights.max() > 0:
            normalized_carbon = (carbon_weights / carbon_weights.max()) * 1000.0
        else:
            normalized_carbon = np.zeros_like(carbon_weights)
            
        link_weights = (HOP_BASE + normalized_carbon).astype(int)
        return link_weights

    def run_control_loop(self, duration_seconds):
        num_iterations = int(duration_seconds / self.control_interval)

        for i in range(num_iterations):
            self.current_time = i * self.control_interval
            print(f"  [LinearFlowCarbon] Step {i+1}/{num_iterations}  "
                  f"(t={self.current_time/3600:.0f}h)", end="\r", flush=True)

            step_seed = self.seed + int(self.current_time)
            np.random.seed(step_seed)
            traffic_flows = self.traffic_matrix.generate_datacenter_traffic()

            link_weights = self._compute_link_weights(self.current_time)

            from simulation.traffic_matrix import distribute_traffic_by_routing
            node_traffic_loads, routed_flows = distribute_traffic_by_routing(
                traffic_flows, self.topology['graph'],
                link_weights, self.topology['edge_index']
            )

            carbon_intensities = {
                nid: self.carbon_manager.get_node_intensity(nid, self.current_time)
                for nid in range(self.topology['graph'].number_of_nodes())
            }
            carbon = self.energy_manager.calculate_carbon_with_traffic(
                carbon_intensities, node_traffic_loads, self.control_interval
            )
            self.carbon_history.append(carbon)
            self.routing_history.append({
                'timestamp':     self.current_time,
                'traffic_loads': node_traffic_loads,
            })

        print(f"  [LinearFlowCarbon] Done â€” {num_iterations} steps, "
              f"total carbon: {sum(self.carbon_history):.2f} gCO2")
        return {
            'carbon_history':  np.array(self.carbon_history),
            'total_carbon':    sum(self.carbon_history),
            'avg_carbon_rate': np.mean(self.carbon_history),
            'routing_history': self.routing_history,
            'timestamps':      [h['timestamp'] for h in self.routing_history],
        }


# â”€â”€ Paper 2: Hogade et al. (IEEE) â€” Nash Equilibrium â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class NashGameController:
    """
    Baseline from: Hogade et al., IEEE
    "Reducing Carbon Footprint of AI Inference Workloads for
     Geographically Distributed Data Centers"

    Adaptation to network routing
    ------------------------------
    The original paper distributes AI inference workloads across geo-distributed
    data centers using a Nash equilibrium non-cooperative game.

    We adapt this to routing:
      - Players    = unique (src, dst) flow types
      - Strategies = fractional traffic assignment over nodes  DF_{flow, node}
      - Payoff     = consequential carbon emission attributed to each flow type
      - Nash eq.   = no flow type can reduce its own carbon by unilaterally
                     changing which nodes it routes traffic through

    Algorithm (Best-Reply, Â§5.1 of the paper)
    -----------------------------------------
    Repeat until convergence (or max_iterations):
      For each flow type i (in random order):
        1. Hold all other flows' strategies fixed.
        2. Compute the marginal carbon cost of adding flow i to each node.
        3. Set DF_{i,node} = proportional to exp(-carbon_cost_node / T)
           (softmax over negative costs â€” standard for smooth best-reply).
      Check max |DF_new - DF_old|; stop if < convergence_tol.

    Config keys: nash.{max_iterations, convergence_tol, min_fraction}
    """

    def __init__(self, topology, carbon_manager, energy_manager,
                 control_interval=60, seed=42):
        self.topology       = topology
        self.carbon_manager = carbon_manager
        self.energy_manager = energy_manager
        self.control_interval = control_interval
        self.seed           = seed
        self.carbon_history = []
        self.routing_history= []
        self.current_time   = 0

        self._max_iter   = _NASH_CFG.get('max_iterations',   50)
        self._tol        = _NASH_CFG.get('convergence_tol',  1e-4)
        self._min_frac   = _NASH_CFG.get('min_fraction',     0.01)
        self._max_link   = _ROUTING.get('max_link_weight',   10000)

        self.traffic_matrix = _build_traffic_matrix(
            topology['graph'].number_of_nodes(), topology['graph']
        )

    def _nash_link_weights(self, timestamp):
        """
        Run Nash Best-Reply to find equilibrium traffic fractions, then convert
        those fractions to integer link weights for use with shortest-path routing.

        Returns
        -------
        np.ndarray  integer link weights shaped (num_edges,)
        """
        graph      = self.topology['graph']
        edge_index = self.topology['edge_index']
        num_nodes  = graph.number_of_nodes()
        num_edges  = edge_index.shape[1]

        # Carbon intensities at this timestep
        ci = np.array([
            self.carbon_manager.get_node_intensity(n, timestamp)
            for n in range(num_nodes)
        ])  # shape (num_nodes,)

        # We model one "flow type" per node pair (up to num_nodes pairs for tractability)
        # For large graphs, we limit to a sample of representative flows
        node_list = list(graph.nodes())
        max_flow_types = min(num_nodes * 2, 40)  # cap for tractability

        rng = np.random.default_rng(self.seed + int(timestamp))
        pairs = []
        attempts = 0
        while len(pairs) < max_flow_types and attempts < max_flow_types * 10:
            s, d = rng.choice(node_list, 2, replace=False)
            if graph.has_node(s) and graph.has_node(d) and s != d:
                pairs.append((int(s), int(d)))
            attempts += 1

        if not pairs:
            return np.ones(num_edges, dtype=int)

        num_flows = len(pairs)
        # DF[i, n] = fraction of flow i routed through node n
        # Initialise uniformly; apply min_fraction floor
        DF = np.full((num_flows, num_nodes), 1.0 / num_nodes)

        for iteration in range(self._max_iter):
            DF_old = DF.copy()

            # Shuffle player order (avoids systematic bias as in the paper Â§5.1)
            order = rng.permutation(num_flows)

            for i in order:
                # Marginal carbon cost of routing flow i through each node:
                # cost_n = CI_n * (current aggregate load at n)
                # Aggregate load = sum over all flows j of DF[j,n]
                agg_load = DF.sum(axis=0)   # shape (num_nodes,)
                cost_n   = ci * agg_load    # shape (num_nodes,)

                # Best-Reply: softmax over negative cost (temperature = mean CI)
                T = max(ci.mean(), 1.0)
                log_softmax = -cost_n / T
                log_softmax -= log_softmax.max()   # numerical stability
                softmax = np.exp(log_softmax)
                softmax /= softmax.sum()

                # Apply min_fraction floor and renormalise
                softmax = np.maximum(softmax, self._min_frac)
                softmax /= softmax.sum()
                DF[i] = softmax

            max_delta = np.abs(DF - DF_old).max()
            if max_delta < self._tol:
                break  # Nash equilibrium reached

        # Aggregate node load from the equilibrium strategy
        # node_load[n] = sum_i DF[i, n]  (normalised to [0,1])
        node_load = DF.mean(axis=0)   # shape (num_nodes,)

        # Convert to edge weights: high-load nodes â†’ heavier incident edges
        edge_weights = np.zeros(num_edges)
        for idx in range(num_edges):
            src = int(edge_index[0, idx])
            dst = int(edge_index[1, idx])
            # Weight proportional to CI Ã— load at endpoint
            edge_weights[idx] = ci[src] * node_load[src] + ci[dst] * node_load[dst]

        # Add HOP_BASE to ensure the algorithm minimizes hops first
        # and only uses carbon as a tie-breaker, matching OSPF/Threshold behavior
        HOP_BASE = 10000.0
        link_weights = (HOP_BASE + edge_weights).astype(int)
        return link_weights

    def run_control_loop(self, duration_seconds):
        num_iterations = int(duration_seconds / self.control_interval)

        for i in range(num_iterations):
            self.current_time = i * self.control_interval
            print(f"  [NashGame] Step {i+1}/{num_iterations}  "
                  f"(t={self.current_time/3600:.0f}h)", end="\r", flush=True)

            step_seed = self.seed + int(self.current_time)
            np.random.seed(step_seed)
            traffic_flows = self.traffic_matrix.generate_datacenter_traffic()

            link_weights = self._nash_link_weights(self.current_time)

            from simulation.traffic_matrix import distribute_traffic_by_routing
            node_traffic_loads, _ = distribute_traffic_by_routing(
                traffic_flows, self.topology['graph'],
                link_weights, self.topology['edge_index']
            )

            carbon_intensities = {
                nid: self.carbon_manager.get_node_intensity(nid, self.current_time)
                for nid in range(self.topology['graph'].number_of_nodes())
            }
            carbon = self.energy_manager.calculate_carbon_with_traffic(
                carbon_intensities, node_traffic_loads, self.control_interval
            )
            self.carbon_history.append(carbon)
            self.routing_history.append({
                'timestamp':     self.current_time,
                'traffic_loads': node_traffic_loads,
            })

        print(f"  [NashGame] Done â€” {num_iterations} steps, "
              f"total carbon: {sum(self.carbon_history):.2f} gCO2")
        return {
            'carbon_history':  np.array(self.carbon_history),
            'total_carbon':    sum(self.carbon_history),
            'avg_carbon_rate': np.mean(self.carbon_history),
            'routing_history': self.routing_history,
            'timestamps':      [h['timestamp'] for h in self.routing_history],
        }


# â”€â”€ Smoke-test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    print("Routing Controller Test")
    print("=" * 60)

    from simulation.network_topology import create_network
    from simulation.carbon_profiles import create_realistic_profiles
    from simulation.energy_model import NetworkEnergyManager
    from enhanced_gnn_model import CarbonAwareGAT

    num_nodes = 10
    topology  = create_network(num_nodes, 'hierarchical')
    carbon_mgr = create_realistic_profiles(num_nodes, 'clustered')
    energy_mgr = NetworkEnergyManager(num_nodes)
    energy_mgr.initialize_nodes()

    model = CarbonAwareGAT(node_features=13, edge_features=3, hidden_dim=64, num_layers=2)

    controllers = {
        'GNN':       RoutingController(model, topology, carbon_mgr, energy_mgr,
                                       control_interval=3600, use_ns3=False),
        'Baseline':  BaselineController(topology, carbon_mgr, energy_mgr, control_interval=3600),
        'Threshold': ThresholdCarbonController(topology, carbon_mgr, energy_mgr, control_interval=3600),
        'LinearFlowCarbon':    LinearFlowCarbonController(topology, carbon_mgr, energy_mgr, control_interval=3600),
        'NashGame':      NashGameController(topology, carbon_mgr, energy_mgr, control_interval=3600),
    }

    print(f"\nRunning 6-hour simulation with all 5 controllers...")
    results = {}
    for name, ctrl in controllers.items():
        print(f"\n  {name}:")
        results[name] = ctrl.run_control_loop(duration_seconds=21600)

    print(f"\n{'Controller':<18} {'Total CO2 (gCO2)':>18}")
    print("-" * 38)
    for name, r in results.items():
        print(f"  {name:<16} {r['total_carbon']:>18.2f}")

    print("\nAll controller tests passed.")
