"""
Train CarbonAwareGAT (13-feature) on real data from carbon_network_data.csv

Strategy:
  Each row in the CSV represents one observed network flow with its path,
  energy characteristics, and measured carbon emission.
  
  We synthesise a small graph for each flow:
    - num_hops+1 nodes in a chain (the routed path)
    - 4-8 additional "alternative-path" nodes connected randomly
    - Node features (13-dim):
        [energy_ratio, carbon_intensity/1000, queue_load, cpu_usage,
         degree/num_nodes, time_factor, carbon_intensity/1500,
         solar_onehot, wind_onehot, hydro_onehot,
         coal_onehot, nuclear_onehot, mixed_onehot]
    - Edge features (3-dim): [bw, delay, utilisation]
    - carbon_target: the measured carbon_emission from the CSV row
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from enhanced_gnn_model import CarbonAwareGAT, train_model

# ---------------------------------------------------------------------------
# Energy source types and their one-hot index
# ---------------------------------------------------------------------------
ENERGY_TYPES = ['solar', 'wind', 'hydro', 'coal', 'nuclear', 'mixed']
PROTOCOL_TO_LOAD = {'TCP': 0.7, 'UDP': 0.5, 'ICMP': 0.2}


def row_to_graph(row: pd.Series) -> Data:
    """
    Convert one CSV flow row → a PyG graph for GNN training.
    
    The graph simulates the path taken by the flow plus a few side nodes
    that represent alternative routing options.
    """
    num_hops = int(row['num_hops'])
    carbon_intensity = float(row['carbon_intensity'])
    cpu_usage = float(row['cpu_usage']) / 100.0
    protocol = str(row['protocol'])
    carbon_emission = float(row['carbon_emission'])
    byte_count = float(row['byte_count'])
    flow_duration = float(row['flow_duration'])

    # Derived scalar for time-of-day (we use flow_duration as a proxy for "time slice")
    # Normalise to [0, 86400] to simulate a timestamp
    pseudo_timestamp = (flow_duration % 86400)
    hour_of_day = (pseudo_timestamp / 3600) % 24
    time_factor = 0.3 + 0.5 * np.sin(hour_of_day * np.pi / 12) ** 2

    # Number of nodes in the graph: path nodes + 4 side nodes
    path_nodes = num_hops + 1          # nodes 0 … num_hops are on the path
    side_nodes = 4
    num_nodes = path_nodes + side_nodes

    # ── Build edges ─────────────────────────────────────────────────────────
    edges = []
    # Path edges (bidirectional)
    for i in range(path_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    # Side-node edges: connect each side node to a random path node
    rng = np.random.default_rng(int(carbon_intensity * 100) % (2**31))
    for s in range(path_nodes, num_nodes):
        target = int(rng.integers(0, path_nodes))
        edges.append([s, target])
        edges.append([target, s])
    # Ensure at least one edge exists (single-hop paths have 2 edges from sides)
    if not edges:
        edges = [[0, 0]]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_edges = edge_index.size(1)

    # ── Edge features (3-dim) ────────────────────────────────────────────────
    # [normalised_bandwidth, normalised_delay, utilisation]
    bw = float(byte_count * 8 / max(flow_duration * 1e6, 1.0))   # Gbps
    bw_norm = min(bw / 10.0, 1.0)
    delay_norm = min(flow_duration / 600.0, 1.0)
    util = PROTOCOL_TO_LOAD.get(protocol, 0.5)
    edge_feat = torch.tensor(
        [[bw_norm, delay_norm, util]] * num_edges, dtype=torch.float
    )

    # ── Assign energy source type to each node ────────────────────────────────
    # Path nodes: type is derived from carbon_intensity range
    if carbon_intensity < 150:
        path_type = 'nuclear'
    elif carbon_intensity < 250:
        path_type = 'hydro'
    elif carbon_intensity < 350:
        path_type = 'solar'
    elif carbon_intensity < 450:
        path_type = 'wind'
    elif carbon_intensity < 550:
        path_type = 'mixed'
    else:
        path_type = 'coal'

    node_types = [path_type] * path_nodes
    for _ in range(side_nodes):
        node_types.append(ENERGY_TYPES[int(rng.integers(0, len(ENERGY_TYPES)))])

    # ── Node features (13-dim) ───────────────────────────────────────────────
    node_features = []
    for idx in range(num_nodes):
        # Core features
        energy_ratio = 1.0 - min(carbon_intensity / 800.0, 1.0)
        ci_norm1 = carbon_intensity / 1000.0
        queue_load = time_factor * (0.5 + 0.3 * np.sin(idx * 1.5))
        degree = sum(1 for e in edges if e[0] == idx)
        degree_norm = degree / max(num_nodes, 1)
        ci_norm2 = carbon_intensity / 1500.0

        # 6-dim one-hot energy source
        one_hot = [0.0] * 6
        et = node_types[idx]
        if et in ENERGY_TYPES:
            one_hot[ENERGY_TYPES.index(et)] = 1.0

        feat = [energy_ratio, ci_norm1, queue_load, cpu_usage,
                degree_norm, time_factor, ci_norm2] + one_hot
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float)

    # ── Timestamp and target ─────────────────────────────────────────────────
    timestamp = torch.tensor([pseudo_timestamp], dtype=torch.float)
    # carbon_target is the raw emission; model predicts * 1000 scale
    carbon_target = torch.tensor([[carbon_emission * 1000.0]], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_feat,
        timestamp=timestamp,
        carbon_target=carbon_target,
    )


def load_real_dataset(csv_path: str) -> list:
    """Convert every CSV row to a PyG Data graph."""
    df = pd.read_csv(csv_path)
    graphs = []
    for _, row in df.iterrows():
        try:
            graphs.append(row_to_graph(row))
        except Exception as e:
            pass   # skip malformed rows silently
    return graphs


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    CSV_PATH = os.path.join(os.path.dirname(__file__), 'carbon_network_data.csv')
    SAVE_PATH = os.path.join(os.path.dirname(__file__), 'best_carbon_gat.pth')

    print("=" * 65)
    print("  CarbonAwareGAT — Real-Data Training")
    print(f"  CSV : {CSV_PATH}")
    print(f"  Save: {SAVE_PATH}")
    print("=" * 65)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nConverting CSV flows → PyG graphs …")
    all_graphs = load_real_dataset(CSV_PATH)
    print(f"  Loaded {len(all_graphs)} graph samples.")
    print(f"  Node-feature dim : {all_graphs[0].x.shape[1]}  (expected 13)")
    print(f"  Edge-feature dim : {all_graphs[0].edge_attr.shape[1]}  (expected 3)")

    # Train/val split (80 / 20)
    n_train = int(0.8 * len(all_graphs))
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(all_graphs))
    train_graphs = [all_graphs[i] for i in idx[:n_train]]
    val_graphs   = [all_graphs[i] for i in idx[n_train:]]
    print(f"  Train: {len(train_graphs)}  |  Val: {len(val_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=64)

    # ── Model ────────────────────────────────────────────────────────────────
    print("\nInitialising CarbonAwareGAT (node_features=13) …")
    model = CarbonAwareGAT(
        node_features=13,
        edge_features=3,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        dropout=0.2,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Training ─────────────────────────────────────────────────────────────
    EPOCHS = 50
    LR     = 5e-4
    print(f"\nTraining for {EPOCHS} epochs  (lr={LR}) …")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    # Verify the file was written
    if os.path.exists(SAVE_PATH):
        size_kb = os.path.getsize(SAVE_PATH) / 1024
        print(f"\n✓ Model saved to {SAVE_PATH}  ({size_kb:.1f} KB)")
    else:
        print("\n✗ Warning: model file was not found after training!")

    print("\nDone! You can now run:")
    print("  python run_multi_seed_experiment.py --seeds 5 --hours 12 --nodes 50")
