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


def row_to_graph(row: pd.Series, row_index: int = 0, total_rows: int = 1,
                 aug_seed: int = 0) -> Data:
    """
    Convert one CSV flow row → a PyG graph for GNN training.

    aug_seed controls data augmentation:
      - Different side-node topologies (graph structure diversity)
      - CI perturbation on side-nodes (±20%) so the ranking loss sees
        a wider spread of clean vs dirty edges in each batch
    """
    num_hops = int(row['num_hops'])
    carbon_intensity = float(row['carbon_intensity'])
    cpu_usage = float(row['cpu_usage']) / 100.0
    protocol = str(row['protocol'])
    carbon_emission = float(row['carbon_emission'])
    byte_count = float(row['byte_count'])
    flow_duration = float(row['flow_duration'])

    # ── Temporal encoding ────────────────────────────────────────────────────
    pseudo_timestamp = (row_index / max(total_rows - 1, 1)) * 86400.0
    hour_of_day = (pseudo_timestamp / 3600) % 24
    time_factor = hour_of_day / 24.0

    # ── Graph structure (aug_seed changes the side-node RNG) ────────────────
    path_nodes = num_hops + 1
    side_nodes = 4
    num_nodes  = path_nodes + side_nodes

    rng = np.random.default_rng(
        (int(carbon_intensity * 100) + aug_seed * 9973) % (2**31)
    )

    # ── Build edges ──────────────────────────────────────────────────────────
    edges = []
    # Path edges (bidirectional)
    for i in range(path_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    # Side-node edges: connect each side node to a random path node
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
    # (carbon intensity correlates with energy source type in practice)
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
    # Side-node CI: augmented with ±20% perturbation for diverse ranking pairs
    ci_perturb = 1.0 + rng.uniform(-0.20, 0.20)
    augmented_ci = float(np.clip(carbon_intensity * ci_perturb, 50, 900))
    for _ in range(side_nodes):
        etype = ENERGY_TYPES[int(rng.integers(0, len(ENERGY_TYPES)))]
        node_types.append(etype)

    node_features = []
    for idx in range(num_nodes):
        # Side nodes use perturbed CI for diversity; path nodes use original
        node_ci = carbon_intensity if idx < path_nodes else augmented_ci
        energy_ratio = 1.0 - min(node_ci / 800.0, 1.0)
        ci_norm1     = node_ci / 1000.0
        queue_load   = cpu_usage
        degree       = sum(1 for e in edges if e[0] == idx)
        degree_norm  = degree / max(num_nodes, 1)
        ci_norm2     = node_ci / 1500.0

        one_hot = [0.0] * 6
        et = node_types[idx]
        if et in ENERGY_TYPES:
            one_hot[ENERGY_TYPES.index(et)] = 1.0

        feat = [energy_ratio, ci_norm1, queue_load, cpu_usage,
                degree_norm, time_factor, ci_norm2] + one_hot
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float)

    # ── Timestamp and target ─────────────────────────────────────────────────
    # Use fixed pseudo_timestamp (row-index based, 0–86400 s) for temporal encoder
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


def load_real_dataset(csv_path: str, augment: bool = True) -> list:
    """
    Convert every CSV row to one or more PyG Data graphs.

    Augmentation (augment=True):
        Each row generates `n_aug` graph variants with:
        - Different random CI perturbations (±20%) across side-nodes
        - Different random side-node topologies (new graph structure each time)
        This exposes the ranking loss to more diverse (clean, dirty) edge pairs
        and prevents the GNN from memorising specific CI values.
    """
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    graphs = []
    n_aug = 3 if augment else 1   # 3 variants per row → ~18 000 samples

    for row_idx, (_, row) in enumerate(df.iterrows()):
        for aug_i in range(n_aug):
            try:
                # For augmentation, shift the pseudo_timestamp slightly so
                # the temporal encoder sees varied hour-of-day values
                shifted_idx = row_idx + aug_i * (total_rows // n_aug)
                g = row_to_graph(
                    row,
                    row_index=shifted_idx % total_rows,
                    total_rows=total_rows,
                    aug_seed=aug_i,
                )
                graphs.append(g)
            except Exception as e:
                print(f"Error parsing row {row_idx}: {e}")
                pass
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

    # -- Load data with augmentation ------------------------------------------
    print("\nConverting CSV flows -> PyG graphs (with augmentation x3) ...")
    all_graphs = load_real_dataset(CSV_PATH, augment=True)
    print(f"  Loaded {len(all_graphs)} graph samples  "
          f"({len(all_graphs)//3} rows × 3 augmentations).")
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
    print("\nInitialising CarbonAwareGAT (node_features=13) ...")
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

    # ── Training with CarbonRoutingLoss ───────────────────────────────────────
    EPOCHS = 100
    LR     = 5e-4
    print(f"\nTraining for {EPOCHS} epochs  (lr={LR}, CarbonRoutingLoss) ...")
    print("  Loss = 0.4*carbon_mse + 0.4*routing_alignment + 0.2*ranking_margin(0.15)")
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    # Verify the file was written
    if os.path.exists(SAVE_PATH):
        size_kb = os.path.getsize(SAVE_PATH) / 1024
        print(f"\n[SUCCESS] Model saved to {SAVE_PATH}  ({size_kb:.1f} KB)")
    else:
        print("\n✗ Warning: model file was not found after training!")

    print("\nDone! Run the simulation to validate:")
    print("  bash run_ns3_wsl.sh --seeds 10 --hours 24 --nodes 20")
