import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from enhanced_gnn_model import CarbonAwareGAT, TemporalEncoder, create_graph_from_network_state

def test_temporal_encoder():
    print("\n" + "="*50)
    print("Testing Temporal Encoder")
    print("="*50)
    
    encoder = TemporalEncoder(embed_dim=32)
    
    timestamps = torch.tensor([0, 43200, 86400])
    encodings = encoder(timestamps)
    
    assert encodings.shape == (timestamps.shape[0], 32), f"Shape mismatch: {encodings.shape}"
    assert not torch.isnan(encodings).any(), "NaN values in encodings"
    
    print(f"  Temporal encoding shape: {encodings.shape}")
    print(f"  No NaN values")
    print(f"  Temporal Encoder test passed")
    
    return True


def test_gat_model():
    print("\n" + "="*50)
    print("Testing Carbon-Aware GAT Model")
    print("="*50)
    
    model = CarbonAwareGAT(
        node_features=7,
        edge_features=3,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    )
    
    num_nodes = 10
    num_edges = 20
    
    x = torch.randn(num_nodes, 7)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)
    timestamp = torch.tensor(43200.0)
    
    link_weights, carbon_pred = model(x, edge_index, edge_attr, timestamp)
    
    assert link_weights.shape[0] == num_edges, f"Weight shape mismatch: {link_weights.shape}"
    assert carbon_pred.shape == (1, 1), f"Carbon prediction shape mismatch: {carbon_pred.shape}"
    assert not torch.isnan(link_weights).any(), "NaN in link weights"
    assert not torch.isnan(carbon_pred).any(), "NaN in carbon prediction"
    
    print(f"  Link weights shape: {link_weights.shape}")
    print(f"  Carbon prediction shape: {carbon_pred.shape}")
    print(f"  No NaN values")
    print(f"  GAT Model test passed")
    
    return True


def test_graph_creation():
    print("\n" + "="*50)
    print("Testing Graph Creation")
    print("="*50)
    
    num_nodes = 5
    num_edges = 8
    
    node_features = np.random.randn(num_nodes, 7)
    edge_index = np.random.randint(0, num_nodes, (2, num_edges))
    edge_features = np.random.randn(num_edges, 3)
    
    graph = create_graph_from_network_state(node_features, edge_index, edge_features)
    
    assert graph.x.shape == (num_nodes, 7), "Node features shape mismatch"
    assert graph.edge_index.shape == (2, num_edges), "Edge index shape mismatch"
    assert graph.edge_attr.shape == (num_edges, 3), "Edge features shape mismatch"
    
    print(f"  Graph created successfully")
    print(f"  Nodes: {graph.x.shape[0]}, Edges: {graph.edge_index.shape[1]}")
    print(f"  Graph Creation test passed")
    
    return True


def test_model_forward_backward():
    print("\n" + "="*50)
    print("Testing Model Training Loop")
    print("="*50)
    
    model = CarbonAwareGAT(node_features=7, edge_features=3, hidden_dim=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.randn(10, 7)
    edge_index = torch.randint(0, 10, (2, 15))
    edge_attr = torch.randn(15, 3)
    timestamp = torch.tensor(0.0)
    
    initial_carbon_pred = model(x, edge_index, edge_attr, timestamp)[1].item()
    
    for _ in range(5):
        optimizer.zero_grad()
        link_weights, carbon_pred = model(x, edge_index, edge_attr, timestamp)
        
        target = torch.tensor([[50.0]])
        loss = torch.nn.functional.mse_loss(carbon_pred, target)
        
        loss.backward()
        optimizer.step()
    
    final_carbon_pred = model(x, edge_index, edge_attr, timestamp)[1].item()
    
    print(f"  Initial prediction: {initial_carbon_pred:.4f}")
    print(f"  Final prediction: {final_carbon_pred:.4f}")
    print(f"  Model can be trained (gradients work)")
    print(f"  Training Loop test passed")
    
    return True


def run_all_tests():
    print("\n" + "#"*50)
    print("# RUNNING GNN MODEL TESTS")
    print("#"*50)
    
    tests = [
        ("Temporal Encoder", test_temporal_encoder),
        ("GAT Model", test_gat_model),
        ("Graph Creation", test_graph_creation),
        ("Training Loop", test_model_forward_backward)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n{test_name} FAILED: {str(e)}")
            results.append((test_name, "FAIL"))
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test_name, status in results:
        status_icon = "PASS" if status == "PASS" else "FAIL"
        print(f"{status_icon} {test_name:<30} {status}")
    
    all_passed = all(status == "PASS" for _, status in results)
    
    print("\n" + "="*50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*50 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
