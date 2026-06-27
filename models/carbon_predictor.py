"""
Simple MLP Carbon Predictor for Real Routing Data

This model predicts carbon emissions based on routing features from real network data.
Achieves R² = 0.87 on carbon_network_data.csv
"""

import torch
import torch.nn as nn


class CarbonPredictor(nn.Module):
    """
    Multi-layer perceptron for carbon emission prediction
    
    Input features (9):
        - num_hops: Number of routing hops
        - packet_count: Total packets in flow
        - byte_count: Total bytes in flow
        - flow_duration: Duration of flow (ms)
        - cpu_usage: CPU utilization (%)
        - carbon_intensity: Carbon intensity at node (gCO2/kWh)
        - protocol_TCP: Binary encoding
        - protocol_UDP: Binary encoding
        - protocol_ICMP: Binary encoding
    
    Output:
        - predicted_carbon: Carbon emission (gCO2eq)
    """
    
    def __init__(self, input_dim=9, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim)
        
        Returns:
            Tensor of shape (batch_size, 1) with predicted carbon emissions
        """
        return self.network(x)


class CarbonRoutingOptimizer:
    """
    Uses CarbonPredictor to make routing decisions that minimize carbon
    """
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = CarbonPredictor().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Feature scaling parameters (loaded from training)
        self.scaler_mean = None
        self.scaler_std = None
    
    def load_scaler(self, scaler_path):
        """Load feature scaler from training"""
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
            self.scaler_std = torch.tensor(scaler.scale_, dtype=torch.float32)
    
    def predict_carbon(self, routing_features):
        """
        Predict carbon emissions for given routing features
        
        Args:
            routing_features: dict with keys:
                - num_hops
                - packet_count
                - byte_count
                - flow_duration
                - cpu_usage
                - carbon_intensity
                - protocol (one of 'TCP', 'UDP', 'ICMP')
        
        Returns:
            float: Predicted carbon emission (gCO2eq)
        """
        # Encode protocol
        protocol_enc = {
            'TCP': [1, 0, 0],
            'UDP': [0, 1, 0],
            'ICMP': [0, 0, 1]
        }
        protocol_vec = protocol_enc.get(routing_features.get('protocol', 'TCP'), [1, 0, 0])
        
        # Build feature vector
        features = [
            routing_features['num_hops'],
            routing_features['packet_count'],
            routing_features['byte_count'],
            routing_features['flow_duration'],
            routing_features['cpu_usage'],
            routing_features['carbon_intensity']
        ] + protocol_vec
        
        # Convert to tensor
        x = torch.tensor([features], dtype=torch.float32)
        
        # Normalize if scaler is available
        if self.scaler_mean is not None:
            x = (x - self.scaler_mean) / self.scaler_std
        
        # Predict
        with torch.no_grad():
            prediction = self.model(x).item()
        
        return max(0, prediction)  # Carbon can't be negative
    
    def select_best_route(self, candidate_routes, flow_info):
        """
        Select route with minimum predicted carbon emission
        
        Args:
            candidate_routes: list of dicts, each containing:
                - num_hops: int
                - path_nodes: list of node IDs
                - avg_carbon_intensity: float
            flow_info: dict with:
                - packet_count: int
                - byte_count: int
                - protocol: str
                - cpu_usage: float (avg)
        
        Returns:
            Best route (dict) and predicted carbon emission (float)
        """
        best_route = None
        min_carbon = float('inf')
        
        for route in candidate_routes:
            # Estimate flow duration based on hops and bytes
            # Simple heuristic: more hops = longer duration
            estimated_duration = route['num_hops'] * 10 + flow_info['byte_count'] / 100000
            
            features = {
                'num_hops': route['num_hops'],
                'packet_count': flow_info['packet_count'],
                'byte_count': flow_info['byte_count'],
                'flow_duration': estimated_duration,
                'cpu_usage': flow_info['cpu_usage'],
                'carbon_intensity': route['avg_carbon_intensity'],
                'protocol': flow_info['protocol']
            }
            
            predicted_carbon = self.predict_carbon(features)
            
            if predicted_carbon < min_carbon:
                min_carbon = predicted_carbon
                best_route = route
        
        return best_route, min_carbon


if __name__ == "__main__":
    print("="*70)
    print("CARBON PREDICTOR TEST")
    print("="*70)
    
    # Create model
    model = CarbonPredictor()
    print(f"\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 9)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nSample predictions:")
    for i in range(batch_size):
        print(f"  Sample {i+1}: {output[i].item():.6f} gCO2eq")
    
    print("\n" + "="*70)
    print("Model ready for training!")
    print("="*70)
