import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import math

class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, timestamps):
        if timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)
        
        hour_of_day = (timestamps % 86400) / 3600
        day_of_week = ((timestamps // 86400) % 7)
        
        hour_rad = 2 * math.pi * hour_of_day / 24
        dow_rad = 2 * math.pi * day_of_week / 7
        
        encodings = []
        for i in range(self.embed_dim // 4):
            freq = 2 ** i
            encodings.append(torch.sin(freq * hour_rad))
            encodings.append(torch.cos(freq * hour_rad))
            encodings.append(torch.sin(freq * dow_rad))
            encodings.append(torch.cos(freq * dow_rad))
        
        return torch.stack(encodings, dim=-1)


class CarbonAwareGAT(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=128, 
                 num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        
        self.temporal_encoder = TemporalEncoder(embed_dim=32)
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                       dropout=dropout, edge_dim=hidden_dim, concat=True)
            )
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        self.carbon_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_attr, timestamp=None, batch=None):
        if timestamp is None:
            timestamp = torch.zeros(x.size(0))
        
        # Handle batched graphs
        if batch is not None:
            # Expand timestamp for each node in the batch
            num_graphs = batch.max().item() + 1
            temporal_features_list = []
            
            for graph_idx in range(num_graphs):
                # Get nodes for this graph
                node_mask = (batch == graph_idx)
                num_nodes_in_graph = node_mask.sum().item()
                
                # Get timestamp for this graph
                if timestamp.dim() > 0 and len(timestamp) > graph_idx:
                    graph_timestamp = timestamp[graph_idx]
                else:
                    graph_timestamp = timestamp if timestamp.dim() == 0 else timestamp[0]
                
                # Encode and expand for all nodes in this graph
                temp_feat = self.temporal_encoder(graph_timestamp)
                temp_feat = temp_feat.expand(num_nodes_in_graph, -1)
                temporal_features_list.append(temp_feat)
            
            temporal_features = torch.cat(temporal_features_list, dim=0)
        else:
            # Single graph mode
            temporal_features = self.temporal_encoder(timestamp)
            if temporal_features.size(0) == 1:
                temporal_features = temporal_features.expand(x.size(0), -1)
        
        x = torch.cat([x, temporal_features], dim=-1)
        x = self.node_encoder(x)
        
        edge_feat = self.edge_encoder(edge_attr)
        
        for gat in self.gat_layers:
            x = gat(x, edge_index, edge_feat)
            x = F.elu(x)
        
        src, dst = edge_index
        edge_embeddings = torch.cat([x[src], x[dst], edge_feat], dim=-1)
        link_weights = self.edge_predictor(edge_embeddings).squeeze(-1)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        carbon_predictions = self.carbon_predictor(global_mean_pool(x, batch))
        
        return link_weights, carbon_predictions


class RelativeImprovementLoss(nn.Module):
    """
    Train on relative carbon reduction percentage instead of absolute values.
    
    This teaches the model to maximize improvement percentage, which aligns
    better with the goal of carbon reduction.
    """
    def __init__(self, margin=0.10, baseline_penalty_weight=2.0):
        super().__init__()
        self.margin = margin  # Target at least 10% improvement
        self.baseline_penalty_weight = baseline_penalty_weight
    
    def forward(self, carbon_pred, carbon_target, baseline_carbon):
        """
        Args:
            carbon_pred: Predicted carbon emissions (denormalized)
            carbon_target: Target optimal carbon emissions (denormalized)
            baseline_carbon: Baseline carbon emissions (denormalized)
        
        Returns:
            total_loss, improvement_loss
        """
        # Squeeze dimensions to ensure compatibility
        carbon_pred = carbon_pred.squeeze(-1) if carbon_pred.dim() > 1 else carbon_pred
        carbon_target = carbon_target.squeeze(-1) if carbon_target.dim() > 1 else carbon_target
        baseline_carbon = baseline_carbon.squeeze(-1) if baseline_carbon.dim() > 1 else baseline_carbon
        
        # Avoid division by zero
        baseline_carbon = torch.clamp(baseline_carbon, min=1.0)
        
        # Calculate improvement percentages
        target_improvement = (baseline_carbon - carbon_target) / baseline_carbon
        pred_improvement = (baseline_carbon - carbon_pred) / baseline_carbon
        
        # Main loss: MSE on improvement percentage
        improvement_loss = F.mse_loss(pred_improvement, target_improvement)
        
        # Penalty for predictions worse than baseline (critical!)
        worse_than_baseline = F.relu(carbon_pred - baseline_carbon)
        baseline_penalty = torch.mean(worse_than_baseline / baseline_carbon)
        
        # Bonus for exceeding improvement margin
        margin_bonus = F.relu(self.margin - pred_improvement).mean()
        
        # Combine losses
        total_loss = (improvement_loss + 
                     self.baseline_penalty_weight * baseline_penalty + 
                     0.5 * margin_bonus)
        
        return total_loss, improvement_loss


class MultiObjectiveLoss(nn.Module):
    def __init__(self, carbon_weight=1.0, latency_weight=0.3, qos_weight=0.2):
        super().__init__()
        self.carbon_weight = carbon_weight
        self.latency_weight = latency_weight
        self.qos_weight = qos_weight
    
    def forward(self, predicted_weights, carbon_pred, carbon_target, 
                latency=None, qos_violation=None):
        # Ensure carbon_pred and carbon_target have matching shapes
        carbon_pred = carbon_pred.squeeze(-1) if carbon_pred.dim() > 1 else carbon_pred
        carbon_target = carbon_target.squeeze(-1) if carbon_target.dim() > 1 else carbon_target
        
        carbon_loss = F.mse_loss(carbon_pred, carbon_target)
        
        total_loss = self.carbon_weight * carbon_loss
        
        if latency is not None:
            latency_penalty = torch.mean(latency)
            total_loss += self.latency_weight * latency_penalty
        
        if qos_violation is not None:
            qos_penalty = torch.mean(F.relu(qos_violation))
            total_loss += self.qos_weight * qos_penalty
        
        return total_loss, carbon_loss


class RouteOptimizer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def optimize_routes(self, graph_data, timestamp=None):
        with torch.no_grad():
            x = graph_data.x.to(self.device)
            edge_index = graph_data.edge_index.to(self.device)
            edge_attr = graph_data.edge_attr.to(self.device)
            
            if timestamp is not None:
                timestamp = torch.tensor([timestamp], dtype=torch.float).to(self.device)
            
            link_weights, carbon_pred = self.model(x, edge_index, edge_attr, timestamp)
            
            link_weights = link_weights.cpu().numpy()
            link_weights = np.clip(link_weights * 100, 1, 65535).astype(int)
            
            # Denormalize carbon prediction (training data is scaled by 1/1000)
            carbon_actual = carbon_pred.cpu().item() * 1000.0
            
            return link_weights, carbon_actual
    
    def batch_optimize(self, graphs, timestamps=None):
        batch_data = Batch.from_data_list(graphs)
        
        with torch.no_grad():
            x = batch_data.x.to(self.device)
            edge_index = batch_data.edge_index.to(self.device)
            edge_attr = batch_data.edge_attr.to(self.device)
            batch = batch_data.batch.to(self.device)
            
            if timestamps is not None:
                timestamps = torch.tensor(timestamps, dtype=torch.float).to(self.device)
            
            link_weights, carbon_preds = self.model(x, edge_index, edge_attr, timestamps, batch)
            
            # Denormalize carbon predictions (training data is scaled by 1/1000)
            carbon_preds_actual = carbon_preds.cpu().numpy() * 1000.0
            
            return link_weights.cpu().numpy(), carbon_preds_actual


def create_graph_from_network_state(node_features, edge_index, edge_features):
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                            factor=0.5, patience=10)
    criterion = MultiObjectiveLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            link_weights, carbon_pred = model(batch.x, batch.edge_index, 
                                              batch.edge_attr, batch.timestamp, batch.batch)
            
            loss, carbon_loss = criterion(link_weights, carbon_pred, batch.carbon_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_carbon_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                link_weights, carbon_pred = model(batch.x, batch.edge_index, 
                                                  batch.edge_attr, batch.timestamp, batch.batch)
                loss, carbon_loss = criterion(link_weights, carbon_pred, batch.carbon_target)
                val_loss += loss.item()
                val_carbon_loss += carbon_loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_carbon_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Carbon: {val_carbon_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_carbon_gat.pth')
    
    return model


if __name__ == "__main__":
    print("Enhanced Carbon-Aware GAT Model")
    print("=" * 50)
    
    node_feat = 7
    edge_feat = 3
    
    model = CarbonAwareGAT(node_feat, edge_feat, hidden_dim=128, num_layers=3, num_heads=4)
    
    num_nodes = 10
    num_edges = 20
    x = torch.randn(num_nodes, node_feat)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_feat)
    timestamp = torch.tensor(43200.0)
    
    link_weights, carbon_pred = model(x, edge_index, edge_attr, timestamp)
    
    print(f"Input: {num_nodes} nodes, {num_edges} edges")
    print(f"Output: {link_weights.shape[0]} link weights, carbon pred: {carbon_pred.item():.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model test passed")
