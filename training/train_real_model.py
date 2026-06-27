"""
Training script for Carbon Predictor on real network data
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from training.real_data_loader import CarbonDataset
from models.carbon_predictor import CarbonPredictor
import numpy as np


def train_epoch(model, optimizer, criterion, X_train, y_train, batch_size=32):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Shuffle data
    indices = torch.randperm(len(X_train))
    
    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, criterion, X, y):
    """Evaluate model"""
    model.eval()
    with torch.no_grad():
        pred = model(X)
        loss = criterion(pred, y)
        
        # Calculate metrics
        mse = loss.item()
        mae = torch.mean(torch.abs(pred - y)).item()
        
        # R² score
        y_mean = torch.mean(y)
        ss_tot = torch.sum((y - y_mean) ** 2)
        ss_res = torch.sum((y - pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2 = r2.item()
    
    return mse, mae, r2


def main():
    print("="*70)
    print("CARBON PREDICTOR TRAINING - REAL DATA")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = CarbonDataset(csv_path='carbon_network_data.csv')
    dataset.save_scaler('models/feature_scaler.pkl')
    
    X_train, y_train = dataset.get_train_tensors()
    X_val, y_val = dataset.get_val_tensors()
    X_test, y_test = dataset.get_test_tensors()
    
    # Create model
    input_dim = X_train.shape[1]
    model = CarbonPredictor(input_dim=input_dim, hidden_dims=[128, 64, 32], dropout=0.2)
    
    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden layers: [128, 64, 32]")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training configuration
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    num_epochs = 200
    batch_size = 32
    early_stop_patience = 20
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: 0.001")
    print(f"  Early stopping patience: {early_stop_patience}")
    
    print(f"\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, criterion, X_train, y_train, batch_size)
        val_mse, val_mae, val_r2 = evaluate(model, criterion, X_val, y_val)
        
        # Scheduler step
        scheduler.step(val_mse)
        
        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), 'models/best_carbon_predictor.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 20 == 0 or patience_counter == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | "
                  f"Val MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.4f} | "
                  f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('models/best_carbon_predictor.pth'))
    
    # Final evaluation
    print(f"\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    # Validation set
    val_mse, val_mae, val_r2 = evaluate(model, criterion, X_val, y_val)
    print(f"\nValidation Set:")
    print(f"  MSE: {val_mse:.6f}")
    print(f"  MAE: {val_mae:.6f}")
    print(f"  R²:  {val_r2:.4f}")
    
    # Test set
    test_mse, test_mae, test_r2 = evaluate(model, criterion, X_test, y_test)
    print(f"\nTest Set:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  R²:  {test_r2:.4f}")
    
    # Carbon reduction analysis
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).numpy().flatten()
    
    test_actual = y_test.numpy().flatten()
    
    # Calculate reduction opportunities
    improvements = []
    for i in range(len(test_actual)):
        if test_pred[i] < test_actual[i]:
            reduction = ((test_actual[i] - test_pred[i]) / test_actual[i]) * 100
            improvements.append(reduction)
        else:
            improvements.append(0)
    
    avg_improvement = np.mean(improvements)
    positive_count = len([x for x in improvements if x > 0])
    positive_pct = (positive_count / len(improvements)) * 100
    
    print(f"\n" + "="*70)
    print("CARBON REDUCTION POTENTIAL")
    print("="*70)
    print(f"  Average improvement: {avg_improvement:.2f}%")
    print(f"  Samples with reduction: {positive_count}/{len(improvements)} ({positive_pct:.1f}%)")
    if positive_count > 0:
        positive_improvements = [x for x in improvements if x > 0]
        print(f"  Avg improvement (positive only): {np.mean(positive_improvements):.2f}%")
    
    print(f"\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: models/best_carbon_predictor.pth")
    print(f"Scaler saved to: models/feature_scaler.pkl")
    print(f"\nReady for NS3 integration!")
    print("="*70)


if __name__ == "__main__":
    main()
