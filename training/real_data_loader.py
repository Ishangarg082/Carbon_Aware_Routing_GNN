"""
Dataset loader for real carbon network data (carbon_network_data.csv)
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


class CarbonDataset:
    """
    Loads and preprocesses carbon_network_data.csv
    
    Features:
        - num_hops, packet_count, byte_count, flow_duration, cpu_usage,
          carbon_intensity, protocol
    
    Target:
        - carbon_emission
    """
    
    def __init__(self, csv_path='carbon_network_data.csv', test_size=0.15, val_size=0.15, random_state=42):
        self.csv_path = csv_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Load and preprocess
        self.df = pd.read_csv(csv_path)
        self._preprocess()
        
        # Split data
        self._split_data()
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def _preprocess(self):
        """One-hot encode protocol and extract features"""
        # One-hot encode protocol
        df_encoded = pd.get_dummies(self.df, columns=['protocol'], drop_first=False)
        
        # Feature columns
        self.feature_cols = ['num_hops', 'packet_count', 'byte_count', 'flow_duration',
                             'cpu_usage', 'carbon_intensity',
                             'protocol_TCP', 'protocol_UDP', 'protocol_ICMP']
        
        self.X = df_encoded[self.feature_cols].values
        self.y = df_encoded['carbon_emission'].values
        
        print(f"Loaded {len(self.df)} samples from {self.csv_path}")
        print(f"Feature dimensions: {self.X.shape}")
    
    def _split_data(self):
        """Split into train/val/test"""
        # First split: train vs (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y,
            test_size=(self.test_size + self.val_size),
            random_state=self.random_state
        )
        
        # Second split: val vs test
        val_ratio = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=self.random_state
        )
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        print(f"\nData splits:")
        print(f"  Train: {len(self.X_train)} samples ({len(self.X_train)/len(self.X)*100:.1f}%)")
        print(f"  Val:   {len(self.X_val)} samples ({len(self.X_val)/len(self.X)*100:.1f}%)")
        print(f"  Test:  {len(self.X_test)} samples ({len(self.X_test)/len(self.X)*100:.1f}%)")
    
    def get_train_tensors(self):
        """Returns (X_train, y_train) as PyTorch tensors"""
        return (
            torch.FloatTensor(self.X_train_scaled),
            torch.FloatTensor(self.y_train).view(-1, 1)
        )
    
    def get_val_tensors(self):
        """Returns (X_val, y_val) as PyTorch tensors"""
        return (
            torch.FloatTensor(self.X_val_scaled),
            torch.FloatTensor(self.y_val).view(-1, 1)
        )
    
    def get_test_tensors(self):
        """Returns (X_test, y_test) as PyTorch tensors"""
        return (
            torch.FloatTensor(self.X_test_scaled),
            torch.FloatTensor(self.y_test).view(-1, 1)
        )
    
    def save_scaler(self, path='models/feature_scaler.pkl'):
        """Save scaler for inference"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {path}")
    
    def get_statistics(self):
        """Return dataset statistics"""
        return {
            'total_samples': len(self.df),
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'carbon_mean': self.y.mean(),
            'carbon_std': self.y.std(),
            'carbon_min': self.y.min(),
            'carbon_max': self.y.max()
        }


if __name__ == "__main__":
    print("="*70)
    print("CARBON DATASET TEST")
    print("="*70)
    
    # Load dataset
    dataset = CarbonDataset()
    
    # Get statistics
    stats = dataset.get_statistics()
    print(f"\nDataset Statistics:")
    for key, value in stats.items():
        if 'carbon' in key and key != 'total_samples':
            print(f"  {key:20s}: {value:.6f}")
        else:
            print(f"  {key:20s}: {value}")
    
    # Test tensor conversion
    X_train, y_train = dataset.get_train_tensors()
    X_val, y_val = dataset.get_val_tensors()
    X_test, y_test = dataset.get_test_tensors()
    
    print(f"\nTensor shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Save scaler
    dataset.save_scaler()
    
    print("\n" + "="*70)
    print("Dataset ready for training!")
    print("="*70)
