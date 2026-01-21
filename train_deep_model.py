"""
GPU-Accelerated Deep Learning Model for Slide Transition Detection
Uses PyTorch for GPU acceleration with fallback to CPU
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class SlideTransitionDataset(Dataset):
    """PyTorch dataset for slide transitions"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SlideTransitionNet(nn.Module):
    """
    Deep neural network for slide transition detection
    Architecture: 6 input features -> 64 -> 128 -> 64 -> 32 -> 1 output
    Uses BatchNorm, Dropout, and ReLU activation
    """
    def __init__(self, input_dim=6, dropout=0.3):
        super(SlideTransitionNet, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 4
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            
            # Output layer
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()


def load_training_data(base_csv, extra_positives=None, extra_negatives=None):
    """Load and combine all training data"""
    print(f"Loading base dataset: {base_csv}")
    df_base = pd.read_csv(base_csv)
    
    dfs = [df_base]
    
    if extra_positives:
        print(f"Loading extra positives: {extra_positives}")
        df_pos = pd.read_csv(extra_positives)
        dfs.append(df_pos)
    
    if extra_negatives:
        print(f"Loading extra negatives: {extra_negatives}")
        df_neg = pd.read_csv(extra_negatives)
        dfs.append(df_neg)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")
    print(f"Positives: {df['is_transition'].sum()}, Negatives: {(~df['is_transition']).sum()}")
    
    return df


def train_model(args):
    """Train the deep learning model"""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    df = load_training_data(args.dataset, args.extra_positives, args.extra_negatives)
    
    # Prepare features
    feature_cols = ['content_fullness', 'frame_quality', 'is_occluded', 
                    'skin_ratio', 'edge_change', 'frame_diff_mean']
    
    X = df[feature_cols].values
    y = df['is_transition'].values.astype(np.float32)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create datasets and dataloaders
    train_dataset = SlideTransitionDataset(X_train, y_train)
    test_dataset = SlideTransitionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = SlideTransitionNet(input_dim=len(feature_cols), dropout=args.dropout).to(device)
    
    # Loss and optimizer
    # Use weighted loss to handle class imbalance
    pos_weight = torch.tensor([len(y_train) / y_train.sum() - 1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training loop
    print("\nTraining...")
    best_f1 = 0.0
    best_model_state = None
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                probs = outputs.cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(y_batch.numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        # Update learning rate
        scheduler.step(f1)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f} AUC: {auc:.3f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Transition', 'Transition']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"\nROC AUC Score: {auc:.4f}")
    except:
        pass
    
    # Save model
    model_path = Path(args.output) / 'trained_model_deep.pth'
    scaler_path = Path(args.output) / 'model_deep_normalization.pkl'
    config_path = Path(args.output) / 'model_deep_config.json'
    
    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'input_dim': len(feature_cols),
        'dropout': args.dropout,
        'device_trained': str(device)
    }, model_path)
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save config
    config = {
        'feature_columns': feature_cols,
        'model_type': 'pytorch_deep',
        'architecture': 'SlideTransitionNet',
        'input_dim': len(feature_cols),
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'best_f1': float(best_f1),
        'device_trained': str(device)
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print(f"✓ Config saved: {config_path}")
    
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description='Train deep learning model for slide transitions')
    parser.add_argument('--dataset', type=str, default='models/labeled_dataset.csv',
                        help='Path to base training dataset')
    parser.add_argument('--extra-positives', type=str, default='models/hard_positives.csv',
                        help='Path to extra positive samples')
    parser.add_argument('--extra-negatives', type=str, default='models/hard_negatives.csv',
                        help='Path to extra negative samples')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory for model files')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()
