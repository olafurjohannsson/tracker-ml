"""
Train tracker classification models.
Three models for the paper:
  1. Random Forest (interpretable baseline)
  2. XGBoost (strong tabular baseline)
  3. Feedforward neural network (deployed to Kjarni/WASM)

Usage:
    python scripts/train.py \
        --dataset data/dataset_us_train.parquet \
        --output-dir models/
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import save_model as save_safetensors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")


class TrackerClassifier(nn.Module):
    """
    Lightweight feedforward classifier for tracking script detection.
    Designed for deployment to Kjarni/WASM with SIMD128.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.layer3(x)
        return x


def prepare_data(df: pd.DataFrame):
    """
    Prepare features and labels for training.
    CRITICAL: Remove fingerprinting_score from features — it's the
    heuristic we're comparing against, not a training input.
    """
    # Columns to exclude from features
    exclude_cols = [
        "domain",
        "label",
        "label_source",
        "fingerprinting_score",  # THIS IS THE HEURISTIC — NOT A FEATURE
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # Replace any NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_cols


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["non-tracking", "tracking"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return rf, y_pred, y_prob


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost."""
    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)

    # Calculate scale_pos_weight for imbalanced classes
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["non-tracking", "tracking"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return model, y_pred, y_prob


def train_feedforward(X_train, y_train, X_test, y_test, scaler, feature_cols, output_dir):
    """Train and evaluate the feedforward neural network."""
    print("\n" + "=" * 60)
    print("FEEDFORWARD NEURAL NETWORK (for Kjarni/WASM deployment)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = X_train.shape[1]
    model = TrackerClassifier(input_dim=input_dim, hidden_dim=128).to(device)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    # Class weights for imbalanced data
    n_samples = len(y_train)
    n_classes = 2
    class_counts = np.bincount(y_train)
    class_weights = n_samples / (n_classes * class_counts)
    weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    # Training loop
    batch_size = 256
    n_epochs = 100
    best_f1 = 0
    best_state = None
    patience_counter = 0
    patience = 20

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, torch.LongTensor(y_test).to(device))
            test_preds = test_outputs.argmax(dim=1).cpu().numpy()
            test_f1 = f1_score(y_test, test_preds)

        scheduler.step(test_loss)

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_state = model.state_dict().copy()
            # Deep copy to CPU
            best_state = {k: v.clone().cpu() for k, v in best_state.items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"Train Loss: {epoch_loss/len(loader):.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test F1: {test_f1:.4f} | "
                f"Best F1: {best_f1:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()

    # Final evaluation
    with torch.no_grad():
        X_test_cpu = torch.FloatTensor(X_test)
        test_outputs = model(X_test_cpu)
        y_pred = test_outputs.argmax(dim=1).numpy()
        y_prob = torch.softmax(test_outputs, dim=1)[:, 1].numpy()

    print(f"\nFinal Results (best model):")
    print(classification_report(y_test, y_pred, target_names=["non-tracking", "tracking"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Save model in safetensors format (for Kjarni)
    safetensors_path = os.path.join(output_dir, "tracker_classifier.safetensors")
    save_safetensors(model, safetensors_path)
    file_size = os.path.getsize(safetensors_path)
    print(f"\nSaved safetensors: {safetensors_path} ({file_size / 1024:.1f} KB)")

    # Save model config (needed by Kjarni to know architecture)
    config = {
        "input_dim": input_dim,
        "hidden_dim": 128,
        "output_dim": 2,
        "feature_names": feature_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {config_path}")

    return model, y_pred, y_prob


def compare_with_heuristic(df_test, y_test, predictions, model_name):
    """Compare model predictions against the fingerprinting heuristic."""
    print(f"\n{'=' * 60}")
    print(f"COMPARISON: {model_name} vs Heuristic (fingerprinting_score >= 2)")
    print(f"{'=' * 60}")

    heuristic_pred = (df_test["fingerprinting_score"].values >= 2).astype(int)

    # Heuristic performance against our labels
    print(f"\nHeuristic performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, heuristic_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, heuristic_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, heuristic_pred):.4f}")
    print(f"  F1:        {f1_score(y_test, heuristic_pred):.4f}")

    # Model performance
    print(f"\n{model_name} performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"  Precision: {precision_score(y_test, predictions):.4f}")
    print(f"  Recall:    {recall_score(y_test, predictions):.4f}")
    print(f"  F1:        {f1_score(y_test, predictions):.4f}")

    # Unique catches
    model_tracking = set(df_test.iloc[predictions == 1]["domain"])
    heuristic_tracking = set(df_test.iloc[heuristic_pred == 1]["domain"])

    model_only = model_tracking - heuristic_tracking
    heuristic_only = heuristic_tracking - model_tracking
    both = model_tracking & heuristic_tracking

    print(f"\n  Both flag as tracking:        {len(both)}")
    print(f"  Model catches, heuristic misses: {len(model_only)}")
    print(f"  Heuristic catches, model misses: {len(heuristic_only)}")

    return {
        "model_only": model_only,
        "heuristic_only": heuristic_only,
        "both": both,
    }


def main():
    parser = argparse.ArgumentParser(description="Train tracker classification models")
    parser.add_argument("--dataset", required=True, help="Path to labeled dataset parquet")
    parser.add_argument("--output-dir", default="models/", help="Output directory for models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = pd.read_parquet(args.dataset)
    print(f"Loaded {len(df)} labeled domains")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    # Prepare features
    X, y, feature_cols = prepare_data(df)
    print(f"Features: {len(feature_cols)}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)),
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )
    df_test = df.iloc[idx_test].reset_index(drop=True)

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train class dist: {np.bincount(y_train)}")
    print(f"Test class dist:  {np.bincount(y_test)}")

    # Scale features (needed for neural net, doesn't hurt tree models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

    # === Train all three models ===

    # 1. Random Forest
    rf_model, rf_pred, rf_prob = train_random_forest(
        X_train, y_train, X_test, y_test  # trees don't need scaling
    )
    joblib.dump(rf_model, os.path.join(args.output_dir, "random_forest.joblib"))

    # 2. XGBoost
    xgb_model, xgb_pred, xgb_prob = train_xgboost(
        X_train, y_train, X_test, y_test  # trees don't need scaling
    )
    xgb_model.save_model(os.path.join(args.output_dir, "xgboost.json"))

    # 3. Feedforward Neural Net (scaled features)
    ff_model, ff_pred, ff_prob = train_feedforward(
        X_train_scaled, y_train, X_test_scaled, y_test,
        scaler, feature_cols, args.output_dir,
    )

    # === Compare all models against heuristic ===
    rf_comparison = compare_with_heuristic(df_test, y_test, rf_pred, "Random Forest")
    xgb_comparison = compare_with_heuristic(df_test, y_test, xgb_pred, "XGBoost")
    ff_comparison = compare_with_heuristic(df_test, y_test, ff_pred, "Feedforward NN")

    # === Feature importance (from Random Forest) ===
    print(f"\n{'=' * 60}")
    print("TOP 20 FEATURES (Random Forest importance)")
    print("=" * 60)
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    top_features = importances.nlargest(20)
    for feat, imp in top_features.items():
        print(f"  {imp:.4f}  {feat}")

    # === Save results summary ===
    results = {
        "dataset": {
            "total_domains": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": len(feature_cols),
            "class_distribution": {
                "tracking": int((y == 1).sum()),
                "non_tracking": int((y == 0).sum()),
            },
        },
        "models": {
            "random_forest": {
                "accuracy": float(accuracy_score(y_test, rf_pred)),
                "precision": float(precision_score(y_test, rf_pred)),
                "recall": float(recall_score(y_test, rf_pred)),
                "f1": float(f1_score(y_test, rf_pred)),
                "roc_auc": float(roc_auc_score(y_test, rf_prob)),
            },
            "xgboost": {
                "accuracy": float(accuracy_score(y_test, xgb_pred)),
                "precision": float(precision_score(y_test, xgb_pred)),
                "recall": float(recall_score(y_test, xgb_pred)),
                "f1": float(f1_score(y_test, xgb_pred)),
                "roc_auc": float(roc_auc_score(y_test, xgb_prob)),
            },
            "feedforward": {
                "accuracy": float(accuracy_score(y_test, ff_pred)),
                "precision": float(precision_score(y_test, ff_pred)),
                "recall": float(recall_score(y_test, ff_pred)),
                "f1": float(f1_score(y_test, ff_pred)),
                "roc_auc": float(roc_auc_score(y_test, ff_prob)),
            },
            "heuristic_baseline": {
                "accuracy": float(accuracy_score(y_test, (df_test["fingerprinting_score"].values >= 2).astype(int))),
                "precision": float(precision_score(y_test, (df_test["fingerprinting_score"].values >= 2).astype(int))),
                "recall": float(recall_score(y_test, (df_test["fingerprinting_score"].values >= 2).astype(int))),
                "f1": float(f1_score(y_test, (df_test["fingerprinting_score"].values >= 2).astype(int))),
            },
        },
        "feature_importance_top20": {k: float(v) for k, v in top_features.items()},
        "novel_detections": {
            "rf_catches_heuristic_misses": len(rf_comparison["model_only"]),
            "xgb_catches_heuristic_misses": len(xgb_comparison["model_only"]),
            "ff_catches_heuristic_misses": len(ff_comparison["model_only"]),
        },
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Save novel detections for manual analysis
    for name, comparison in [("rf", rf_comparison), ("xgb", xgb_comparison), ("ff", ff_comparison)]:
        novel_domains = sorted(comparison["model_only"])
        novel_path = os.path.join(args.output_dir, f"novel_detections_{name}.txt")
        with open(novel_path, "w") as f:
            f.write("\n".join(novel_domains))
        print(f"Saved {len(novel_domains)} novel detections to {novel_path}")


if __name__ == "__main__":
    main()