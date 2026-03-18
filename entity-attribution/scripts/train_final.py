"""
Train and export the two entity attribution models:
  1. Entity cluster classifier (4 classes: ad_tech, cdn_infra, platform, ad_management)
  2. Tracking entity classifier (13 classes: Google, Microsoft, Adobe, etc.)

Both export as safetensors for Kjarni WASM deployment.

Usage:
    python entity-attribution/scripts/train_final.py \
        --data entity-attribution/data \
        --output entity-attribution/models
"""

import sys
import os
import argparse
import json
from collections import Counter

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score, f1_score
from safetensors.torch import save_file


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


ENTITY_CLUSTERS = {
    "Google LLC": "ad_tech",
    "Microsoft Corporation": "ad_tech",
    "Adobe Inc.": "ad_tech",
    "Oracle Corporation": "ad_tech",
    "Conversant LLC": "ad_tech",
    "ByteDance Ltd.": "ad_tech",
    "Yahoo Inc.": "ad_tech",
    "Yandex LLC": "ad_tech",
    "HubSpot, Inc.": "ad_tech",
    "Impact": "ad_tech",
    "Comcast Corporation": "ad_tech",
    "Salesforce.com, Inc.": "ad_tech",
    "Amazon Technologies, Inc.": "cdn_infra",
    "Akamai Technologies": "cdn_infra",
    "BunnyCDN": "cdn_infra",
    "Fastly, Inc.": "cdn_infra",
    "Imperva Inc.": "cdn_infra",
    "Shopify Inc.": "platform",
    "GitHub, Inc.": "platform",
    "DataCamp Limited": "platform",
    "Alibaba Group": "platform",
    "Leven Labs, Inc. DBA Admiral": "ad_management",
}

TRACKING_ENTITIES = {
    "Google LLC", "Microsoft Corporation", "Adobe Inc.",
    "Oracle Corporation", "Conversant LLC", "ByteDance Ltd.",
    "Yahoo Inc.", "Yandex LLC", "HubSpot, Inc.", "Impact",
    "Comcast Corporation", "Salesforce.com, Inc.",
    "Leven Labs, Inc. DBA Admiral",
}


def get_feature_columns(df):
    exclude = {"domain", "entity", "has_entity", "entity_label"}
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]


def train_model(X_train, y_train, X_test, y_test, num_classes,
                hidden_dim=128, epochs=100, patience=20, batch_size=64, lr=1e-3):
    """Train and return best model + scaler."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    class_counts = np.bincount(y_train, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = len(y_train) / (num_classes * class_counts)
    class_weights_t = torch.FloatTensor(class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test_s), torch.LongTensor(y_test))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = Classifier(X_train.shape[1], hidden_dim, num_classes).to(device)
    class_weights_t = class_weights_t.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    best_f1 = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for bx, by in test_dl:
                logits = model(bx.to(device))
                probs = torch.softmax(logits, dim=1)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(by.numpy())

        wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        if (epoch + 1) % 10 == 0:
            acc = accuracy_score(all_labels, all_preds)
            print(f"    Epoch {epoch+1:3d} | Acc: {acc:.3f} | Weighted F1: {wf1:.3f}")

        if wf1 > best_f1:
            best_f1 = wf1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Final eval
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for bx, by in test_dl:
            logits = model(bx.to(device))
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(by.numpy())

    acc = accuracy_score(all_labels, all_preds)
    k = min(5, num_classes)
    top5 = top_k_accuracy_score(all_labels, np.array(all_probs), k=k, labels=range(num_classes))
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return model, scaler, {
        "accuracy": float(acc),
        "top5_accuracy": float(top5),
        "weighted_f1": float(wf1),
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def save_model(model, scaler, feature_cols, label_names, metrics, output_dir, name):
    """Save model as safetensors + scaler + config."""
    os.makedirs(output_dir, exist_ok=True)

    # Safetensors
    tensors = {k: v.cpu() for k, v in model.state_dict().items()}
    st_path = os.path.join(output_dir, f"{name}.safetensors")
    save_file(tensors, st_path)
    size_kb = os.path.getsize(st_path) / 1024

    # Scaler
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": feature_cols,
    }
    with open(os.path.join(output_dir, f"{name}_scaler.json"), "w") as f:
        json.dump(scaler_data, f)

    # Config (for Kjarni WASM loading)
    hidden_dim = model.net[0].out_features
    num_classes = model.net[-1].out_features
    config = {
        "input_dim": len(feature_cols),
        "hidden_dim": hidden_dim,
        "output_dim": num_classes,
        "labels": label_names,
        "accuracy": metrics["accuracy"],
        "top5_accuracy": metrics["top5_accuracy"],
        "weighted_f1": metrics["weighted_f1"],
    }
    with open(os.path.join(output_dir, f"{name}_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved {name}.safetensors ({size_kb:.1f} KB)")
    print(f"  Saved {name}_scaler.json")
    print(f"  Saved {name}_config.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="entity-attribution/data")
    parser.add_argument("--output", default="entity-attribution/models")
    args = parser.parse_args()

    df = pd.read_parquet(os.path.join(args.data, "labeled.parquet"))
    feature_cols = get_feature_columns(df)
    print(f"Loaded {len(df):,} domains, {len(feature_cols)} features\n")

    # MODEL 1: Entity clusters (4 classes)
    print("=" * 60)
    print("  Training: Entity Cluster Classifier")
    print("=" * 60)

    df_c = df.copy()
    df_c["cluster"] = df_c["entity_label"].map(ENTITY_CLUSTERS).fillna("other")
    df_c = df_c[df_c["cluster"] != "other"]

    cluster_names = sorted(df_c["cluster"].unique())
    cidx = {c: i for i, c in enumerate(cluster_names)}

    X_c = df_c[feature_cols].values.astype(np.float32)
    y_c = df_c["cluster"].map(cidx).values.astype(np.int64)

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_c, y_c, test_size=0.2, stratify=y_c, random_state=42
    )
    print(f"  Train: {len(X_tr_c):,}, Test: {len(X_te_c):,}, Classes: {len(cluster_names)}")
    print(f"  Labels: {cluster_names}\n")

    model_c, scaler_c, metrics_c = train_model(
        X_tr_c, y_tr_c, X_te_c, y_te_c,
        num_classes=len(cluster_names),
        hidden_dim=64,
    )

    print(f"\n  Results: Acc={metrics_c['accuracy']:.3f}, F1={metrics_c['weighted_f1']:.3f}")
    print(classification_report(
        metrics_c["all_labels"], metrics_c["all_preds"],
        target_names=cluster_names, zero_division=0
    ))

    save_model(model_c, scaler_c, feature_cols, cluster_names, metrics_c,
               args.output, "entity_cluster_classifier")

    # MODEL 2: Tracking entity attribution (13 classes)
    print("\n" + "=" * 60)
    print("  Training: Tracking Entity Classifier")
    print("=" * 60)

    df_t = df[df["entity_label"].isin(TRACKING_ENTITIES)].copy()

    entity_names = sorted(df_t["entity_label"].unique())
    eidx = {e: i for i, e in enumerate(entity_names)}

    X_t = df_t[feature_cols].values.astype(np.float32)
    y_t = df_t["entity_label"].map(eidx).values.astype(np.int64)

    # Filter classes with < 2 samples
    class_counts = Counter(y_t)
    valid = {c for c, n in class_counts.items() if n >= 2}
    mask = np.array([y in valid for y in y_t])
    X_t, y_t = X_t[mask], y_t[mask]

    # Remap to contiguous
    unique_t = sorted(set(y_t))
    remap = {old: new for new, old in enumerate(unique_t)}
    y_t = np.array([remap[y] for y in y_t])
    entity_names = [entity_names[old] for old in unique_t]

    X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(
        X_t, y_t, test_size=0.2, stratify=y_t, random_state=42
    )
    print(f"  Train: {len(X_tr_t):,}, Test: {len(X_te_t):,}, Classes: {len(entity_names)}")
    print(f"  Labels: {entity_names}\n")

    model_t, scaler_t, metrics_t = train_model(
        X_tr_t, y_tr_t, X_te_t, y_te_t,
        num_classes=len(entity_names),
        hidden_dim=128,
    )

    print(f"\n  Results: Acc={metrics_t['accuracy']:.3f}, F1={metrics_t['weighted_f1']:.3f}")
    print(classification_report(
        metrics_t["all_labels"], metrics_t["all_preds"],
        target_names=entity_names, zero_division=0
    ))

    save_model(model_t, scaler_t, feature_cols, entity_names, metrics_t,
               args.output, "tracking_entity_classifier")

    # Summary
    print("\n" + "=" * 60)
    print("  EXPORT SUMMARY")
    print("=" * 60)
    print(f"  Entity cluster classifier:")
    print(f"    Classes: {cluster_names}")
    print(f"    Accuracy: {metrics_c['accuracy']:.3f}, Weighted F1: {metrics_c['weighted_f1']:.3f}")
    print(f"  Tracking entity classifier:")
    print(f"    Classes: {len(entity_names)} entities")
    print(f"    Accuracy: {metrics_t['accuracy']:.3f}, Weighted F1: {metrics_t['weighted_f1']:.3f}")
    print(f"\n  All models saved to {args.output}/")


if __name__ == "__main__":
    main()