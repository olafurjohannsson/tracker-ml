"""
Entity attribution experiments — try multiple approaches and compare.
Run once, see which works best, use that for the demo.

Usage:
    python entity-attribution/scripts/experiment.py \
        --data entity-attribution/data
"""

import sys
import os
import argparse
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score, f1_score
from collections import Counter


class EntityClassifier(nn.Module):
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


# Entity clusters for Option C
ENTITY_CLUSTERS = {
    # Ad tech / tracking
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
    # CDN / Infrastructure
    "Amazon Technologies, Inc.": "cdn_infra",
    "Akamai Technologies": "cdn_infra",
    "BunnyCDN": "cdn_infra",
    "Fastly, Inc.": "cdn_infra",
    "Imperva Inc.": "cdn_infra",
    # Hosting / Platform
    "Shopify Inc.": "platform",
    "GitHub, Inc.": "platform",
    "DataCamp Limited": "platform",
    "Alibaba Group": "platform",
    # Ad blocking / Privacy
    "Leven Labs, Inc. DBA Admiral": "ad_management",
}

# Tracking-focused entities for Option B
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


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, label_names,
                       hidden_dim=128, epochs=100, patience=20, batch_size=64, lr=1e-3,
                       experiment_name=""):
    """Train a model and return results dict."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Class weights
    class_counts = np.bincount(y_train, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)
    class_weights = len(y_train) / (num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test_s), torch.LongTensor(y_test))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    n_features = X_train.shape[1]
    model = EntityClassifier(n_features, hidden_dim, num_classes).to(device)
    class_weights = class_weights.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_acc = 0
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

        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Final eval with best model
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for bx, by in test_dl:
            logits = model(bx.to(device))
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(by.numpy())

    acc = accuracy_score(all_labels, all_preds)
    all_probs_np = np.array(all_probs)
    k = min(5, num_classes)
    top5 = top_k_accuracy_score(all_labels, all_probs_np, k=k, labels=range(num_classes))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "model": model,
        "scaler": scaler,
        "acc": acc,
        "top5": top5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "label_names": label_names,
        "num_classes": num_classes,
    }


def print_results(name, results):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Classes:     {results['num_classes']}")
    print(f"  Top-1 Acc:   {results['acc']:.3f}")
    print(f"  Top-5 Acc:   {results['top5']:.3f}")
    print(f"  Macro F1:    {results['macro_f1']:.3f}")
    print(f"  Weighted F1: {results['weighted_f1']:.3f}")
    print()
    print(classification_report(
        results['all_labels'], results['all_preds'],
        target_names=results['label_names'],
        zero_division=0,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="entity-attribution/data")
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    df = pd.read_parquet(os.path.join(args.data, "labeled.parquet"))
    with open(os.path.join(args.data, "entity_map.json")) as f:
        entity_info = json.load(f)

    entity_list = entity_info["entities"]
    entity_to_idx = entity_info["entity_to_idx"]
    feature_cols = get_feature_columns(df)
    print(f"Loaded {len(df):,} domains, {len(entity_list)} entities, {len(feature_cols)} features")

    # EXPERIMENT A: Downsample to 200 per entity
    print("\n>>> Preparing Experiment A: Downsample to 200 per entity")
    dfs = []
    for entity in df["entity_label"].unique():
        subset = df[df["entity_label"] == entity]
        if len(subset) > 200:
            subset = subset.sample(n=200, random_state=42)
        dfs.append(subset)
    df_a = pd.concat(dfs).reset_index(drop=True)

    # Recompute label mapping for this subset
    entities_a = sorted(df_a["entity_label"].unique())
    eidx_a = {e: i for i, e in enumerate(entities_a)}

    X_a = df_a[feature_cols].values.astype(np.float32)
    y_a = df_a["entity_label"].map(eidx_a).values.astype(np.int64)

    X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
        X_a, y_a, test_size=0.2, stratify=y_a, random_state=42
    )
    print(f"  Train: {len(X_tr_a):,}, Test: {len(X_te_a):,}, Classes: {len(entities_a)}")

    results_a = train_and_evaluate(
        X_tr_a, y_tr_a, X_te_a, y_te_a,
        num_classes=len(entities_a),
        label_names=entities_a,
        hidden_dim=args.hidden_dim,
    )
    print_results("EXPERIMENT A: Downsample (max 200 per entity)", results_a)

    # EXPERIMENT B: Tracking entities only
    print("\n>>> Preparing Experiment B: Tracking entities only")
    df_b = df[df["entity_label"].isin(TRACKING_ENTITIES)].copy()

    if len(df_b) > 0:
        entities_b = sorted(df_b["entity_label"].unique())
        eidx_b = {e: i for i, e in enumerate(entities_b)}

        X_b = df_b[feature_cols].values.astype(np.float32)
        y_b = df_b["entity_label"].map(eidx_b).values.astype(np.int64)

        # Need at least 2 per class for stratified split
        class_counts_b = Counter(y_b)
        valid_classes = {c for c, n in class_counts_b.items() if n >= 2}
        mask_b = np.array([y in valid_classes for y in y_b])
        X_b, y_b = X_b[mask_b], y_b[mask_b]

        # Remap labels to be contiguous
        unique_b = sorted(set(y_b))
        remap_b = {old: new for new, old in enumerate(unique_b)}
        y_b = np.array([remap_b[y] for y in y_b])
        entities_b = [entities_b[old] for old in unique_b]

        X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(
            X_b, y_b, test_size=0.2, stratify=y_b, random_state=42
        )
        print(f"  Train: {len(X_tr_b):,}, Test: {len(X_te_b):,}, Classes: {len(entities_b)}")

        results_b = train_and_evaluate(
            X_tr_b, y_tr_b, X_te_b, y_te_b,
            num_classes=len(entities_b),
            label_names=entities_b,
            hidden_dim=args.hidden_dim,
        )
        print_results("EXPERIMENT B: Tracking entities only", results_b)
    else:
        print("  No tracking entities found in labeled data!")
        results_b = None

    # EXPERIMENT C: Entity clusters
    print("\n>>> Preparing Experiment C: Entity clusters")
    df_c = df.copy()
    df_c["cluster"] = df_c["entity_label"].map(ENTITY_CLUSTERS).fillna("other")
    # Drop "other" — these are entities we didn't assign
    df_c = df_c[df_c["cluster"] != "other"]

    clusters = sorted(df_c["cluster"].unique())
    cidx = {c: i for i, c in enumerate(clusters)}

    X_c = df_c[feature_cols].values.astype(np.float32)
    y_c = df_c["cluster"].map(cidx).values.astype(np.int64)

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_c, y_c, test_size=0.2, stratify=y_c, random_state=42
    )
    print(f"  Train: {len(X_tr_c):,}, Test: {len(X_te_c):,}, Classes: {len(clusters)}")

    results_c = train_and_evaluate(
        X_tr_c, y_tr_c, X_te_c, y_te_c,
        num_classes=len(clusters),
        label_names=clusters,
        hidden_dim=64,  # fewer classes, smaller model
    )
    print_results("EXPERIMENT C: Entity clusters", results_c)

    # EXPERIMENT D: Downsample + bigger model
    print("\n>>> Preparing Experiment D: Downsample + hidden_dim=256")
    results_d = train_and_evaluate(
        X_tr_a, y_tr_a, X_te_a, y_te_a,
        num_classes=len(entities_a),
        label_names=entities_a,
        hidden_dim=256,
        epochs=150,
        patience=25,
    )
    print_results("EXPERIMENT D: Downsample + larger model", results_d)

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"{'Experiment':<45} {'Top-1':>6} {'Top-5':>6} {'F1-W':>6}")
    print("-"*60)
    print(f"{'A: Downsample 200':<45} {results_a['acc']:>6.3f} {results_a['top5']:>6.3f} {results_a['weighted_f1']:>6.3f}")
    if results_b:
        print(f"{'B: Tracking only':<45} {results_b['acc']:>6.3f} {results_b['top5']:>6.3f} {results_b['weighted_f1']:>6.3f}")
    print(f"{'C: Entity clusters':<45} {results_c['acc']:>6.3f} {results_c['top5']:>6.3f} {results_c['weighted_f1']:>6.3f}")
    print(f"{'D: Downsample + big model':<45} {results_d['acc']:>6.3f} {results_d['top5']:>6.3f} {results_d['weighted_f1']:>6.3f}")
    print(f"\n(Baseline: original model was Top-1=0.240, Top-5=0.671)")


if __name__ == "__main__":
    main()