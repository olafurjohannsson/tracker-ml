"""
Sanity check all three models on known domains.
Runs inference in Python (same math as WASM) to verify predictions.

Usage:
    python site/scripts/test_models.py \
        --tracker-features data/features_us.parquet \
        --entity-data entity-attribution/data/all_features.parquet \
        --tracker-scaler tracker-classifier/models/scaler.joblib \
        --cluster-scaler entity-attribution/models/entity_cluster_classifier_scaler.json \
        --tracking-scaler entity-attribution/models/tracking_entity_classifier_scaler.json \
        --cluster-config entity-attribution/models/entity_cluster_classifier_config.json \
        --tracking-config entity-attribution/models/tracking_entity_classifier_config.json
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import joblib
import torch
from safetensors.torch import load_file


def load_ff_model(safetensors_path):
    """Load a feedforward model from safetensors, return weight/bias tuples."""
    tensors = load_file(safetensors_path)
    layers = []
    for i in range(1, 4):
        w = tensors[f"layer{i}.weight"].numpy()
        b = tensors[f"layer{i}.bias"].numpy()
        layers.append((w, b))
    return layers


def forward(layers, x):
    """Run feedforward inference: 3 linear layers with ReLU on first two."""
    h = x
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < 2:  # ReLU on layers 1 and 2, not output
            h = np.maximum(h, 0)
    # Softmax
    exp = np.exp(h - h.max())
    probs = exp / exp.sum()
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-features", default="data/features_us.parquet")
    parser.add_argument("--entity-data", default="entity-attribution/data/all_features.parquet")
    parser.add_argument("--tracker-scaler", default="tracker-classifier/models/scaler.joblib")
    parser.add_argument("--tracker-model", default="tracker-classifier/models/tracker_classifier.safetensors")
    parser.add_argument("--cluster-scaler", default="entity-attribution/models/entity_cluster_classifier_scaler.json")
    parser.add_argument("--cluster-model", default="entity-attribution/models/entity_cluster_classifier.safetensors")
    parser.add_argument("--cluster-config", default="entity-attribution/models/entity_cluster_classifier_config.json")
    parser.add_argument("--tracking-scaler", default="entity-attribution/models/tracking_entity_classifier_scaler.json")
    parser.add_argument("--tracking-model", default="entity-attribution/models/tracking_entity_classifier.safetensors")
    parser.add_argument("--tracking-config", default="entity-attribution/models/tracking_entity_classifier_config.json")
    args = parser.parse_args()

    # Load data
    tracker_df = pd.read_parquet(args.tracker_features).set_index("domain")
    entity_df = pd.read_parquet(args.entity_data).set_index("domain")

    # Load models
    tracker_layers = load_ff_model(args.tracker_model)
    cluster_layers = load_ff_model(args.cluster_model)
    tracking_layers = load_ff_model(args.tracking_model)

    # Load scalers
    tracker_scaler = joblib.load(args.tracker_scaler)

    with open(args.cluster_scaler) as f:
        cluster_scaler_data = json.load(f)
    with open(args.tracking_scaler) as f:
        tracking_scaler_data = json.load(f)

    # Load configs
    with open(args.cluster_config) as f:
        cluster_config = json.load(f)
    with open(args.tracking_config) as f:
        tracking_config = json.load(f)

    tracker_labels = ["non-tracking", "tracking"]
    cluster_labels = cluster_config["labels"]
    tracking_labels = tracking_config["labels"]

    # Feature columns
    tracker_exclude = {"domain", "fingerprinting_score", "label", "label_source"}
    tracker_feature_cols = [c for c in tracker_df.columns if c not in tracker_exclude]
    entity_feature_cols = cluster_scaler_data["feature_names"]

    cluster_mean = np.array(cluster_scaler_data["mean"])
    cluster_scale = np.array(cluster_scaler_data["scale"])
    tracking_mean = np.array(tracking_scaler_data["mean"])
    tracking_scale = np.array(tracking_scaler_data["scale"])

    # Test domains — mix of known trackers, CDNs, unknowns
    test_domains = [
        # Known trackers
        "facebook.com",
        "doubleclick.net",
        "google-analytics.com",
        "adnxs.com",
        "demdex.net",
        # Known CDN / functional
        "cloudflare.com",
        "akamaihd.net",
        "jsdelivr.net",
        "googleapis.com",
        "gstatic.com",
        # Google properties (mixed)
        "googlesyndication.com",
        "googletagmanager.com",
        "googleadservices.com",
        # Microsoft
        "bing.com",
        "clarity.ms",
        # Other
        "shopify.com",
        "hubspot.com",
        "yahoo.com",
        "tiktok.com",
        "instagram.com",
    ]

    print(f"{'Domain':<30} {'Tracker':<14} {'Cluster':<20} {'Entity':<30} {'FP':>3}")
    print("─" * 100)

    for domain in test_domains:
        # Tracker classification
        tracker_pred = "?"
        tracker_conf = 0
        if domain in tracker_df.index:
            row = tracker_df.loc[domain]
            feats = np.array([row.get(c, 0) for c in tracker_feature_cols], dtype=np.float32)
            feats = np.nan_to_num(feats)
            scaled = tracker_scaler.transform(feats.reshape(1, -1)).flatten()
            probs = forward(tracker_layers, scaled)
            label = np.argmax(probs)
            tracker_pred = tracker_labels[label]
            tracker_conf = probs[label]
            fp_score = int(row.get("fingerprinting_score", 0))
        else:
            fp_score = -1

        # Cluster classification
        cluster_pred = "?"
        cluster_conf = 0
        if domain in entity_df.index:
            row = entity_df.loc[domain]
            feats = np.array([row.get(c, 0) for c in entity_feature_cols], dtype=np.float32)
            feats = np.nan_to_num(feats)
            scaled = (feats - cluster_mean) / np.where(cluster_scale == 0, 1, cluster_scale)
            probs = forward(cluster_layers, scaled)
            label = np.argmax(probs)
            cluster_pred = cluster_labels[label]
            cluster_conf = probs[label]

        # Tracking entity classification
        entity_pred = "?"
        entity_conf = 0
        if domain in entity_df.index:
            row = entity_df.loc[domain]
            feats = np.array([row.get(c, 0) for c in entity_feature_cols], dtype=np.float32)
            feats = np.nan_to_num(feats)
            scaled = (feats - tracking_mean) / np.where(tracking_scale == 0, 1, tracking_scale)
            probs = forward(tracking_layers, scaled)
            label = np.argmax(probs)
            entity_pred = tracking_labels[label]
            entity_conf = probs[label]

        print(
            f"{domain:<30} "
            f"{tracker_pred:<10} {tracker_conf:>3.0%}  "
            f"{cluster_pred:<16} {cluster_conf:>3.0%}  "
            f"{entity_pred:<26} {entity_conf:>3.0%}  "
            f"{fp_score:>3}"
        )

    # Summary of unknowns
    print(f"\n\n{'='*60}")
    print("  UNKNOWN DOMAIN PREDICTIONS")
    print(f"{'='*60}")
    print("  Domains with no entity in Tracker Radar, showing model guesses:\n")

    unknowns = entity_df[entity_df.get("entity", "__unknown__") == "__unknown__"]
    if "entity" not in entity_df.columns:
        unknowns = entity_df[entity_df.get("has_entity", 0) == 0]

    # Take a sample of unknowns that are in the tracker dataset too
    sample_domains = []
    for domain in unknowns.index:
        if domain in tracker_df.index and len(sample_domains) < 15:
            row = tracker_df.loc[domain]
            if row.get("sites", 0) > 100:  # Non-trivial domains
                sample_domains.append(domain)

    print(f"{'Domain':<30} {'Tracker':<14} {'Cluster':<20} {'Entity':<30} {'Sites':>8}")
    print("─" * 105)

    for domain in sample_domains:
        row_t = tracker_df.loc[domain]
        sites = int(row_t.get("sites", 0))

        # Tracker
        feats = np.array([row_t.get(c, 0) for c in tracker_feature_cols], dtype=np.float32)
        feats = np.nan_to_num(feats)
        scaled = tracker_scaler.transform(feats.reshape(1, -1)).flatten()
        probs = forward(tracker_layers, scaled)
        tracker_label = np.argmax(probs)
        tracker_pred = tracker_labels[tracker_label]
        tracker_conf = probs[tracker_label]

        # Cluster
        row_e = entity_df.loc[domain]
        feats = np.array([row_e.get(c, 0) for c in entity_feature_cols], dtype=np.float32)
        feats = np.nan_to_num(feats)
        scaled = (feats - cluster_mean) / np.where(cluster_scale == 0, 1, cluster_scale)
        probs = forward(cluster_layers, scaled)
        cluster_label = np.argmax(probs)
        cluster_pred = cluster_labels[cluster_label]
        cluster_conf = probs[cluster_label]

        # Entity
        feats = np.array([row_e.get(c, 0) for c in entity_feature_cols], dtype=np.float32)
        feats = np.nan_to_num(feats)
        scaled = (feats - tracking_mean) / np.where(tracking_scale == 0, 1, tracking_scale)
        probs = forward(tracking_layers, scaled)
        entity_label = np.argmax(probs)
        entity_pred = tracking_labels[entity_label]
        entity_conf = probs[entity_label]

        print(
            f"{domain:<30} "
            f"{tracker_pred:<10} {tracker_conf:>3.0%}  "
            f"{cluster_pred:<16} {cluster_conf:>3.0%}  "
            f"{entity_pred:<26} {entity_conf:>3.0%}  "
            f"{sites:>8,}"
        )


if __name__ == "__main__":
    main()