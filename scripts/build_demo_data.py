"""
Build the combined demo data for the force graph visualization.
Merges graph structure with pre-computed features for all three models.

Each node gets:
  - Graph metadata (entity, categories, prevalence, etc.)
  - Feature vector for tracker classifier (295-dim)
  - Feature vector for entity models (164-dim, shared by cluster + tracking)

Scaler params are included so JS can normalize before inference.

Usage:
    python site/scripts/build_demo_data.py \
        --tracker-radar data/tracker-radar \
        --tracker-features data/features.parquet \
        --entity-data entity-attribution/data/all_features.parquet \
        --graph site/data/graph.json \
        --tracker-scaler tracker-classifier/models/scaler.joblib \
        --cluster-scaler entity-attribution/models/entity_cluster_classifier_scaler.json \
        --tracking-scaler entity-attribution/models/tracking_entity_classifier_scaler.json \
        --output site/data/demo_data.json
"""

import sys
import os
import argparse
import json

import pandas as pd
import numpy as np
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-features", required=True, help="Parquet from tracker-classifier extraction")
    parser.add_argument("--entity-data", required=True, help="Parquet from entity-attribution extraction")
    parser.add_argument("--graph", required=True, help="graph.json from build_graph.py")
    parser.add_argument("--tracker-scaler", required=True, help="scaler.joblib from tracker-classifier")
    parser.add_argument("--cluster-scaler", required=True, help="scaler JSON from entity cluster model")
    parser.add_argument("--tracking-scaler", required=True, help="scaler JSON from tracking entity model")
    parser.add_argument("--cluster-config", required=True, help="config JSON from entity cluster model")
    parser.add_argument("--tracking-config", required=True, help="config JSON from tracking entity model")
    parser.add_argument("--output", default="site/data/demo_data.json")
    args = parser.parse_args()

    # Load graph
    with open(args.graph) as f:
        graph = json.load(f)
    graph_domains = {n["id"] for n in graph["nodes"]}
    print(f"Graph has {len(graph_domains)} nodes")

    # Load tracker classifier features
    tracker_df = pd.read_parquet(args.tracker_features)
    tracker_df = tracker_df.set_index("domain")
    print(f"Tracker features: {len(tracker_df)} domains, {len(tracker_df.columns)} columns")

    # Load entity attribution features
    entity_df = pd.read_parquet(args.entity_data)
    entity_df = entity_df.set_index("domain")
    print(f"Entity features: {len(entity_df)} domains, {len(entity_df.columns)} columns")

    # Load scalers
    tracker_scaler = joblib.load(args.tracker_scaler)
    with open(args.cluster_scaler) as f:
        cluster_scaler = json.load(f)
    with open(args.tracking_scaler) as f:
        tracking_scaler = json.load(f)

    # Load configs for label names
    with open(args.cluster_config) as f:
        cluster_config = json.load(f)
    with open(args.tracking_config) as f:
        tracking_config = json.load(f)

    # Get feature column names for each model
    # Tracker classifier: exclude non-feature columns
    tracker_exclude = {"domain", "fingerprinting_score", "label", "label_source"}
    tracker_feature_cols = [c for c in tracker_df.columns if c not in tracker_exclude]

    # Entity models: use the feature names from the scaler
    entity_feature_cols = cluster_scaler["feature_names"]

    print(f"Tracker features: {len(tracker_feature_cols)}")
    print(f"Entity features: {len(entity_feature_cols)}")

    # Build scaler data for JS
    scalers = {
        "tracker": {
            "mean": tracker_scaler.mean_.tolist(),
            "scale": tracker_scaler.scale_.tolist(),
            "features": tracker_feature_cols,
        },
        "cluster": {
            "mean": cluster_scaler["mean"],
            "scale": cluster_scaler["scale"],
            "features": entity_feature_cols,
        },
        "tracking": {
            "mean": tracking_scaler["mean"],
            "scale": tracking_scaler["scale"],
            "features": entity_feature_cols,
        },
    }

    # Attach features to each node
    nodes_with_features = 0
    for node in graph["nodes"]:
        domain = node["id"]

        # Tracker classifier features
        if domain in tracker_df.index:
            row = tracker_df.loc[domain]
            # Store the raw feature values in order
            tracker_feats = []
            for col in tracker_feature_cols:
                val = row.get(col, 0)
                tracker_feats.append(0.0 if pd.isna(val) else float(val))
            node["tracker_features"] = tracker_feats

            # Also store fingerprinting score for display
            node["fp_score"] = int(row.get("fingerprinting_score", 0))
        else:
            node["tracker_features"] = None

        # Entity model features
        if domain in entity_df.index:
            row = entity_df.loc[domain]
            entity_feats = []
            for col in entity_feature_cols:
                val = row.get(col, 0)
                entity_feats.append(0.0 if pd.isna(val) else float(val))
            node["entity_features"] = entity_feats
            nodes_with_features += 1
        else:
            node["entity_features"] = None

    print(f"Nodes with features: {nodes_with_features}/{len(graph['nodes'])}")

    # Build output
    demo_data = {
        "graph": graph,
        "scalers": scalers,
        "models": {
            "tracker": {
                "labels": ["non-tracking", "tracking"],
                "feature_count": len(tracker_feature_cols),
            },
            "cluster": {
                "labels": cluster_config["labels"],
                "feature_count": len(entity_feature_cols),
            },
            "tracking": {
                "labels": tracking_config["labels"],
                "feature_count": len(entity_feature_cols),
            },
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(demo_data, f)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()