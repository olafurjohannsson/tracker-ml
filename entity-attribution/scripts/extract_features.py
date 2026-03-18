"""
Extract features for entity attribution: predicting which entity
owns a domain based purely on behavioral patterns.

Key design choice: we EXCLUDE entity metadata features (has_owner,
has_privacy_policy) since entity identity is the prediction target.
The model must learn to identify ownership from behavior alone.

Usage:
    python entity-attribution/scripts/extract_features.py \
        --tracker-radar data/tracker-radar \
        --region US \
        --min-domains 10 \
        --output entity-attribution/data
"""

import sys
import os
import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared import TrackerRadarLoader


def extract_behavioral_features(domain_data: dict, api_weights: dict, all_apis: list) -> dict:
    """
    Extract behavioral features for a domain, deliberately excluding
    any entity/ownership metadata.
    """
    features = {}
    resources = domain_data.get("resources", [])

    # Domain identifiers (not used as features, kept for reference)
    features["domain"] = domain_data.get("domain", "")

    # === Behavioral metadata (no ownership info) ===
    features["prevalence"] = domain_data.get("prevalence", 0)
    features["sites"] = domain_data.get("sites", 0)
    features["fingerprinting_score"] = domain_data.get("fingerprinting", 0)
    features["subdomain_count"] = len(domain_data.get("subdomains", []))
    features["has_cnames"] = 1 if domain_data.get("cnames", []) else 0
    features["resource_count"] = len(resources)

    # === Resource type distribution ===
    type_counts = defaultdict(int)
    for r in resources:
        t = r.get("type", "Unknown")
        type_counts[t] += 1

    features["script_count"] = type_counts.get("Script", 0)
    features["image_count"] = type_counts.get("Image", 0)
    features["fetch_count"] = type_counts.get("Fetch", 0)
    features["xhr_count"] = type_counts.get("XHR", 0)
    features["stylesheet_count"] = type_counts.get("Stylesheet", 0)
    features["font_count"] = type_counts.get("Font", 0)
    features["document_count"] = type_counts.get("Document", 0)
    features["media_count"] = type_counts.get("Media", 0)

    # Resource type ratios (normalized)
    total_resources = max(len(resources), 1)
    features["script_ratio"] = features["script_count"] / total_resources
    features["image_ratio"] = features["image_count"] / total_resources

    # === Cookie behavior ===
    total_cookie_prevalence = 0
    total_first_party_cookies = 0
    max_cookie_ttl = 0
    total_cookies_sent = 0

    for r in resources:
        total_cookie_prevalence += r.get("cookies", 0)
        fp_cookies = r.get("firstPartyCookies", {})
        total_first_party_cookies += len(fp_cookies)
        for cookie_data in fp_cookies.values():
            if isinstance(cookie_data, dict):
                ttl = cookie_data.get("ttl", 0) or 0
                max_cookie_ttl = max(max_cookie_ttl, ttl)
        total_cookies_sent += len(r.get("firstPartyCookiesSent", {}))

    features["total_cookie_prevalence"] = total_cookie_prevalence
    features["total_first_party_cookies"] = total_first_party_cookies
    features["max_cookie_ttl_days"] = max_cookie_ttl / 86400 if max_cookie_ttl > 0 else 0
    features["total_cookies_sent"] = total_cookies_sent

    # === API usage patterns ===
    api_call_counts = defaultdict(int)
    api_call_binary = set()

    for r in resources:
        apis = r.get("apis", {})
        for api_name, count in apis.items():
            api_call_counts[api_name] += count
            api_call_binary.add(api_name)

    features["distinct_api_count"] = len(api_call_binary)
    features["total_api_calls"] = sum(api_call_counts.values())

    # API weight statistics
    if api_call_binary:
        weights_used = [api_weights.get(api, 0) for api in api_call_binary]
        features["mean_api_weight"] = np.mean(weights_used)
        features["max_api_weight"] = np.max(weights_used)
        features["median_api_weight"] = np.median(weights_used)
        features["std_api_weight"] = np.std(weights_used)
        features["weighted_fp_score"] = sum(weights_used)
    else:
        features["mean_api_weight"] = 0
        features["max_api_weight"] = 0
        features["median_api_weight"] = 0
        features["std_api_weight"] = 0
        features["weighted_fp_score"] = 0

    # API category counts
    canvas = [a for a in api_call_binary if "Canvas" in a or "WebGL" in a]
    audio = [a for a in api_call_binary if "Audio" in a or "OfflineAudio" in a]
    navigator = [a for a in api_call_binary if "Navigator" in a]
    screen = [a for a in api_call_binary if "Screen" in a or "screen" in a.lower()]
    storage = [a for a in api_call_binary if any(k in a for k in ["Storage", "storage", "localStorage", "sessionStorage", "indexedDB"])]
    timing = [a for a in api_call_binary if "Performance" in a or "Date" in a]

    features["canvas_api_count"] = len(canvas)
    features["audio_api_count"] = len(audio)
    features["navigator_api_count"] = len(navigator)
    features["screen_api_count"] = len(screen)
    features["storage_api_count"] = len(storage)
    features["timing_api_count"] = len(timing)

    # Per-API binary features
    for api_name in all_apis:
        safe = api_name.replace(".", "_").replace("(", "").replace(")", "").replace('"', '').replace(",", "").replace(" ", "_")
        features[f"api_{safe}"] = 1 if api_name in api_call_binary else 0

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-radar", required=True)
    parser.add_argument("--region", default="US")
    parser.add_argument("--min-domains", type=int, default=10,
                        help="Minimum domains per entity to include in training")
    parser.add_argument("--output", default="entity-attribution/data")
    args = parser.parse_args()

    # Load data
    loader = TrackerRadarLoader(args.tracker_radar, args.region)
    loader.load()

    all_apis = loader.get_all_apis()
    print(f"Using {len(all_apis)} APIs")

    # Find entities with enough domains in the dataset
    entity_domain_counts = {}
    for entity_name, domains in loader.domains_for.items():
        # Only count domains that are actually in our region's dataset
        in_dataset = [d for d in domains if d in loader.domains]
        if len(in_dataset) >= args.min_domains:
            entity_domain_counts[entity_name] = len(in_dataset)

    print(f"\nEntities with >= {args.min_domains} domains in dataset: {len(entity_domain_counts)}")
    print("Top 20:")
    for name, count in sorted(entity_domain_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count:4d}  {name}")

    # Extract features for all domains
    rows = []
    for domain_name, domain_data in tqdm(loader.iter_domains(), total=len(loader.domains), desc="Extracting features"):
        features = extract_behavioral_features(domain_data, loader.api_weights, all_apis)

        # Add entity label
        entity = loader.get_entity(domain_name)
        features["entity"] = entity if entity else "__unknown__"
        features["has_entity"] = 1 if entity else 0

        rows.append(features)

    df = pd.DataFrame(rows)

    # Split into labeled (known entity with enough samples) and unlabeled
    known_entities = set(entity_domain_counts.keys())
    df["entity_label"] = df["entity"].apply(lambda e: e if e in known_entities else "__other__")

    labeled = df[df["entity_label"] != "__other__"].copy()
    unlabeled = df[df["entity"] == "__unknown__"].copy()
    other_known = df[(df["entity"] != "__unknown__") & (df["entity_label"] == "__other__")].copy()

    print(f"\n=== Dataset Split ===")
    print(f"  Labeled (trainable entities):   {len(labeled):,}")
    print(f"  Unknown (no owner, inference):   {len(unlabeled):,}")
    print(f"  Other known (too few samples):   {len(other_known):,}")
    print(f"  Unique trainable entities:       {labeled['entity_label'].nunique()}")

    print(f"\n=== Entity distribution (labeled set) ===")
    print(labeled["entity_label"].value_counts().head(20))

    # Save
    os.makedirs(args.output, exist_ok=True)

    labeled.to_parquet(os.path.join(args.output, "labeled.parquet"), index=False)
    unlabeled.to_parquet(os.path.join(args.output, "unlabeled.parquet"), index=False)
    df.to_parquet(os.path.join(args.output, "all_features.parquet"), index=False)

    # Save entity mapping for the model
    entity_list = sorted(known_entities)
    entity_map = {name: idx for idx, name in enumerate(entity_list)}
    with open(os.path.join(args.output, "entity_map.json"), "w") as f:
        json.dump({"entities": entity_list, "entity_to_idx": entity_map}, f, indent=2)

    print(f"\nSaved to {args.output}/")
    print(f"  labeled.parquet:     {len(labeled):,} domains")
    print(f"  unlabeled.parquet:   {len(unlabeled):,} domains")
    print(f"  all_features.parquet: {len(df):,} domains")
    print(f"  entity_map.json:     {len(entity_list)} entities")


if __name__ == "__main__":
    main()