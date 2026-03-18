"""
Extract features from DuckDuckGo Tracker Radar domain JSON files.
Produces a tabular dataset suitable for ML classification.

Usage:
    python scripts/extract_features.py \
        --tracker-radar data/tracker-radar \
        --region US \
        --output data/features.parquet
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm


def load_api_weights(tracker_radar_path: str) -> dict:
    """Load the API fingerprint weights from Tracker Radar build data."""
    weights_path = os.path.join(
        tracker_radar_path,
        "build-data", "generated", "api_fingerprint_weights.json"
    )
    with open(weights_path) as f:
        return json.load(f)


def load_categories(tracker_radar_path: str) -> dict:
    """Load category assignments if available."""
    cat_path = os.path.join(
        tracker_radar_path,
        "build-data", "static", "categorized_trackers.csv"
    )
    categories = {}
    if os.path.exists(cat_path):
        df = pd.read_csv(cat_path)
        for _, row in df.iterrows():
            categories[row["domain"]] = row["category"]
    return categories


def get_all_api_names(tracker_radar_path: str, region: str, sample_size: int = 500) -> set:
    """Scan a sample of domain files to discover all API names used."""
    domains_dir = os.path.join(tracker_radar_path, "domains", region)
    api_names = set()
    files = list(Path(domains_dir).glob("*.json"))[:sample_size]
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            for resource in data.get("resources", []):
                api_names.update(resource.get("apis", {}).keys())
        except (json.JSONDecodeError, KeyError):
            continue
    return api_names


def extract_domain_features(domain_data: dict, api_weights: dict, all_apis: list) -> dict:
    """
    Extract features from a single domain JSON object.
    Features are aggregated across all resources in the domain.
    """
    features = {}
    domain = domain_data.get("domain", "")
    resources = domain_data.get("resources", [])

    # === Domain-level metadata ===
    features["domain"] = domain
    features["prevalence"] = domain_data.get("prevalence", 0)
    features["sites"] = domain_data.get("sites", 0)
    features["fingerprinting_score"] = domain_data.get("fingerprinting", 0)
    features["subdomain_count"] = len(domain_data.get("subdomains", []))
    features["has_cnames"] = 1 if domain_data.get("cnames", []) else 0
    features["resource_count"] = len(resources)

    # Owner info
    owner = domain_data.get("owner", {})
    features["has_owner"] = 1 if owner.get("name") else 0
    features["has_privacy_policy"] = 1 if owner.get("privacyPolicy") else 0

    # === Aggregate resource-level features ===
    total_cookie_prevalence = 0
    max_resource_fingerprinting = 0
    resource_types = set()
    script_count = 0
    image_count = 0
    fetch_count = 0

    # API call aggregation: sum across all resources
    api_call_counts = defaultdict(int)
    api_call_binary = defaultdict(int)  # 1 if any resource uses this API

    # First-party cookie features
    total_first_party_cookies = 0
    max_cookie_ttl = 0
    total_cookies_sent = 0

    for resource in resources:
        # Cookie behavior
        total_cookie_prevalence += resource.get("cookies", 0)

        # Per-resource fingerprinting
        res_fp = resource.get("fingerprinting", 0)
        max_resource_fingerprinting = max(max_resource_fingerprinting, res_fp)

        # Resource type
        res_type = resource.get("type", "Unknown")
        resource_types.add(res_type)
        if res_type == "Script":
            script_count += 1
        elif res_type == "Image":
            image_count += 1
        elif res_type == "Fetch":
            fetch_count += 1

        # API calls
        apis = resource.get("apis", {})
        for api_name, count in apis.items():
            api_call_counts[api_name] += count
            api_call_binary[api_name] = 1

        # First-party cookies
        fp_cookies = resource.get("firstPartyCookies", {})
        total_first_party_cookies += len(fp_cookies)
        for cookie_name, cookie_data in fp_cookies.items():
            if isinstance(cookie_data, dict):
                ttl = cookie_data.get("ttl", 0) or 0
                max_cookie_ttl = max(max_cookie_ttl, ttl)

        # Cookies sent to this resource
        fps_sent = resource.get("firstPartyCookiesSent", {})
        total_cookies_sent += len(fps_sent)

    # Store aggregate features
    features["total_cookie_prevalence"] = total_cookie_prevalence
    features["max_resource_fingerprinting"] = max_resource_fingerprinting
    features["resource_type_count"] = len(resource_types)
    features["script_count"] = script_count
    features["image_count"] = image_count
    features["fetch_count"] = fetch_count
    features["total_first_party_cookies"] = total_first_party_cookies
    features["max_cookie_ttl_seconds"] = max_cookie_ttl
    features["max_cookie_ttl_days"] = max_cookie_ttl / 86400 if max_cookie_ttl > 0 else 0
    features["total_cookies_sent"] = total_cookies_sent

    # === API features ===
    # Total distinct APIs used
    features["distinct_api_count"] = len(api_call_binary)

    # Total API calls across all resources
    features["total_api_calls"] = sum(api_call_counts.values())

    # Weighted fingerprint score (our own calculation using their weights)
    weighted_fp_score = 0
    for api_name, count in api_call_binary.items():
        weight = api_weights.get(api_name, 0)
        weighted_fp_score += weight
    features["weighted_fp_score"] = weighted_fp_score

    # Mean API weight (average suspiciousness of APIs used)
    if api_call_binary:
        weights_used = [api_weights.get(api, 0) for api in api_call_binary]
        features["mean_api_weight"] = np.mean(weights_used)
        features["max_api_weight"] = np.max(weights_used)
        features["median_api_weight"] = np.median(weights_used)
        features["std_api_weight"] = np.std(weights_used)
    else:
        features["mean_api_weight"] = 0
        features["max_api_weight"] = 0
        features["median_api_weight"] = 0
        features["std_api_weight"] = 0

    # === Per-API binary features (does this domain use API X?) ===
    for api_name in all_apis:
        safe_name = api_name.replace(".", "_").replace("(", "").replace(")", "").replace('"', '').replace(",", "").replace(" ", "_")
        features[f"api_{safe_name}"] = 1 if api_name in api_call_binary else 0

    # === Per-API call count features (how often is API X called?) ===
    # Only for top-weighted APIs to keep feature count manageable
    for api_name in all_apis:
        safe_name = api_name.replace(".", "_").replace("(", "").replace(")", "").replace('"', '').replace(",", "").replace(" ", "_")
        features[f"api_count_{safe_name}"] = api_call_counts.get(api_name, 0)

    # === Fingerprinting category features ===
    # Group APIs into functional categories
    canvas_apis = [a for a in api_call_binary if "Canvas" in a or "WebGL" in a]
    audio_apis = [a for a in api_call_binary if "Audio" in a or "OfflineAudio" in a]
    navigator_apis = [a for a in api_call_binary if "Navigator" in a]
    screen_apis = [a for a in api_call_binary if "Screen" in a or "screen" in a.lower()]
    device_apis = [a for a in api_call_binary if "Device" in a or "Gyroscope" in a or "Sensor" in a]
    storage_apis = [a for a in api_call_binary if "Storage" in a or "storage" in a or "localStorage" in a or "sessionStorage" in a or "indexedDB" in a or "Cookie" in a.split(".")[0]]
    timing_apis = [a for a in api_call_binary if "Performance" in a or "Date" in a or "Animation" in a]
    media_apis = [a for a in api_call_binary if "Media" in a]
    webrtc_apis = [a for a in api_call_binary if "RTC" in a]

    features["canvas_api_count"] = len(canvas_apis)
    features["audio_api_count"] = len(audio_apis)
    features["navigator_api_count"] = len(navigator_apis)
    features["screen_api_count"] = len(screen_apis)
    features["device_api_count"] = len(device_apis)
    features["storage_api_count"] = len(storage_apis)
    features["timing_api_count"] = len(timing_apis)
    features["media_api_count"] = len(media_apis)
    features["webrtc_api_count"] = len(webrtc_apis)

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract features from Tracker Radar data")
    parser.add_argument("--tracker-radar", required=True, help="Path to tracker-radar repo")
    parser.add_argument("--region", default="US", help="Region to process (default: US)")
    parser.add_argument("--output", default="data/features.parquet", help="Output parquet file")
    args = parser.parse_args()

    domains_dir = os.path.join(args.tracker_radar, "domains", args.region)
    domain_files = list(Path(domains_dir).glob("*.json"))
    print(f"Found {len(domain_files)} domain files in {args.region}")

    # Load API weights
    api_weights = load_api_weights(args.tracker_radar)
    print(f"Loaded {len(api_weights)} API weights")

    # Discover all API names used across domains
    print("Discovering API names...")
    all_apis_set = set()
    for f in tqdm(domain_files, desc="Scanning APIs"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            for resource in data.get("resources", []):
                all_apis_set.update(resource.get("apis", {}).keys())
        except (json.JSONDecodeError, KeyError):
            continue

    # Only keep APIs that appear in the weights file or are seen frequently
    # This prevents the feature space from exploding with rare APIs
    all_apis = sorted(all_apis_set & set(api_weights.keys()))
    print(f"Using {len(all_apis)} APIs (intersection with weights file)")

    # Extract features for all domains
    rows = []
    errors = 0
    for f in tqdm(domain_files, desc="Extracting features"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            features = extract_domain_features(data, api_weights, all_apis)
            rows.append(features)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            errors += 1
            continue

    print(f"Extracted features for {len(rows)} domains ({errors} errors)")

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Print summary stats
    print(f"\n=== Dataset Summary ===")
    print(f"Domains: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"\nFingerprinting score distribution:")
    print(df["fingerprinting_score"].value_counts().sort_index())
    print(f"\nPrevalence stats:")
    print(df["prevalence"].describe())
    print(f"\nDistinct API count stats:")
    print(df["distinct_api_count"].describe())
    print(f"\nWeighted FP score stats:")
    print(df["weighted_fp_score"].describe())

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
