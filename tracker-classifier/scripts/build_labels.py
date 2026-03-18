"""
Build binary classification labels (tracking vs non-tracking) from
multiple independent sources:
  1. Tracker Radar categories
  2. EasyPrivacy list
  3. Disconnect services list

The key insight: these labels are INDEPENDENT of the fingerprinting
heuristic score, so we can train a model and then compare its
predictions against the heuristic.

Usage:
    python scripts/build_labels.py \
        --features data/features_us.parquet \
        --tracker-radar data/tracker-radar \
        --output data/dataset_us.parquet
"""

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


# Categories from Tracker Radar that indicate tracking intent
TRACKING_CATEGORIES = {
    "Ad Motivated Tracking",
    "Advertising",
    "Analytics",
    "Audience Measurement",
    "Session Replay",
    "Third-Party Analytics",
    "Action Pixels",
    "Ad Fraud",
    "Obscure Ownership",
    "Social - Share",  # often used for cross-site tracking
    "Badge",  # social badges that track
    "Federated Login",  # can be used for cross-site identification
    "SSO",
}

# Categories that indicate non-tracking / functional purpose
FUNCTIONAL_CATEGORIES = {
    "CDN",
    "Embedded Content",
    "Online Payment",
    "Hosted Libraries",
    "Consent Management",
    "Tag Manager",  # debatable, but DDG doesn't block GTM itself
}

# Known CDN / functional domains (high confidence non-tracking)
KNOWN_FUNCTIONAL_DOMAINS = {
    "cdnjs.cloudflare.com", "cdn.jsdelivr.net", "unpkg.com",
    "fonts.googleapis.com", "fonts.gstatic.com", "ajax.googleapis.com",
    "code.jquery.com", "stackpath.bootstrapcdn.com",
    "maxcdn.bootstrapcdn.com", "use.fontawesome.com",
}


def load_tracker_radar_categories(tracker_radar_path: str) -> dict:
    """
    Load category assignments from Tracker Radar's static build data.
    Returns {domain: [list of categories]}
    """
    categories = {}

    # Check for categorized_trackers in build-data
    cat_dir = os.path.join(tracker_radar_path, "build-data", "static", "categories")
    if os.path.isdir(cat_dir):
        for cat_file in Path(cat_dir).glob("*.json"):
            try:
                with open(cat_file) as f:
                    data = json.load(f)
                category_name = cat_file.stem
                # Format varies — could be list of domains or dict
                if isinstance(data, list):
                    for domain in data:
                        if domain not in categories:
                            categories[domain] = []
                        categories[domain].append(category_name)
                elif isinstance(data, dict):
                    for domain in data.keys():
                        if domain not in categories:
                            categories[domain] = []
                        categories[domain].append(category_name)
            except (json.JSONDecodeError, KeyError):
                continue

    # Also check domain files for embedded categories
    # (some versions have categories in the domain JSON itself)
    return categories


def load_domain_categories_from_files(tracker_radar_path: str, region: str) -> dict:
    """
    Load categories directly from domain JSON files.
    """
    categories = {}
    domains_dir = os.path.join(tracker_radar_path, "domains", region)
    for f in Path(domains_dir).glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            domain = data.get("domain", "")
            cats = data.get("categories", [])
            if cats:
                categories[domain] = cats
        except (json.JSONDecodeError, KeyError):
            continue
    return categories


def download_easyprivacy() -> set:
    """
    Download EasyPrivacy list and extract domains.
    Returns set of domains known to be tracking.
    """
    url = "https://easylist.to/easylist/easyprivacy.txt"
    print(f"Downloading EasyPrivacy list...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TrackerML/1.0"})
        response = urllib.request.urlopen(req, timeout=30)
        content = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  Warning: Could not download EasyPrivacy: {e}")
        return set()

    domains = set()
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("!") or line.startswith("[") or not line:
            continue
        # Extract domain-like patterns
        # Match ||domain.com^ style rules
        match = re.match(r"\|\|([a-zA-Z0-9\-\.]+)\^", line)
        if match:
            domain = match.group(1).lower()
            # Skip overly broad patterns
            if "." in domain and len(domain) > 4:
                domains.add(domain)

    print(f"  Extracted {len(domains)} domains from EasyPrivacy")
    return domains


def download_disconnect() -> set:
    """
    Download Disconnect tracking list and extract domains.
    Returns set of domains known to be tracking.
    """
    url = "https://raw.githubusercontent.com/nicedoc/nicedoc.io/master/test/disconnect/services.json"
    print(f"Downloading Disconnect list...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TrackerML/1.0"})
        response = urllib.request.urlopen(req, timeout=30)
        data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"  Warning: Could not download Disconnect: {e}")
        # Try alternative URL
        try:
            alt_url = "https://raw.githubusercontent.com/nicedoc/nicedoc.io/master/test/disconnect/services.json"
            req = urllib.request.Request(alt_url, headers={"User-Agent": "TrackerML/1.0"})
            response = urllib.request.urlopen(req, timeout=30)
            data = json.loads(response.read().decode("utf-8"))
        except Exception as e2:
            print(f"  Warning: Could not download Disconnect (alt): {e2}")
            return set()

    domains = set()
    categories = data.get("categories", {})
    for category_name, entries in categories.items():
        for entry in entries:
            if isinstance(entry, dict):
                for company, company_data in entry.items():
                    if isinstance(company_data, dict):
                        for url, domain_list in company_data.items():
                            if isinstance(domain_list, list):
                                for d in domain_list:
                                    if isinstance(d, str) and "." in d:
                                        domains.add(d.lower())
                            elif isinstance(domain_list, str) and "." in domain_list:
                                domains.add(domain_list.lower())

    print(f"  Extracted {len(domains)} domains from Disconnect")
    return domains


def assign_labels(
    df: pd.DataFrame,
    tr_categories: dict,
    easyprivacy_domains: set,
    disconnect_domains: set,
) -> pd.DataFrame:
    """
    Assign binary labels based on multiple sources.

    Label strategy:
      1 (tracking) if:
        - Domain has a tracking category in Tracker Radar, OR
        - Domain is in EasyPrivacy list, OR
        - Domain is in Disconnect list
      0 (non-tracking) if:
        - Domain has ONLY functional categories, OR
        - Domain has no categories and is NOT in any tracking list
          AND has low fingerprinting signals

    -1 (ambiguous, excluded from training) if:
        - Conflicting signals between sources
    """
    labels = []
    label_sources = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Assigning labels"):
        domain = row["domain"]
        sources = []

        # Check Tracker Radar categories
        cats = tr_categories.get(domain, [])
        has_tracking_cat = bool(set(cats) & TRACKING_CATEGORIES) if cats else False
        has_functional_cat = bool(set(cats) & FUNCTIONAL_CATEGORIES) if cats else False
        has_only_functional = has_functional_cat and not has_tracking_cat

        # Check external lists
        in_easyprivacy = domain in easyprivacy_domains
        # Also check if any parent domain matches
        parts = domain.split(".")
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in easyprivacy_domains:
                in_easyprivacy = True
                break

        in_disconnect = domain in disconnect_domains
        for i in range(len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in disconnect_domains:
                in_disconnect = True
                break

        # Known functional
        is_known_functional = domain in KNOWN_FUNCTIONAL_DOMAINS

        # Determine label
        if is_known_functional:
            label = 0
            sources.append("known_functional")
        elif has_tracking_cat:
            label = 1
            sources.append("tr_category")
            if in_easyprivacy:
                sources.append("easyprivacy")
            if in_disconnect:
                sources.append("disconnect")
        elif in_easyprivacy or in_disconnect:
            label = 1
            if in_easyprivacy:
                sources.append("easyprivacy")
            if in_disconnect:
                sources.append("disconnect")
        elif has_only_functional:
            label = 0
            sources.append("tr_functional")
        elif not cats and not in_easyprivacy and not in_disconnect:
            # No category info, not in any list
            # Use a conservative heuristic: if no APIs used and low
            # cookie prevalence, likely non-tracking
            if row["distinct_api_count"] == 0 and row["total_cookie_prevalence"] < 0.001:
                label = 0
                sources.append("inferred_benign")
            else:
                # Ambiguous — could be tracking or functional
                label = -1
                sources.append("ambiguous")
        else:
            label = -1
            sources.append("ambiguous")

        labels.append(label)
        label_sources.append("|".join(sources))

    df = df.copy()
    df["label"] = labels
    df["label_source"] = label_sources
    return df


def main():
    parser = argparse.ArgumentParser(description="Build labels for tracker classification")
    parser.add_argument("--features", required=True, help="Path to features parquet")
    parser.add_argument("--tracker-radar", required=True, help="Path to tracker-radar repo")
    parser.add_argument("--region", default="US", help="Region")
    parser.add_argument("--output", default="data/dataset_us.parquet", help="Output path")
    args = parser.parse_args()

    # Load features
    df = pd.read_parquet(args.features)
    print(f"Loaded {len(df)} domains with {len(df.columns)} features")

    # Load category data
    print("\nLoading Tracker Radar categories...")
    tr_categories = load_tracker_radar_categories(args.tracker_radar)
    file_categories = load_domain_categories_from_files(args.tracker_radar, args.region)
    # Merge
    for domain, cats in file_categories.items():
        if domain not in tr_categories:
            tr_categories[domain] = cats
        else:
            tr_categories[domain] = list(set(tr_categories[domain] + cats))
    print(f"  Categories for {len(tr_categories)} domains")

    # Download external lists
    easyprivacy_domains = download_easyprivacy()
    disconnect_domains = download_disconnect()

    # Assign labels
    print("\nAssigning labels...")
    df = assign_labels(df, tr_categories, easyprivacy_domains, disconnect_domains)

    # Report
    print(f"\n=== Label Distribution ===")
    print(df["label"].value_counts().sort_index())
    print(f"\nLabel sources:")
    # Count individual sources
    all_sources = []
    for s in df["label_source"]:
        all_sources.extend(s.split("|"))
    source_counts = pd.Series(all_sources).value_counts()
    print(source_counts)

    # Show label vs fingerprinting score crosstab
    print(f"\n=== Label vs Fingerprinting Heuristic Score ===")
    labeled_df = df[df["label"] >= 0]
    ct = pd.crosstab(labeled_df["label"], labeled_df["fingerprinting_score"], margins=True)
    print(ct)

    # Agreement analysis
    print(f"\n=== Agreement Analysis ===")
    # How often does the heuristic (score >= 2 = tracking) agree with our labels?
    heuristic_tracking = labeled_df["fingerprinting_score"] >= 2
    our_tracking = labeled_df["label"] == 1
    agreement = (heuristic_tracking == our_tracking).mean()
    print(f"Agreement between heuristic (score>=2) and our labels: {agreement:.1%}")

    # Cases where they disagree — this is the interesting part
    disagree_mask = heuristic_tracking != our_tracking
    disagree_df = labeled_df[disagree_mask]
    print(f"Disagreements: {len(disagree_df)} domains ({len(disagree_df)/len(labeled_df):.1%})")

    # Our label=tracking but heuristic=low
    our_yes_heuristic_no = labeled_df[(labeled_df["label"] == 1) & (labeled_df["fingerprinting_score"] < 2)]
    print(f"  Labeled tracking, heuristic says benign: {len(our_yes_heuristic_no)}")

    # Our label=benign but heuristic=high
    our_no_heuristic_yes = labeled_df[(labeled_df["label"] == 0) & (labeled_df["fingerprinting_score"] >= 2)]
    print(f"  Labeled benign, heuristic says tracking: {len(our_no_heuristic_yes)}")

    # Save
    df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # Also save train-ready version (excluding ambiguous)
    train_df = df[df["label"] >= 0].copy()
    train_path = args.output.replace(".parquet", "_train.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"Saved training set ({len(train_df)} domains) to {train_path}")


if __name__ == "__main__":
    main()