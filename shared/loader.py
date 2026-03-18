"""
Shared data loader for DuckDuckGo Tracker Radar.
Parses domain and entity JSON files and provides structured access
for all downstream models.

Usage:
    from shared import TrackerRadarLoader
    loader = TrackerRadarLoader("data/tracker-radar", region="US")
    loader.load()

    # Access parsed data
    loader.domains         # dict of {domain_name: parsed_dict}
    loader.entities        # dict of {entity_name: entity_dict}
    loader.entity_for      # dict of {domain_name: entity_name}
    loader.domains_for     # dict of {entity_name: [domain_names]}
    loader.api_weights     # dict of {api_name: weight}
    loader.cname_map       # dict of {domain_name: [cname_targets]}
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

from tqdm import tqdm


class TrackerRadarLoader:
    def __init__(self, tracker_radar_path: str, region: str = "US"):
        self.path = tracker_radar_path
        self.region = region

        # Populated by load()
        self.domains: dict = {}
        self.entities: dict = {}
        self.entity_for: dict = {}         # domain -> entity name
        self.domains_for: dict = {}        # entity name -> [domains]
        self.api_weights: dict = {}
        self.categories_for: dict = {}     # domain -> [categories]
        self.cname_map: dict = {}          # domain -> [cname target domains]

        self._loaded = False

    def load(self, verbose: bool = True):
        """Load and index all data."""
        if self._loaded:
            return

        self._load_api_weights(verbose)
        self._load_entities(verbose)
        self._load_domains(verbose)
        self._build_cname_map(verbose)
        self._loaded = True

        if verbose:
            print(f"\n=== Tracker Radar Loaded ===")
            print(f"  Domains:  {len(self.domains):,}")
            print(f"  Entities: {len(self.entities):,}")
            print(f"  Domains with owner: {len(self.entity_for):,} ({len(self.entity_for)/len(self.domains)*100:.1f}%)")
            print(f"  Domains with categories: {len(self.categories_for):,}")
            print(f"  Domains with CNAMEs: {len(self.cname_map):,}")
            print(f"  API weights: {len(self.api_weights):,}")

    def _load_api_weights(self, verbose: bool):
        weights_path = os.path.join(
            self.path, "build-data", "generated", "api_fingerprint_weights.json"
        )
        with open(weights_path) as f:
            self.api_weights = json.load(f)
        if verbose:
            print(f"Loaded {len(self.api_weights)} API weights")

    def _load_entities(self, verbose: bool):
        entities_dir = os.path.join(self.path, "entities")
        entity_files = list(Path(entities_dir).glob("*.json"))

        if verbose:
            print(f"Loading {len(entity_files)} entity files...")

        for f in tqdm(entity_files, desc="Entities", disable=not verbose):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                name = data.get("name", "")
                if not name:
                    continue
                self.entities[name] = data
                properties = data.get("properties", [])
                self.domains_for[name] = properties
                for domain in properties:
                    self.entity_for[domain] = name
            except (json.JSONDecodeError, KeyError):
                continue

    def _load_domains(self, verbose: bool):
        domains_dir = os.path.join(self.path, "domains", self.region)
        domain_files = list(Path(domains_dir).glob("*.json"))

        if verbose:
            print(f"Loading {len(domain_files)} domain files from {self.region}...")

        errors = 0
        for f in tqdm(domain_files, desc="Domains", disable=not verbose):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                domain = data.get("domain", "")
                if not domain:
                    errors += 1
                    continue
                self.domains[domain] = data

                # Index categories
                cats = data.get("categories", [])
                if cats:
                    self.categories_for[domain] = cats

                # Also index owner from domain file if not already known
                owner = data.get("owner", {})
                if owner.get("name") and domain not in self.entity_for:
                    self.entity_for[domain] = owner["name"]
                    if owner["name"] not in self.domains_for:
                        self.domains_for[owner["name"]] = []
                    if domain not in self.domains_for[owner["name"]]:
                        self.domains_for[owner["name"]].append(domain)

            except (json.JSONDecodeError, KeyError, TypeError):
                errors += 1
                continue

        if verbose and errors:
            print(f"  ({errors} files skipped due to errors)")

    def _build_cname_map(self, verbose: bool):
        """Build mapping of domains to their CNAME targets."""
        for domain, data in self.domains.items():
            cnames = data.get("cnames", [])
            if cnames:
                # cnames can be strings or dicts depending on version
                targets = []
                for cname in cnames:
                    if isinstance(cname, str):
                        targets.append(cname)
                    elif isinstance(cname, dict):
                        # Some versions have {"original": ..., "resolved": ...}
                        resolved = cname.get("resolved", cname.get("original", ""))
                        if resolved:
                            targets.append(resolved)
                if targets:
                    self.cname_map[domain] = targets

    def get_domain_data(self, domain: str) -> Optional[dict]:
        """Get full domain data dict."""
        return self.domains.get(domain)

    def get_entity(self, domain: str) -> Optional[str]:
        """Get entity name for a domain."""
        return self.entity_for.get(domain)

    def get_categories(self, domain: str) -> list:
        """Get categories for a domain."""
        return self.categories_for.get(domain, [])

    def get_cnames(self, domain: str) -> list:
        """Get CNAME targets for a domain."""
        return self.cname_map.get(domain, [])

    def get_entity_domains(self, entity_name: str) -> list:
        """Get all domains owned by an entity."""
        return self.domains_for.get(entity_name, [])

    def iter_domains(self):
        """Iterate over (domain_name, domain_data) pairs."""
        yield from self.domains.items()

    def get_all_apis(self) -> list:
        """Get sorted list of all APIs that appear in the weights file."""
        all_apis = set()
        for domain, data in self.domains.items():
            for resource in data.get("resources", []):
                all_apis.update(resource.get("apis", {}).keys())
        return sorted(all_apis & set(self.api_weights.keys()))