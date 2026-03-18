"""
Build the graph data for the interactive visualization.
Exports nodes (domains) and edges (entity ownership + CNAME relationships)
as JSON for the frontend.

Usage:
    python site/scripts/build_graph.py \
        --tracker-radar data/tracker-radar \
        --region US \
        --max-nodes 3000 \
        --output site/data/graph.json
"""

import sys
import os
import argparse
import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared import TrackerRadarLoader


# Top entity colors (rest get gray)
ENTITY_COLORS = {
    "Google LLC": "#4285F4",
    "Facebook, Inc.": "#1877F2",
    "Microsoft Corporation": "#00A4EF",
    "Amazon Technologies, Inc.": "#FF9900",
    "Adobe Inc.": "#FF0000",
    "Oracle Corporation": "#F80000",
    "Salesforce.com, Inc.": "#00A1E0",
    "The Trade Desk Inc": "#00C1B5",
    "Akamai Technologies": "#009BDE",
    "Shopify Inc.": "#7AB55C",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker-radar", required=True)
    parser.add_argument("--region", default="US")
    parser.add_argument("--max-nodes", type=int, default=3000,
                        help="Max domains to include (by prevalence)")
    parser.add_argument("--output", default="site/data/graph.json")
    args = parser.parse_args()

    loader = TrackerRadarLoader(args.tracker_radar, args.region)
    loader.load()

    # Sort domains by prevalence and take top N
    domain_list = []
    for domain_name, data in loader.domains.items():
        domain_list.append({
            "domain": domain_name,
            "prevalence": data.get("prevalence", 0),
            "sites": data.get("sites", 0),
        })

    domain_list.sort(key=lambda x: -x["prevalence"])
    selected_domains = set(d["domain"] for d in domain_list[:args.max_nodes])

    print(f"Selected {len(selected_domains)} domains by prevalence")

    # Build nodes
    nodes = []
    for domain_name in tqdm(selected_domains, desc="Building nodes"):
        data = loader.domains[domain_name]
        entity = loader.get_entity(domain_name)
        categories = loader.get_categories(domain_name)
        cnames = loader.get_cnames(domain_name)

        # Determine node type
        is_tracking = bool(set(categories) & {
            "Ad Motivated Tracking", "Advertising", "Analytics",
            "Audience Measurement", "Session Replay", "Action Pixels",
            "Third-Party Analytics Marketing", "Ad Fraud",
        })
        is_cdn = "CDN" in categories

        # Aggregate API info
        apis = set()
        total_cookies = 0
        for r in data.get("resources", []):
            apis.update(r.get("apis", {}).keys())
            total_cookies += r.get("cookies", 0)

        node = {
            "id": domain_name,
            "prevalence": data.get("prevalence", 0),
            "sites": data.get("sites", 0),
            "entity": entity or None,
            "entity_display": loader.entities.get(entity, {}).get("displayName", entity) if entity else None,
            "categories": categories,
            "fp_score": data.get("fingerprinting", 0),
            "is_tracking": is_tracking,
            "is_cdn": is_cdn,
            "distinct_apis": len(apis),
            "cookie_prevalence": round(total_cookies, 6),
            "has_cnames": len(cnames) > 0,
            "cname_targets": [c for c in cnames if c in selected_domains][:5],
            "subdomain_count": len(data.get("subdomains", [])),
            "color": ENTITY_COLORS.get(entity, None),
        }
        nodes.append(node)

    # Build edges
    # Edge type 1: entity ownership (domains sharing an entity)
    # We don't create edges between ALL domains of an entity (too dense)
    # Instead, we create a star topology: highest-prevalence domain is hub
    edges = []
    entity_groups = defaultdict(list)
    for node in nodes:
        if node["entity"]:
            entity_groups[node["entity"]].append(node)

    for entity, entity_nodes in entity_groups.items():
        if len(entity_nodes) < 2:
            continue
        # Sort by prevalence, hub is highest
        entity_nodes.sort(key=lambda x: -x["prevalence"])
        hub = entity_nodes[0]["id"]
        for other in entity_nodes[1:]:
            edges.append({
                "source": hub,
                "target": other["id"],
                "type": "ownership",
            })

    # Edge type 2: CNAME relationships
    for node in nodes:
        for target in node.get("cname_targets", []):
            edges.append({
                "source": node["id"],
                "target": target,
                "type": "cname",
            })

    # Entity summary for legend
    entity_summary = []
    for entity, entity_nodes in sorted(entity_groups.items(), key=lambda x: -len(x[1])):
        display = loader.entities.get(entity, {}).get("displayName", entity)
        entity_summary.append({
            "name": entity,
            "display": display,
            "domain_count": len(entity_nodes),
            "color": ENTITY_COLORS.get(entity, None),
        })

    graph = {
        "nodes": nodes,
        "edges": edges,
        "entities": entity_summary[:50],
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "ownership_edges": sum(1 for e in edges if e["type"] == "ownership"),
            "cname_edges": sum(1 for e in edges if e["type"] == "cname"),
            "entities_shown": len(entity_groups),
            "nodes_with_entity": sum(1 for n in nodes if n["entity"]),
            "nodes_without_entity": sum(1 for n in nodes if not n["entity"]),
            "tracking_nodes": sum(1 for n in nodes if n["is_tracking"]),
            "cdn_nodes": sum(1 for n in nodes if n["is_cdn"]),
        }
    }

    print(f"\n=== Graph Summary ===")
    for key, val in graph["stats"].items():
        print(f"  {key}: {val:,}")

    print(f"\nTop entities by domain count in graph:")
    for e in entity_summary[:15]:
        color_tag = f" [{e['color']}]" if e['color'] else ""
        print(f"  {e['domain_count']:4d}  {e['display']}{color_tag}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(graph, f)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()