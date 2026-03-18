# TrackerML

ML-based analysis of web tracking infrastructure using DuckDuckGo's [Tracker Radar](https://github.com/duckduckgo/tracker-radar) dataset. Two models, one interactive visualization, all with on-device inference via [Kjarni](https://github.com/olafurjohannsson/kjarni).

## Projects

### 1. [Tracker Classification](tracker-classifier/)
Binary classification of tracking vs non-tracking domains using 295 behavioral features. XGBoost achieves F1 of 0.893. Identifies 338 tracking domains missed by the fingerprinting heuristic, including facebook.com and adnxs.com.

### 2. [Entity Attribution](entity-attribution/)
Predicts which company owns a domain based purely on behavioral patterns — API usage, cookie behavior, resource types. 57% of domains in the dataset have no ownership info. Includes a 4-class entity type classifier (75% accuracy) and a 13-class tracking entity attributor for ad tech domains.

### [Interactive Demo](site/)
Force graph visualization of the tracking ecosystem with all three models running in-browser via WebAssembly. Click any domain to see its tracking classification, entity type, and predicted owner — all computed locally in sub-millisecond time.

## Setup

```bash
# Clone
git clone https://github.com/olafurjohannsson/tracker-ml
cd tracker-ml

# Install dependencies
pip install -r requirements.txt

# Get Tracker Radar data (CC-BY-NC-SA 4.0)
git clone https://github.com/duckduckgo/tracker-radar data/tracker-radar
```

## Training Pipeline

```bash
# 1. Tracker classifier
python tracker-classifier/scripts/extract_features.py --tracker-radar data/tracker-radar --output data/features_us.parquet
python tracker-classifier/scripts/build_labels.py --features data/features_us.parquet --tracker-radar data/tracker-radar --output data/dataset_us.parquet
python tracker-classifier/scripts/train.py --data data/dataset_us_train.parquet --output tracker-classifier/models

# 2. Entity attribution
python entity-attribution/scripts/extract_features.py --tracker-radar data/tracker-radar --output entity-attribution/data
python entity-attribution/scripts/train.py --data entity-attribution/data --output entity-attribution/models

# 3. Build visualization data
python scripts/build_graph.py --tracker-radar data/tracker-radar --output site/data/graph.json
python scripts/build_demo_data.py \
    --tracker-features data/features_us.parquet \
    --entity-data entity-attribution/data/all_features.parquet \
    --graph site/data/graph.json \
    --tracker-scaler tracker-classifier/models/scaler.joblib \
    --cluster-scaler entity-attribution/models/entity_cluster_classifier_scaler.json \
    --tracking-scaler entity-attribution/models/tracking_entity_classifier_scaler.json \
    --cluster-config entity-attribution/models/entity_cluster_classifier_config.json \
    --tracking-config entity-attribution/models/tracking_entity_classifier_config.json \
    --output site/data/demo_data.json
```

## Acknowledgments

DuckDuckGo for open-sourcing the Tracker Radar dataset. EasyPrivacy maintainers for their work on tracking protection.

## License

Code: MIT. Models and dataset: CC-BY-NC-SA 4.0 (derived from Tracker Radar).