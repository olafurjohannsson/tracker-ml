"""
Generate figures, run cross-validation, and produce final evaluation
for the paper.

Usage:
    python scripts/evaluate.py \
        --dataset data/dataset_us_train.parquet \
        --output-dir results/
"""

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from safetensors.torch import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


class TrackerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(hidden_dim // 2, 2)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.layer3(x)
        return x


def prepare_data(df):
    exclude_cols = ["domain", "label", "label_source", "fingerprinting_score"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feature_cols


def train_feedforward_fold(X_train, y_train, X_test, y_test):
    """Train feedforward for one fold, return predictions and probabilities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    model = TrackerClassifier(input_dim=input_dim, hidden_dim=128).to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)

    n_samples = len(y_train)
    class_counts = np.bincount(y_train)
    class_weights = n_samples / (2 * class_counts)
    weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(100):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(device)
            out = model(X_test_t)
            val_loss = criterion(out, torch.LongTensor(y_test).to(device))
            preds = out.argmax(dim=1).cpu().numpy()
            f1 = f1_score(y_test, preds)

        scheduler.step(val_loss)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 20:
            break

    model.load_state_dict(best_state)
    model = model.cpu().eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X_test))
        y_pred = out.argmax(dim=1).numpy()
        y_prob = torch.softmax(out, dim=1)[:, 1].numpy()

    return y_pred, y_prob


def run_cross_validation(X, y, feature_cols, n_folds=5):
    """Run stratified k-fold CV for all three models + heuristic."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {name: {"f1": [], "precision": [], "recall": [], "accuracy": [], "roc_auc": []}
               for name in ["Random Forest", "XGBoost", "Feedforward NN"]}

    print(f"\n{'='*60}")
    print(f"{n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_folds}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]

        # XGBoost
        n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            scale_pos_weight=n_neg/n_pos, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred = xgb_model.predict(X_test)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

        # Feedforward
        ff_pred, ff_prob = train_feedforward_fold(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

        # Record metrics
        for name, pred, prob in [
            ("Random Forest", rf_pred, rf_prob),
            ("XGBoost", xgb_pred, xgb_prob),
            ("Feedforward NN", ff_pred, ff_prob),
        ]:
            results[name]["f1"].append(f1_score(y_test, pred))
            results[name]["precision"].append(precision_score(y_test, pred))
            results[name]["recall"].append(recall_score(y_test, pred))
            results[name]["accuracy"].append(accuracy_score(y_test, pred))
            results[name]["roc_auc"].append(roc_auc_score(y_test, prob))

    # Print summary
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({n_folds}-fold)")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'F1':>12} {'Precision':>12} {'Recall':>12} {'ROC-AUC':>12}")
    print("-" * 68)
    for name in results:
        f1 = results[name]["f1"]
        prec = results[name]["precision"]
        rec = results[name]["recall"]
        auc = results[name]["roc_auc"]
        print(f"{name:<20} {np.mean(f1):.3f}±{np.std(f1):.3f}"
              f" {np.mean(prec):.3f}±{np.std(prec):.3f}"
              f" {np.mean(rec):.3f}±{np.std(rec):.3f}"
              f" {np.mean(auc):.3f}±{np.std(auc):.3f}")

    return results


def plot_roc_curves(df, X, y, feature_cols, output_dir):
    """Generate ROC curves for all models + heuristic baseline."""
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=0.2, stratify=y, random_state=42
    )
    df_test = df.iloc[idx_test]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Heuristic "probability" = normalized fingerprinting score
    heuristic_score = df_test["fingerprinting_score"].values / 3.0
    fpr_h, tpr_h, _ = roc_curve(y_test, heuristic_score)
    auc_h = roc_auc_score(y_test, heuristic_score)
    ax.plot(fpr_h, tpr_h, "--", color="gray", linewidth=1.5,
            label=f"FP Heuristic (AUC={auc_h:.3f})")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
    auc_rf = roc_auc_score(y_test, rf_prob)
    ax.plot(fpr_rf, tpr_rf, color="#2196F3", linewidth=1.5,
            label=f"Random Forest (AUC={auc_rf:.3f})")

    # XGBoost
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        scale_pos_weight=n_neg/n_pos, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
    auc_xgb = roc_auc_score(y_test, xgb_prob)
    ax.plot(fpr_xgb, tpr_xgb, color="#4CAF50", linewidth=1.5,
            label=f"XGBoost (AUC={auc_xgb:.3f})")

    # Feedforward
    ff_pred, ff_prob = train_feedforward_fold(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    fpr_ff, tpr_ff, _ = roc_curve(y_test, ff_prob)
    auc_ff = roc_auc_score(y_test, ff_prob)
    ax.plot(fpr_ff, tpr_ff, color="#FF9800", linewidth=1.5,
            label=f"Feedforward NN (AUC={auc_ff:.3f})")

    ax.plot([0, 1], [0, 1], ":", color="lightgray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: ML Models vs Fingerprinting Heuristic")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Saved ROC curves to {path}")
    plt.close()


def plot_feature_importance(X, y, feature_cols, output_dir, top_n=15):
    """Generate feature importance bar chart from Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    top = importances.nlargest(top_n).sort_values()

    # Clean up feature names for display
    display_names = []
    for name in top.index:
        name = name.replace("api_", "").replace("_prototype_", ".")
        name = name.replace("api_count_", "count: ")
        # Truncate long names
        if len(name) > 40:
            name = name[:37] + "..."
        display_names.append(name)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = ["#2196F3" if not n.startswith("api") else "#90CAF9"
              for n in top.index]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Features by Random Forest Importance")

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Saved feature importance to {path}")
    plt.close()


def plot_confusion_matrices(df, X, y, feature_cols, output_dir):
    """Generate confusion matrices for all models."""
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)), test_size=0.2, stratify=y, random_state=42
    )
    df_test = df.iloc[idx_test]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        scale_pos_weight=n_neg/n_pos, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    ff_pred, _ = train_feedforward_fold(X_train_scaled, y_train, X_test_scaled, y_test)

    heuristic_pred = (df_test["fingerprinting_score"].values >= 2).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    labels = ["Non-tracking", "Tracking"]

    for ax, name, preds in [
        (axes[0], "FP Heuristic\n(score ≥ 2)", heuristic_pred),
        (axes[1], "Random Forest", rf.predict(X_test)),
        (axes[2], "XGBoost", xgb_model.predict(X_test)),
        (axes[3], "Feedforward NN", ff_pred),
    ]:
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels, cbar=False)
        f1 = f1_score(y_test, preds)
        ax.set_title(f"{name}\nF1={f1:.3f}")
        ax.set_ylabel("True" if ax == axes[0] else "")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Saved confusion matrices to {path}")
    plt.close()


def plot_label_vs_heuristic(df, output_dir):
    """Visualize the disagreement between labels and heuristic."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ct = pd.crosstab(df["label"], df["fingerprinting_score"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)

    ct_norm.plot(kind="bar", stacked=True, ax=ax,
                 color=["#E8EAF6", "#9FA8DA", "#5C6BC0", "#283593"])
    ax.set_xlabel("Ground Truth Label")
    ax.set_ylabel("Proportion")
    ax.set_title("Fingerprinting Score Distribution by Label")
    ax.set_xticklabels(["Non-tracking (0)", "Tracking (1)"], rotation=0)
    ax.legend(title="FP Score", labels=["0", "1", "2", "3"])

    plt.tight_layout()
    path = os.path.join(output_dir, "label_vs_heuristic.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Saved label vs heuristic to {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="results/")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV (slow)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_parquet(args.dataset)
    X, y, feature_cols = prepare_data(df)
    print(f"Loaded {len(df)} domains, {len(feature_cols)} features")

    # === Cross-validation ===
    if not args.skip_cv:
        cv_results = run_cross_validation(X, y, feature_cols, n_folds=args.cv_folds)
        cv_path = os.path.join(args.output_dir, "cv_results.json")
        cv_export = {}
        for name in cv_results:
            cv_export[name] = {
                metric: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "folds": [float(v) for v in values],
                }
                for metric, values in cv_results[name].items()
            }
        with open(cv_path, "w") as f:
            json.dump(cv_export, f, indent=2)
        print(f"\nSaved CV results to {cv_path}")

    # === Figures ===
    print("\nGenerating figures...")
    plot_roc_curves(df, X, y, feature_cols, args.output_dir)
    plot_feature_importance(X, y, feature_cols, args.output_dir)
    plot_confusion_matrices(df, X, y, feature_cols, args.output_dir)
    plot_label_vs_heuristic(df, args.output_dir)

    print("\nDone! All figures saved to", args.output_dir)


if __name__ == "__main__":
    main()