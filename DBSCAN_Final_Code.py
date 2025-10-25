import os, re, time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score
from sklearn.manifold import TSNE

# Central settings so I can tweak the run in one place
DATA_CSV = "/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
MAX_ROWS = None
RANDOM_STATE = 42
KDIST_MAX = 225_000
VIS_MAX = None
AUTO_TUNE_PER_VARIANT = False
N_JOBS = 1
MIN_SAMPLES = 20
PCA_COMPONENTS_VIS = 2
PCA_VARIANCES = [95, 90, 85, 80, 75]

# Load the dataset and clean up any weird header spacing
def read_csv_clean(path: str) -> pd.DataFrame:
    """Load CSV file and trim header spaces."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"[Info] Loaded: {path}  shape={df.shape}")
    return df

# Try to figure out which column holds the human-readable labels
def guess_label_column(df: pd.DataFrame) -> Optional[str]:
    """Guess target/label column in dataset."""
    for c in ["Label", "label", "Attack", "attack", "Class", "class"]:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            return c
    pat = re.compile(r"(label|attack|class)", re.I)
    for c in df.columns:
        if pat.search(c) and not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None

# Normalize label text so comparisons are reliable
def tidy_text_labels(y: pd.Series) -> pd.Series:
    """Trim and clean text labels."""
    return y.astype(str).str.strip()

# Collapse all classes into a simple BENIGN vs ATTACK view
def as_binary_benign_attack(y_text: pd.Series) -> pd.Series:
    """Convert all labels to BENIGN or ATTACK."""
    upper = y_text.str.upper()
    return pd.Series(np.where(upper.str.contains("BENIGN"), "BENIGN", "ATTACK"),
                     index=y_text.index, name="BinaryLabel")

# Optionally downsample rows to keep runs light on slower machines
def take_rows(df: pd.DataFrame, max_rows: Optional[int], seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Subsample dataframe rows if exceeding limit."""
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)

# Build a clean numeric feature matrix and drop junky columns
def build_numeric_X(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select numeric columns and clean NaN/zero variance."""
    info = {"dropped_non_numeric": [], "dropped_all_nan": [], "dropped_zero_var": []}
    num = df.select_dtypes(include=[np.number]).copy()
    info["dropped_non_numeric"] = [c for c in df.columns if c not in num.columns]
    num = num.replace([np.inf, -np.inf], np.nan)
    all_nan = [c for c in num.columns if num[c].isna().all()]
    if all_nan:
        num = num.drop(columns=all_nan)
        info["dropped_all_nan"] = all_nan
    zero_var = num.nunique()
    zero_cols = zero_var[zero_var <= 1].index.tolist()
    if zero_cols:
        num = num.drop(columns=zero_cols)
    info["dropped_zero_var"] = zero_cols
    for c in num.columns:
        if num[c].isna().any():
            med = num[c].median()
            num[c] = num[c].fillna(0.0 if pd.isna(med) else med)
    return num, info

# Standardize features so DBSCAN distances behave nicely
def zscore32(Xdf: pd.DataFrame) -> np.ndarray:
    """Normalize features using z-score scaling."""
    scaler = StandardScaler()
    return scaler.fit_transform(Xdf.values).astype(np.float32)

# Find the “elbow” of a curve using a simple geometry trick
def knee_from_curve(sorted_vals: np.ndarray):
    """Find geometric knee point on curve."""
    n = len(sorted_vals)
    if n < 3:
        return n - 1, float(sorted_vals[-1])
    x = np.arange(n, dtype=float)
    y = sorted_vals.astype(float)
    x0, y0 = 0.0, y[0]
    x1, y1 = float(n - 1), y[-1]
    v = np.array([x1 - x0, y1 - y0], float)
    L = np.hypot(v[0], v[1])
    if L == 0:
        return n - 1, float(y[-1])
    u = v / L
    d = []
    for i in range(n):
        p = np.array([x[i] - x0, y[i] - y0], float)
        proj = np.dot(p, u) * u
        perp = p - proj
        d.append(np.hypot(perp[0], perp[1]))
    k = int(np.argmax(d))
    return k, float(y[k])

# Build the k-distance curve that guides DBSCAN’s eps choice
def kdist_curve(X: np.ndarray, k: int, limit: int = KDIST_MAX) -> np.ndarray:
    """Compute k-distance curve for DBSCAN."""
    n = X.shape[0]
    if n > limit:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=limit, replace=False)
        X_used = X[idx]
    else:
        X_used = X
    nn = NearestNeighbors(n_neighbors=k, n_jobs=N_JOBS).fit(X_used)
    dists, _ = nn.kneighbors(X_used)
    return np.sort(dists[:, -1])

# Run DBSCAN and get one label per point (clusters and noise=-1)
def cluster_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run DBSCAN clustering algorithm."""
    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=N_JOBS).fit_predict(X)

# Turn a confusion matrix into the four basic counts
def cm_to_counts(y_true, y_pred) -> Tuple[int, int, int, int]:
    """Return TN, FP, FN, TP counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["BENIGN", "ATTACK"]).ravel()
    return tn, fp, fn, tp

# Compute the headline binary metrics we care about
def score_binary(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    """Compute ACC, PR, DR, F1, FAR metrics."""
    ACC = accuracy_score(y_true_bin, y_pred_bin)
    PR  = precision_score(y_true_bin, y_pred_bin, pos_label="ATTACK", zero_division=0)
    DR  = recall_score(y_true_bin, y_pred_bin, pos_label="ATTACK", zero_division=0)
    F1  = f1_score(y_true_bin, y_pred_bin, pos_label="ATTACK", zero_division=0)
    tn, fp, fn, tp = cm_to_counts(y_true_bin, y_pred_bin)
    FAR = fp / (fp + tn) if (fp + tn) else 0.0
    return {"ACC": ACC, "PR": PR, "DR": DR, "F1": F1, "FAR": FAR}

# Check how often each attack type gets flagged as an attack
def attackwise_recall(y_true_text: pd.Series, y_pred_bin: np.ndarray) -> pd.Series:
    """Compute per-attack detection rate."""
    y_upper = y_true_text.str.upper()
    types = sorted([t for t in y_upper.unique() if t != "BENIGN"])
    out = {}
    for t in types:
        m = (y_upper == t)
        out[t] = np.nan if m.sum() == 0 else (y_pred_bin[m] == "ATTACK").mean()
    return pd.Series(out).sort_values(ascending=False)

# Figure 3: show how accuracy/precision/recall/F1 move with PCA variance
def fig_perf_vs_var(variances, ACCs, PRs, DRs, F1s):
    """Figure 3: performance vs PCA variance."""
    plt.figure(figsize=(8.8, 4.8))
    plt.plot(variances, ACCs, marker='o', label="Accuracy")
    plt.plot(variances, PRs, marker='o', label="Precision")
    plt.plot(variances, DRs, marker='o', label="Recall")
    plt.plot(variances, F1s, marker='o', label="F1 score")
    plt.gca().invert_xaxis()
    plt.ylim(0, 1.0)
    plt.ylabel("Score (0–1)")
    plt.xlabel("Variance kept by PCA (%)")
    plt.title("Figure 3: How performance shifts as PCA keeps more variance")
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=.6)
    plt.tight_layout()
    plt.show()

# Figure 4: track how long clustering takes as we keep more variance
def fig_time_vs_var(variances, times):
    """Figure 4: training time vs PCA variance."""
    plt.figure(figsize=(8.8, 4.2))
    plt.plot(variances, times, marker='o')
    plt.gca().invert_xaxis()
    plt.ylabel("Seconds to cluster")
    plt.xlabel("Variance kept by PCA (%)")
    plt.title("Figure 4: Training time vs. PCA variance (lower is faster)")
    plt.grid(axis='y', linestyle=':', alpha=.6)
    plt.tight_layout()
    plt.show()

# Figure 5: see how the false alarm rate changes with PCA variance
def fig_far_vs_var(variances, fars):
    """Figure 5: FAR vs PCA variance."""
    plt.figure(figsize=(8.8, 4.2))
    plt.plot(variances, fars, marker='o')
    plt.gca().invert_xaxis()
    plt.ylabel("False Alarm Rate")
    plt.xlabel("Variance kept by PCA (%)")
    plt.title("Figure 5: False alarms vs. PCA variance")
    plt.grid(axis='y', linestyle=':', alpha=.6)
    plt.tight_layout()
    plt.show()

# Figure 6: bar chart for recall by each attack category
def fig_attack_recall(sr: pd.Series):
    """Figure 6: detection rate by attack type."""
    plt.figure(figsize=(10, 4.6))
    ax = sr.plot(kind='bar')
    plt.ylim(0, 1.0)
    plt.ylabel("Recall")
    plt.xlabel("Attack type")
    plt.title("Figure 6: Detection rate by attack type (higher is better)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()

# Figure 2: quick snapshot of the four main metrics
def fig_summary_bars(metrics: Dict[str, float]):
    """Figure 2: summary of ACC/PR/DR/F1."""
    names = ["ACC", "PR", "DR", "F1"]
    vals = [metrics[k] for k in names]
    plt.figure(figsize=(6.4, 4.2))
    plt.bar(names, vals, color=['#4B9CD3','#8AC926','#FFB703','#FB8500'])
    plt.ylim(0, 1.0)
    plt.ylabel("Score (0–1)")
    plt.title("Figure 2: DBSCAN at a glance (Accuracy · Precision · Recall · F1)")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

# Figure 7: simple count of each class in this slice of the dataset
def class_hist(y_text: pd.Series):
    """Figure 7: class distribution histogram."""
    y_up = y_text.str.upper()
    counts = y_up.value_counts()
    plt.figure(figsize=(10, 4.6))
    counts.plot(kind='bar', color='#7aa4d8')
    plt.ylabel("Number of flows")
    plt.xlabel("Class")
    plt.title("Figure 7: Class balance in CIC-IDS-2017 (this subset)")
    plt.tight_layout()
    plt.show()

# Figure 8: 2D PCA view contrasting clustered points and anomalies
def fig_pca_scatter_cluster_vs_anomaly(pca_vis, labels_full):
    """Figure 8: PCA 2D — clusters vs anomalies."""
    plt.figure(figsize=(7.2, 6))
    mask_noise = (labels_full == -1)
    plt.scatter(pca_vis[~mask_noise, 0], pca_vis[~mask_noise, 1], s=6, alpha=0.35, label="Clustered")
    plt.scatter(pca_vis[mask_noise, 0], pca_vis[mask_noise, 1], s=10, alpha=0.9, label="Anomaly (noise)")
    plt.title("Figure 8: PCA view: clusters vs. flagged anomalies")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2"); plt.legend(); plt.tight_layout(); plt.show()

# Figure 9: 3D PCA view to spot cluster structure and noise
def fig_pca3_scatter(X_vis, labels_vis):
    """Figure 9: PCA 3D — clusters vs anomalies."""
    mask_noise_vis = (labels_vis == -1)
    pca3 = PCA(n_components=3, random_state=RANDOM_STATE).fit_transform(X_vis)
    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca3[~mask_noise_vis, 0], pca3[~mask_noise_vis, 1], pca3[~mask_noise_vis, 2],
               s=6, alpha=0.35, label="Clustered")
    ax.scatter(pca3[mask_noise_vis, 0], pca3[mask_noise_vis, 1], pca3[mask_noise_vis, 2],
               s=10, alpha=0.9, label="Anomaly (noise)")
    ax.set_title("Figure 9: 3D PCA view: clusters vs. anomalies")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.set_zlabel("PCA 3")
    ax.legend()
    plt.tight_layout(); plt.show()

# Figure 10: 2D PCA focusing only on the clustered (non-noise) points
def fig_pca2_cluster_only(X_vis, labels_vis):
    """Figure 10: PCA 2D — clustered-only points."""
    mask_noise_vis = (labels_vis == -1)
    pca2_vis = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_vis)
    plt.figure(figsize=(7.2, 6))
    plt.scatter(pca2_vis[~mask_noise_vis, 0], pca2_vis[~mask_noise_vis, 1],
                s=6, alpha=0.6, label="Clustered")
    plt.title("Figure 10: PCA (2D): clustered points only")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2"); plt.legend()
    plt.tight_layout(); plt.show()

# Figure 11: 2D PCA focusing only on points DBSCAN marked as noise
def fig_pca2_noise_only(X_vis, labels_vis):
    """Figure 11: PCA 2D — anomalies-only points."""
    mask_noise_vis = (labels_vis == -1)
    pca2_vis = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_vis)
    plt.figure(figsize=(7.2, 6))
    plt.scatter(pca2_vis[mask_noise_vis, 0], pca2_vis[mask_noise_vis, 1],
                s=10, alpha=0.9, label="Anomalies (noise)")
    plt.title("Figure 11: PCA (2D): anomalies (noise) only")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2"); plt.legend()
    plt.tight_layout(); plt.show()

# Figures 12–15: t-SNE views (raw, DBSCAN labels, ground truth, and a suptitle)
def fig_tsne_panels(X, y_text, labels_full):
    """Figures 12–15: t-SNE panels and suptitle."""
    n = X.shape[0]
    max_n = 8000
    if n > max_n:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=max_n, replace=False)
        X_tsne = X[idx]
        y_text_tsne = y_text.iloc[idx].reset_index(drop=True)
        labels_tsne = labels_full[idx]
    else:
        X_tsne, y_text_tsne, labels_tsne = X, y_text, labels_full
    X50 = PCA(n_components=min(50, X_tsne.shape[1]), random_state=RANDOM_STATE).fit_transform(X_tsne)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200.0, init="pca",
                random_state=RANDOM_STATE, angle=0.5, method="barnes_hut")
    Z = tsne.fit_transform(X50)
    cluster_keys = np.unique(labels_tsne)
    palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#393b79","#637939","#8c6d31","#843c39","#7b4173",
    ]
    clust_cmap = {k: palette[i % len(palette)] for i, k in enumerate(cluster_keys)}
    clust_cmap[-1] = "#000000"
    y_up = y_text_tsne.str.upper()
    if y_up.nunique() > 12:
        top = set(y_up.value_counts().index[:11])
        y_up = y_up.where(y_up.isin(top), other="OTHER")
    gt_keys = sorted(y_up.unique())
    gt_cmap = {k: palette[i % len(palette)] for i, k in enumerate(gt_keys)}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7)
    axes[0].set_title("Figure 12: t-SNE map")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    for k in cluster_keys:
        m = (labels_tsne == k)
        axes[1].scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.85, c=clust_cmap[k],
                        label=("Noise (-1)" if k == -1 else f"C{k}"))
    axes[1].set_title("Figure 13: t-SNE colored by DBSCAN")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(markerscale=2, fontsize=8, frameon=True, loc="best")
    for cls in gt_keys:
        m = (y_up == cls)
        axes[2].scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.85, c=gt_cmap[cls], label=cls)
    axes[2].set_title("Figure 14: t-SNE colored by ground truth")
    axes[2].set_xlabel("t-SNE 1")
    axes[2].set_ylabel("t-SNE 2")
    axes[2].legend(markerscale=2, fontsize=8, frameon=True, loc="best")
    plt.suptitle("Figure 15: t-SNE maps of CIC-IDS-2017 (raw, DBSCAN labels, ground truth)", y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

# Auto-pick a good eps using the knee plus a tiny silhouette sweep
def pick_eps_auto(X: np.ndarray, min_samples: int) -> float:
    """Figure 1: auto-pick eps (knee+silhouette) and plot k-distance."""
    kth = kdist_curve(X, k=min_samples, limit=KDIST_MAX)
    knee_idx, knee_eps = knee_from_curve(kth)
    base = knee_eps if knee_eps > 0 else float(np.median(kth))
    grid = np.unique(np.clip(base * np.linspace(0.8, 1.3, 6), 1e-8, None))
    best_eps, best_sil = base, -1.0
    for e in grid:
        labels = cluster_dbscan(X, eps=float(e), min_samples=min_samples)
        mask = labels != -1
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            continue
        try:
            sil = silhouette_score(X[mask], labels[mask])
        except Exception:
            continue
        if sil > best_sil:
            best_eps, best_sil = float(e), sil
    plt.figure(figsize=(9, 5))
    plt.plot(kth, lw=2.0, label="k-distance")
    plt.axvline(knee_idx, ls="--", lw=1.8, label="Knee index")
    plt.axhline(knee_eps, ls="--", lw=1.8, label=f"Knee eps≈{knee_eps:.4f}")
    plt.axhline(best_eps, ls=":", lw=2.0, label=f"Chosen eps≈{best_eps:.4f}")
    plt.xlabel("Points sorted by k-neighbor distance")
    plt.ylabel(f"Distance to k-th neighbor (k={min_samples})")
    plt.title("Figure 1: K-distance curve: knee vs. chosen eps")
    plt.legend()
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.show()
    print(f"[Tune] eps* ≈ {best_eps:.4f} (knee≈{knee_eps:.4f})")
    return best_eps

# Main flow: load data, prep features, cluster, score, and draw all figures
if __name__ == "__main__":
    # Read the raw CSV and optionally downsample it
    df = read_csv_clean(DATA_CSV)
    df = take_rows(df, MAX_ROWS)

    # Find the label column and standardize label text
    target_col = guess_label_column(df)
    if target_col is None:
        raise RuntimeError("Target/Label column not found.")
    y_text = tidy_text_labels(df[target_col])

    # Create a simple BENIGN vs ATTACK label for evaluation
    y_bin = as_binary_benign_attack(y_text)

    # Build the numeric feature table and scale it
    df_num, drop_info = build_numeric_X(df)
    X = zscore32(df_num)
    print(f"[Prep] X shape: {X.shape} | numeric feats: {df_num.shape[1]} | dropped: {drop_info}")

    # Pick eps automatically using the k-distance knee and a small search
    eps_star = pick_eps_auto(X, MIN_SAMPLES)

    # Run DBSCAN once on the full standardized feature set
    t0 = time.time()
    labels_full = cluster_dbscan(X, eps=eps_star, min_samples=MIN_SAMPLES)
    base_time = time.time() - t0

    # Print quick cluster stats and how many points were flagged as noise
    n_clusters = len(set(labels_full)) - (1 if -1 in set(labels_full) else 0)
    noise_count = int((labels_full == -1).sum())
    total_points = int(labels_full.shape[0])
    print(f"[Clusters] Number of clusters: {n_clusters}")
    print(f"[Anomalies] Noise points: {noise_count}/{total_points} ({noise_count/total_points:.2%})")

    # Check silhouette quality for the non-noise portion if it makes sense
    mask_clustered = (labels_full != -1)
    if mask_clustered.sum() > 1 and len(np.unique(labels_full[mask_clustered])) >= 2:
        sil = silhouette_score(X[mask_clustered], labels_full[mask_clustered])
        print(f"[Silhouette] {sil:.4f}")

    # Turn DBSCAN output into a simple BENIGN/ATTACK prediction for scoring
    y_pred_full = np.where(labels_full == -1, "ATTACK", "BENIGN")
    metrics_full = score_binary(y_bin, y_pred_full)
    print("[Metrics] ACC:{ACC:.4f}  PR:{PR:.4f}  DR:{DR:.4f}  F1:{F1:.4f}  FAR:{FAR:.4f}  time:{t:.3f}s"
          .format(**metrics_full, t=base_time))

    # Figure 2: show the big four metrics in one glance
    fig_summary_bars(metrics_full)

    # Sweep a few PCA variance levels and track performance and runtime
    ACCs, PRs, DRs, F1s, FARs, TIMES = [], [], [], [], [], []
    for var in PCA_VARIANCES:
        pca = PCA(n_components=var/100.0, svd_solver="full", random_state=RANDOM_STATE)
        X_var = pca.fit_transform(X)
        eps_use = eps_star if not AUTO_TUNE_PER_VARIANT else pick_eps_auto(X_var, MIN_SAMPLES)
        t0 = time.time()
        labels_var = cluster_dbscan(X_var, eps=eps_use, min_samples=MIN_SAMPLES)
        TIMES.append(time.time() - t0)
        y_pred_var = np.where(labels_var == -1, "ATTACK", "BENIGN")
        m = score_binary(y_bin, y_pred_var)
        ACCs.append(m["ACC"]); PRs.append(m["PR"]); DRs.append(m["DR"]); F1s.append(m["F1"]); FARs.append(m["FAR"])
        print(f"[PCA {var}%] ACC:{m['ACC']:.4f} PR:{m['PR']:.4f} DR:{m['DR']:.4f} F1:{m['F1']:.4f} FAR:{m['FAR']:.4f} time:{TIMES[-1]:.3f}s")

    # Figures 3–5: plot performance curves, runtime, and false alarms
    fig_perf_vs_var(PCA_VARIANCES, ACCs, PRs, DRs, F1s)
    fig_time_vs_var(PCA_VARIANCES, TIMES)
    fig_far_vs_var(PCA_VARIANCES, FARs)

    # Figures 6–7: per-attack recall and class balance
    per_attack_dr = attackwise_recall(y_text, y_pred_full)
    fig_attack_recall(per_attack_dr)
    class_hist(y_text)

    # Figure 8: quick 2D PCA view of clustered vs noise points
    pca_vis = PCA(n_components=PCA_COMPONENTS_VIS, random_state=RANDOM_STATE).fit_transform(X)
    fig_pca_scatter_cluster_vs_anomaly(pca_vis, labels_full)

    # Prep a visualization subset for the 3D/2D PCA plots if needed
    rng = np.random.default_rng(RANDOM_STATE)
    if (VIS_MAX is not None) and (X.shape[0] > VIS_MAX):
        idx_vis = rng.choice(X.shape[0], VIS_MAX, replace=False)
    else:
        idx_vis = np.arange(X.shape[0])
    X_vis = X[idx_vis]
    labels_vis = labels_full[idx_vis]

    # Figures 9–11: 3D PCA view, clusters-only, and anomalies-only
    fig_pca3_scatter(X_vis, labels_vis)
    fig_pca2_cluster_only(X_vis, labels_vis)
    fig_pca2_noise_only(X_vis, labels_vis)

    # Figures 12–15: full t-SNE panel set with the combined suptitle
    fig_tsne_panels(X, y_text, labels_full)