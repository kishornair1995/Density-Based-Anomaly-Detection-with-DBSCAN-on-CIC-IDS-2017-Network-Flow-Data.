# Density-Based-Anomaly-Detection-with-DBSCAN-on-CIC-IDS-2017-Network-Flow-Data.

This project provides a single Python script that runs an **unsupervised DBSCAN** workflow on a subset of the CIC-IDS-2017 dataset (for example, `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`). The script:

- Loads and prepares the dataset
- Standardizes numeric features
- **Chooses `eps` automatically** using a k-distance knee plus a tiny silhouette sweep
- Clusters with DBSCAN and evaluates predictions after mapping to **BENIGN vs ATTACK**
- Produces **15 clearly labeled figures** covering performance, PCA projections, and t‑SNE maps

Every plot is titled as **Figure 1** through **Figure 15**, so you can copy them straight into a report.

---

## Repository Layout

- `dbscan_cicids.py` — main script with the full pipeline and all figure code
- `README.md` — this guide

> The dataset itself is not included. Please obtain CIC-IDS-2017 from the original source and respect their terms of use.

---

## Data Setup

- Point `DATA_CSV` at the CSV you want to analyze (top of the script):
  ```python
  DATA_CSV = "/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
  ```
- The script automatically searches for a label column (e.g., `Label`) and cleans text labels.

---

## Environment & Installation

**Python:** 3.9–3.12 recommended

**Dependencies:**
- numpy
- pandas
- scikit-learn
- matplotlib

### Quick install with pip
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib
```

*(Optional)* create a `requirements.txt`:
```
numpy
pandas
scikit-learn
matplotlib
```
and then run:
```bash
pip install -r requirements.txt
```

### Google Colab
- Upload the script and CSV (or mount Drive)
- Set `DATA_CSV` accordingly
- Run the script cells; figures will render inline

---

## How to Run

```bash
python DBSCAN_Final_Code
```

You’ll see console messages for:
- Dataset load and feature prep
- Auto-selected `eps` (knee vs chosen)
- Cluster and noise counts
- Silhouette (if valid on clustered points)
- Final metrics: **ACC**, **Precision**, **Recall (DR)**, **F1**, **FAR**
- PCA sweep performance and timing

All **15 figures** open in matplotlib (or inline in notebooks).

---

## Figures Overview (What You’ll Get)

1. **K-distance curve: knee vs. chosen eps**  
2. **DBSCAN at a glance** (Accuracy · Precision · Recall · F1)  
3. **Performance vs retained PCA variance**  
4. **Clustering time vs PCA variance** (lower is faster)  
5. **False-alarm rate vs PCA variance**  
6. **Detection rate by attack type**  
7. **Class distribution in this CIC-IDS-2017 split**  
8. **PCA (2D): clustered vs anomaly points**  
9. **PCA (3D): clustered vs anomaly points**  
10. **PCA (2D): clustered points only**  
11. **PCA (2D): anomalies (noise) only**  
12. **t‑SNE map**  
13. **t‑SNE colored by DBSCAN labels**  
14. **t‑SNE colored by ground truth**  
15. **t‑SNE overview suptitle** (three-panel summary)

Each title is already set in the code; you don’t need to rename anything.

---

## Key Config Options (edit at the top of the script)

| Name | Purpose | Default |
| --- | --- | --- |
| `DATA_CSV` | Path to the CSV file | `"/content/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"` |
| `MAX_ROWS` | Optional subsample size (None = use all rows) | `None` |
| `RANDOM_STATE` | Seed for reproducibility | `42` |
| `KDIST_MAX` | Cap for k-distance computation | `225_000` |
| `VIS_MAX` | Cap for visualization subsets | `None` |
| `AUTO_TUNE_PER_VARIANT` | Re-tune `eps` for each PCA variant | `False` |
| `N_JOBS` | Parallelism hint for sklearn ops | `1` |
| `MIN_SAMPLES` | DBSCAN `min_samples` | `20` |
| `PCA_COMPONENTS_VIS` | PCA components for 2D view | `2` |
| `PCA_VARIANCES` | PCA variance levels for sweep (%) | `[95, 90, 85, 80, 75]` |

**Performance tips**
- If memory is tight, lower `MAX_ROWS` and/or `KDIST_MAX`, or set `VIS_MAX` to a smaller value.
- Leave `AUTO_TUNE_PER_VARIANT = False` for faster runs.

---

## Metrics Explained (after binarization)

The script maps DBSCAN **noise** (`-1`) to **ATTACK** and cluster members to **BENIGN** for scoring:

- **ACC** – Accuracy
- **PR** – Precision (positive = ATTACK)
- **DR** – Detection Rate / Recall (positive = ATTACK)
- **F1** – Harmonic mean of precision and recall
- **FAR** – False Alarm Rate = FP / (FP + TN)

---

## Saving Plots (optional)

To save figures instead of showing them, add a `plt.savefig(...)` before `plt.show()`. For example:
```python
plt.tight_layout()
plt.savefig("figure_01_kdistance.png", dpi=300)
# plt.show()  # comment if you only want files
```

---

## Troubleshooting

- **FileNotFoundError**: Set the correct `DATA_CSV` path.  
- **No figures in notebooks**: Add `%matplotlib inline` in the first cell.  
- **High memory usage**: Reduce `MAX_ROWS`, `KDIST_MAX`, or set `VIS_MAX`.  
- **Silhouette not printed**: DBSCAN may have < 2 clusters among non-noise points, which is expected in some runs.

---

## Reproducibility

`RANDOM_STATE = 42` is used for subsampling, PCA, and t‑SNE where applicable.

---

## Credits & Licensing

- Data: **CIC-IDS-2017** (Canadian Institute for Cybersecurity). Please cite the original authors when publishing.  
- Algorithms and components: scikit-learn (`DBSCAN`, `PCA`, `TSNE`, metrics).

You may license your repository under **MIT** or another OSI-approved license. Example MIT header:

```
MIT License

Copyright (c) 2025 <Your Name>
...
```

---

## Quickstart Recap

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or: pip install numpy pandas scikit-learn matplotlib

# Edit DATA_CSV in the script to point to your CSV
python dbscan_cicids.py
```

That’s it — the console will print metrics and the **15 figures** will render.
