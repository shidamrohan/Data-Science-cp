"""
src/clustering.py
──────────────────────────────────────────────────────────
RFM Customer Segmentation using K-Means (K=4).

Pipeline:
  RFM features → log1p → StandardScaler → KMeans(K=4)
  → Persona label mapping → cluster summary

Saves:
  models/kmeans_model.pkl
  models/rfm_scaler.pkl
  data/processed/rfm_clusters.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from preprocessing import load_clean, run_pipeline, PROCESSED_DIR
from feature_engineering import build_rfm

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

K = 4
RFM_FEATURES = ["R_log", "F_log", "M_log"]


def _assign_persona(cluster_centers: np.ndarray, labels: np.ndarray, rfm: pd.DataFrame) -> dict:
    """
    Map cluster IDs to human-readable personas by ranking cluster centroids.

    Rules (applied to log-transformed, standardised centroids):
      • VIP       → highest M, highest F, lowest R
      • At-Risk   → high M (past spend), highest R (long lapse)
      • Regular   → moderate M, moderate F
      • Occasional→ lowest M, lowest F
    """
    centre_df = pd.DataFrame(cluster_centers, columns=RFM_FEATURES)
    centre_df["cluster"] = range(K)

    # Rank by Monetary desc, Recency asc
    centre_df = centre_df.sort_values(["M_log", "R_log"], ascending=[False, True])
    personas = ["VIP Customers", "Regular Customers", "Occasional Shoppers", "At-Risk Customers"]

    # Heuristic: highest M & lowest R → VIP; highest M & highest R → At-Risk
    sorted_by_m = centre_df.sort_values("M_log", ascending=False)
    top2 = sorted_by_m.head(2)
    vip_cluster     = int(top2.sort_values("R_log").iloc[0]["cluster"])
    atrisk_cluster  = int(top2.sort_values("R_log", ascending=False).iloc[0]["cluster"])

    bottom2 = sorted_by_m.tail(2)
    regular_cluster   = int(bottom2.sort_values("F_log", ascending=False).iloc[0]["cluster"])
    occasion_cluster  = int(bottom2.sort_values("F_log").iloc[0]["cluster"])

    mapping = {
        vip_cluster:      "VIP Customers",
        regular_cluster:  "Regular Customers",
        occasion_cluster: "Occasional Shoppers",
        atrisk_cluster:   "At-Risk Customers",
    }
    # Fall back if duplicates
    for i in range(K):
        if i not in mapping:
            mapping[i] = personas[i]

    return mapping


def train() -> dict:
    try:
        df = load_clean()
    except Exception:
        df = run_pipeline()

    rfm = build_rfm(df)

    # ── Scale ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[RFM_FEATURES])

    # ── KMeans ─────────────────────────────────────────────────────
    km = KMeans(n_clusters=K, random_state=42, n_init=20, max_iter=500)
    km.fit(X)
    labels = km.labels_

    sil = silhouette_score(X, labels)
    print(f"[clustering] Silhouette Score = {sil:.4f}")

    # ── Persona mapping ────────────────────────────────────────────
    persona_map = _assign_persona(km.cluster_centers_, labels, rfm)
    rfm["Cluster"]      = labels
    rfm["Segment_Label"] = rfm["Cluster"].map(persona_map)

    # ── Save ───────────────────────────────────────────────────────
    rfm.to_parquet(PROCESSED_DIR / "rfm_clusters.parquet", index=False)
    joblib.dump(km,     MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "rfm_scaler.pkl")
    joblib.dump(persona_map, MODELS_DIR / "persona_map.pkl")

    return {"Silhouette": round(sil, 4), "K": K, "PersonaMap": persona_map}


if __name__ == "__main__":
    train()


