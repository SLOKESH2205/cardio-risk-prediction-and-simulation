"""Patient segmentation module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger
from src.utils import ensure_directory, load_joblib, save_joblib


LOGGER = get_logger(__name__)


class PatientSegmenter:
    """Fit patient clusters and profile segment characteristics."""

    CLUSTERING_FEATURES = [
        "age_years",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "pulse_pressure",
        "cholesterol_raw",
        "glucose_raw",
        "lifestyle_risk_score",
    ]

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize segmentation paths.

        Args:
            base_dir: Optional project root path.

        Returns:
            None.
        """
        self.base_dir = base_dir or Path.cwd()
        self.models_dir = ensure_directory(self.base_dir / "models")
        self.outputs_dir = ensure_directory(self.base_dir / "outputs")
        self.cluster_profiles: dict[int, dict[str, Any]] = {}

    def find_optimal_k(self, df: pd.DataFrame, k_range: range = range(2, 8)) -> int:
        """Find recommended cluster count using inertia and silhouette.

        Args:
            df: Feature-engineered dataframe.
            k_range: Candidate cluster counts.

        Returns:
            Recommended cluster count.
        """
        clean_df = df.dropna(subset=self.CLUSTERING_FEATURES).copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clean_df[self.CLUSTERING_FEATURES])
        inertia_values: list[float] = []
        silhouette_values: list[float] = []

        for k in k_range:
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X_scaled)
            inertia_values.append(float(model.inertia_))
            silhouette = float(silhouette_score(X_scaled, labels))
            silhouette_values.append(silhouette)
            LOGGER.info("k=%s | silhouette=%.4f", k, silhouette)

        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), inertia_values, marker="o")
        plt.title("Elbow Plot")
        plt.xlabel("k")
        plt.ylabel("Inertia")
        plt.tight_layout()
        plt.savefig(self.outputs_dir / "elbow_plot.png", dpi=200)
        plt.close()

        best_index = int(pd.Series(silhouette_values).idxmax())
        return list(k_range)[best_index]

    def fit(self, df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """Fit KMeans model and save segmentation artifact.

        Args:
            df: Feature-engineered dataframe.
            k: Number of clusters.

        Returns:
            Dataframe with cluster assignments.
        """
        frame = df.dropna(subset=self.CLUSTERING_FEATURES).copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(frame[self.CLUSTERING_FEATURES])
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        frame["cluster"] = kmeans.fit_predict(X_scaled)
        silhouette = float(silhouette_score(X_scaled, frame["cluster"]))
        LOGGER.info("Segmentation silhouette score: %.4f", silhouette)

        intra_cluster_distances: dict[int, float] = {}
        distances = kmeans.transform(X_scaled)
        for cluster_id in sorted(frame["cluster"].unique()):
            mask = frame["cluster"] == cluster_id
            intra_cluster_distances[int(cluster_id)] = float(distances[mask, int(cluster_id)].mean())

        save_joblib(
            self.models_dir / "segmentation.pkl",
            {
                "scaler": scaler,
                "kmeans": kmeans,
                "features": self.CLUSTERING_FEATURES,
                "cluster_profiles": self.cluster_profiles,
                "mean_distances": intra_cluster_distances,
            },
        )
        return frame

    def profile_clusters(self, df: pd.DataFrame) -> dict[int, dict[str, Any]]:
        """Profile each cluster and save summary CSV.

        Args:
            df: Segmented dataframe.

        Returns:
            Cluster metadata dictionary.
        """
        profiles: list[dict[str, Any]] = []
        cluster_dict: dict[int, dict[str, Any]] = {}
        total_rows = len(df)

        for cluster_id, cluster_df in df.groupby("cluster"):
            target_rate = float(cluster_df["target"].mean())
            if target_rate > 0.60:
                label = "High-Risk Metabolic"
            elif target_rate < 0.30:
                label = "Low-Risk Active"
            else:
                label = "Moderate-Risk Lifestyle"

            dominant_bp = int(cluster_df["bp_category"].mode().iloc[0])
            description = (
                f"Mean age {cluster_df['age_years'].mean():.1f}, "
                f"BMI {cluster_df['bmi'].mean():.1f}, "
                f"systolic BP {cluster_df['systolic_bp'].mean():.1f}, "
                f"pulse pressure {cluster_df['pulse_pressure'].mean():.1f}."
            )
            profile_row = {
                "cluster": int(cluster_id),
                "label": label,
                "size_n": int(len(cluster_df)),
                "size_pct": round(len(cluster_df) / total_rows * 100, 2),
                "age_years_mean": round(cluster_df["age_years"].mean(), 2),
                "bmi_mean": round(cluster_df["bmi"].mean(), 2),
                "systolic_bp_mean": round(cluster_df["systolic_bp"].mean(), 2),
                "pulse_pressure_mean": round(cluster_df["pulse_pressure"].mean(), 2),
                "lifestyle_risk_score_mean": round(cluster_df["lifestyle_risk_score"].mean(), 2),
                "cholesterol_raw_mean": round(cluster_df["cholesterol_raw"].mean(), 2),
                "target_rate": round(target_rate, 4),
                "dominant_bp_category": dominant_bp,
                "description": description,
            }
            LOGGER.info("Cluster %s profile: %s", cluster_id, profile_row)
            profiles.append(profile_row)
            cluster_dict[int(cluster_id)] = {
                "label": label,
                "target_rate": target_rate,
                "description": description,
            }

        pd.DataFrame(profiles).to_csv(self.outputs_dir / "cluster_profiles.csv", index=False)
        self.cluster_profiles = cluster_dict

        artifact = load_joblib(self.models_dir / "segmentation.pkl")
        artifact["cluster_profiles"] = cluster_dict
        save_joblib(self.models_dir / "segmentation.pkl", artifact)
        return cluster_dict

    def predict_cluster(self, patient_df: pd.DataFrame) -> tuple[int, str]:
        """Predict cluster assignment and drift warning for a patient.

        Args:
            patient_df: Single-row patient dataframe.

        Returns:
            Tuple of cluster id and label string.
        """
        artifact = load_joblib(self.models_dir / "segmentation.pkl")
        scaler: StandardScaler = artifact["scaler"]
        kmeans: KMeans = artifact["kmeans"]
        features = artifact["features"]
        cluster_profiles = artifact.get("cluster_profiles", {})
        mean_distances = artifact.get("mean_distances", {})

        scaled = scaler.transform(patient_df[features])
        cluster_id = int(kmeans.predict(scaled)[0])
        centroid_distance = float(kmeans.transform(scaled)[0, cluster_id])
        baseline_distance = float(mean_distances.get(cluster_id, 0.0))
        label = cluster_profiles.get(cluster_id, {}).get("label", f"Cluster {cluster_id}")
        if baseline_distance and centroid_distance > 1.5 * baseline_distance:
            label = f"{label} | Note: this patient's profile is atypical for their cluster."
        return cluster_id, label



def describe_cluster(cluster_id: int) -> str:
    """Return a default cluster description for a numeric cluster id."""
    descriptions = {
        0: "Moderate-risk lifestyle group with mid-range blood pressure and manageable metabolic burden.",
        1: "Low-risk active group with healthier blood pressure and stronger lifestyle indicators.",
        2: "High-risk metabolic group with elevated blood pressure and cholesterol burden.",
    }
    return descriptions.get(cluster_id, "Unknown cluster profile.")

if __name__ == "__main__":
    demo_path = Path.cwd() / "data" / "processed" / "harmonized.csv"
    if demo_path.exists():
        demo_df = pd.read_csv(demo_path)
        from src.features import FeatureEngineer

        engineered = FeatureEngineer().engineer(demo_df)
        segmenter = PatientSegmenter()
        best_k = segmenter.find_optimal_k(engineered)
        segmented = segmenter.fit(engineered, k=best_k)
        segmenter.profile_clusters(segmented)
    else:
        LOGGER.info("Run ingest.py first to create harmonized.csv.")


