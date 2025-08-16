import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------
# Utility / IO
# ---------------------

def load_default_dataset():
    path = os.path.join("data", "Mall_Customers.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Default dataset Mall_Customers.csv not found in data/ folder.")
    df = pd.read_csv(path)
    return df

def load_csv_dataset(path):
    return pd.read_csv(path)

def preprocess_dataset(df):
    # Drop obvious ID-like columns
    to_drop = [c for c in df.columns if c.lower().startswith("id")]
    df = df.drop(columns=to_drop, errors="ignore")

    # Handle missing
    df = df.fillna(df.mean(numeric_only=True))

    return df

# ---------------------
# Plotting
# ---------------------

def plot_elbow(X, out_path):
    distortions = []
    K = range(1, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.figure()
    plt.plot(K, distortions, "bx-")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Distortion (Inertia)")
    plt.title("Elbow Method for Optimal K")
    plt.savefig(out_path)
    plt.close()

def plot_clusters(X, labels, out_path):
    plt.figure()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="Set1", s=50)
    plt.title("Cluster Visualization")
    plt.savefig(out_path)
    plt.close()

def plot_silhouette(X, labels, out_path):
    n_clusters = len(np.unique(labels))
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    fig, ax = plt.subplots()
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette Plot (avg={silhouette_avg:.2f})")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.savefig(out_path)
    plt.close()

# ---------------------
# Training / Evaluation
# ---------------------

def train_and_evaluate(df, n_clusters=3, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # Select features
    if {"Age", "Annual Income (k$)", "Spending Score (1-100)"}.issubset(df.columns):
        features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
        X = df[features].values
    else:
        X = df.select_dtypes(include=[np.number]).values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow
    plot_elbow(X_scaled, os.path.join(out_dir, "elbow_method.png"))

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # PCA for visualization (if needed)
    if X_scaled.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_vis = pca.fit_transform(X_scaled)
    else:
        X_vis = X_scaled

    # Plots
    plot_clusters(X_vis, labels, os.path.join(out_dir, "clusters.png"))
    plot_silhouette(X_scaled, labels, os.path.join(out_dir, "silhouette.png"))

    # Metrics
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "cluster_centers": kmeans.cluster_centers_.tolist()
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("Done. Metrics saved to outputs/metrics.json")

# ---------------------
# Main
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters for KMeans")
    args = parser.parse_args()

    if args.csv:
        df = load_csv_dataset(args.csv)
    else:
        df = load_default_dataset()

    df = preprocess_dataset(df)
    train_and_evaluate(df, args.clusters)

if __name__ == "__main__":
    main()