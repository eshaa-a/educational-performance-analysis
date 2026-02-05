# pipeline.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid", context="talk")

DATA_DIR = "data"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    X = pd.read_csv(os.path.join(DATA_DIR, "learner_skill_matrix.csv"), index_col=0)
    jobs = pd.read_csv(os.path.join(DATA_DIR, "job_skill_vectors.csv"), index_col=0)
    syllabus = pd.read_csv(os.path.join(DATA_DIR, "syllabus.csv"))
    return X, jobs, syllabus

def run_pca(X, n_components=0.90):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
    Xp = pca.fit_transform(Xs)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns,
                             columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    return scaler, pca, Xp, loadings

def cluster(Xp, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labs = km.fit_predict(Xp)
    sil = silhouette_score(Xp, labs)
    return km, labs, sil

def compute_job_distances(X, jobs):
    # X: raw learner matrix, jobs: role x skills
    # ensure same columns/order
    jobs = jobs[X.columns]
    dists = cosine_distances(X.values, jobs.values)
    dist_df = pd.DataFrame(dists, index=X.index, columns=jobs.index)
    return dist_df

def plot_scree(pca, out_path):
    ratios = pca.explained_variance_ratio_
    cum = np.cumsum(ratios)
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(1, len(ratios)+1), ratios, marker="o", label="Individual")
    plt.plot(np.arange(1, len(ratios)+1), cum, marker="s", label="Cumulative")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.legend()
    plt.title("Scree plot")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_loadings(loadings, out_path, top_k=10):
    # heatmap of loadings
    plt.figure(figsize=(10, max(6, loadings.shape[1]*0.5)))
    sns.heatmap(loadings.abs().sort_values(by=loadings.columns[0], ascending=False).T,
                cmap="vlag", center=0, cbar_kws={"label":"abs(loading)"})
    plt.title("Component loadings (absolute)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    X, jobs, syllabus = load_data()
    scaler, pca, Xp, loadings = run_pca(X, n_components=0.90)
    km, labs, sil = cluster(Xp, n_clusters=4)
    print("PCA components:", pca.n_components_)
    print("Silhouette score (PC space):", sil)

    dist_df = compute_job_distances(X, jobs)
    best_role = dist_df.idxmin(axis=1)
    best_dist = dist_df.min(axis=1)

    out = X.copy()
    out["cluster"] = labs
    out["best_role"] = best_role
    out["best_dist"] = best_dist
    out.to_csv(os.path.join(OUT_DIR, "learner_profiles_with_clusters.csv"))

    # save PCA components & loadings
    loadings.to_csv(os.path.join(OUT_DIR, "pca_loadings.csv"))
    pd.DataFrame(pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(pca.n_components_)],
                 columns=["explained_variance_ratio"]).to_csv(os.path.join(OUT_DIR, "pca_explained_variance.csv"))

    # save distances
    dist_df.to_csv(os.path.join(OUT_DIR, "learner_role_distances.csv"))

    # plots
    plot_scree(pca, os.path.join(OUT_DIR, "scree_plot.png"))
    plot_loadings(loadings, os.path.join(OUT_DIR, "loadings_heatmap.png"))

    # cluster profiles
    cluster_profiles = X.groupby(labs).mean()
    cluster_profiles.to_csv(os.path.join(OUT_DIR, "cluster_profiles.csv"))

    print("Saved outputs to", OUT_DIR)

if __name__ == "__main__":
    main()
