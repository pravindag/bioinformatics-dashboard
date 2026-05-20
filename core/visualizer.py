import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd

# =========================
# PCA VISUALIZATION
# =========================
def plot_with_labels(X_reduced, labels, true_labels=None):

    fig, ax = plt.subplots(figsize=(8, 6))

    if true_labels is not None:

        scatter = ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=true_labels,
            cmap="plasma",
            alpha=0.7,
            s=40
        )

        ax.set_title("True Labels Visualization")
        label_name = "True Label"

    else:

        scatter = ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.7,
            s=40
        )

        ax.set_title("Cluster Visualization")
        label_name = "Cluster Label"

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label_name)

    ax.grid(True)

    return fig

# =========================
# t-SNE VISUALIZATION
# =========================
def plot_tsne_clusters(X, labels):

    perplexity = min(30, max(5, len(X)//3))

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity
    )

    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
        s=40
    )

    ax.set_title("t-SNE Cluster Visualization")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

    ax.grid(True)

    return fig

# =========================
# SURVIVAL ANALYSIS
# =========================
def plot_survival(time, event, labels):

    kmf = KaplanMeierFitter()

    time = pd.to_numeric(
        time,
        errors="coerce"
    )

    event = pd.to_numeric(
        event,
        errors="coerce"
    )

    labels = np.array(labels)

    valid = ~np.isnan(time)

    time = time[valid]
    event = event[valid]
    labels = labels[valid]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for cluster in np.unique(labels):

        idx = labels == cluster

        kmf.fit(

            time[idx],

            event_observed=event[idx],

            label=f"Cluster {cluster}"

        )

        kmf.plot(ax=ax)

    ax.set_title(
        "Kaplan-Meier Survival Analysis "
        "by Patient Cluster"
    )

    ax.set_xlabel("Time (Days)")

    ax.set_ylabel(
        "Survival Probability"
    )

    ax.grid(True)

    ax.legend()

    return fig