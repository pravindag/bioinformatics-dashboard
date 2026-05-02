import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_pca_clusters(X_reduced, labels):
    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=labels,
        cmap="viridis"
    )

    ax.set_title("PCA Cluster Visualization")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # 🔥 Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster Label")

    return fig


def plot_with_labels(X_reduced, labels, true_labels=None):
    fig, ax = plt.subplots()

    if true_labels is not None:
        scatter = ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=true_labels,
            cmap="plasma"
        )
        ax.set_title("True Labels Visualization")
        label_name = "True Label"
    else:
        scatter = ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=labels,
            cmap="viridis"
        )
        ax.set_title("Cluster Visualization")
        label_name = "Cluster Label"

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # 🔥 Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label_name)

    return fig

def plot_tsne_clusters(X, labels):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=labels,
        cmap="viridis"
    )

    ax.set_title("t-SNE Cluster Visualization")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

    return fig