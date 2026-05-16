from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def kmeans_model(X, k=5):
    """
    Perform K-Means clustering
    """
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels, model

def apply_pca(X, n_components=2):
    """
    Reduce dimensionality for visualization
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def hierarchical_clustering(X, k=5):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    return labels, model