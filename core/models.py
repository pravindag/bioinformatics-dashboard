from sklearn.cluster import KMeans

def kmeans_model(X, k=5):
    """
    Perform K-Means clustering
    """
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels, model

from sklearn.decomposition import PCA

def apply_pca(X, n_components=2):
    """
    Reduce dimensionality for visualization
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

from sklearn.ensemble import RandomForestClassifier

def random_forest_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering(X, k=5):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    return labels, model