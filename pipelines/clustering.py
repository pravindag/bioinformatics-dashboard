from core.models import kmeans_model, apply_pca, hierarchical_clustering
from core.evaluator import evaluate_clustering

def clustering_pipeline(X, model_type="kmeans", k=5):

    X_reduced, _ = apply_pca(X)

    if model_type == "kmeans":
        labels, model = kmeans_model(X_reduced, k)
    elif model_type == "hierarchical":
        labels, model = hierarchical_clustering(X_reduced, k)
    else:
        raise ValueError("Unsupported model type")

    score = evaluate_clustering(X_reduced, labels)

    return {
        "labels": labels,
        "score": score,
        "X_reduced": X_reduced
    }