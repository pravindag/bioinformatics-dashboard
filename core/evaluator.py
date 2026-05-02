from sklearn.metrics import silhouette_score

def evaluate_clustering(X, labels):
    score = silhouette_score(X, labels)
    return score

from sklearn.metrics import accuracy_score, f1_score

def evaluate_classification(model, X, y):
    preds = model.predict(X)

    return {
        "accuracy": accuracy_score(y, preds),
        "f1_score": f1_score(y, preds, average='weighted')
    }

from sklearn.metrics import adjusted_rand_score

def compare_with_labels(cluster_labels, true_labels):
    return adjusted_rand_score(true_labels, cluster_labels)