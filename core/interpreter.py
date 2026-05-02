def interpret_clusters(labels):
    n_clusters = len(set(labels))

    return {
        "num_clusters": n_clusters,
        "message": f"The model identified {n_clusters} distinct patient subgroups."
    }

def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    
    sorted_features = sorted(
        zip(feature_names, importances),
        key=lambda x: -x[1]
    )

    return sorted_features[:10]