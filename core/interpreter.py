def interpret_clusters(labels):
    n_clusters = len(set(labels))

    message = f"""
    The model identified {n_clusters} distinct patient clusters.

    These clusters may correspond to different molecular subtypes of breast cancer.
    
    For example:
    - Cluster patterns may align with known subtypes such as Luminal A, Luminal B, HER2-enriched, or Basal-like.
    
    Further validation with clinical labels (e.g., PAM50) is required.
    """

    return {"message": message}

def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    
    sorted_features = sorted(
        zip(feature_names, importances),
        key=lambda x: -x[1]
    )

    return sorted_features[:10]