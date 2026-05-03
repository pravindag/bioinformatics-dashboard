def interpret_clusters(labels):
    n_clusters = len(set(labels))

    message = f"""
            The model identified {n_clusters} clusters.

            These clusters may correspond to molecular subtypes such as:
            - Luminal A
            - Luminal B
            - HER2-enriched
            - Basal-like

            Further validation with PAM50 labels is required.
            """

    return {"message": message}

def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    
    sorted_features = sorted(
        zip(feature_names, importances),
        key=lambda x: -x[1]
    )

    return sorted_features[:10]