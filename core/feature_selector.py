from sklearn.feature_selection import VarianceThreshold

def select_features(X, k):
    selector = VarianceThreshold()
    X_var = selector.fit_transform(X)

    # select top k features by variance
    variances = X_var.var(axis=0)
    top_k_idx = variances.argsort()[-k:]

    return X_var[:, top_k_idx]