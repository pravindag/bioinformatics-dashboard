from sklearn.feature_selection import VarianceThreshold

def select_features(X, k=2000):
    selector = VarianceThreshold()
    X_var = selector.fit_transform(X)

    if X_var.shape[1] > k:
        return X_var[:, :k]

    return X_var