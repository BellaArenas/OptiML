from sklearn.model_selection import KFold, RepeatedKFold

# Set up K-Fold cross-validation
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Global trained model
trained_model = None

baseline = 0