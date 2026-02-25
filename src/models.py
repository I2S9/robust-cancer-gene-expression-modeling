from sklearn.linear_model import LogisticRegression


def make_logreg_l2(seed: int = 42) -> LogisticRegression:
    """Build a baseline Logistic Regression model with L2 regularization."""
    return LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )
