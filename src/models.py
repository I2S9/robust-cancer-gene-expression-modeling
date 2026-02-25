from sklearn.linear_model import LogisticRegression


def make_logreg_l2(seed: int = 42) -> LogisticRegression:
    """Build a baseline Logistic Regression model with L2 regularization."""
    return LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    )


def make_logreg_l1(seed: int = 42, c: float = 1.0) -> LogisticRegression:
    """Build a sparse Logistic Regression baseline with L1 regularization."""
    return LogisticRegression(
        penalty="l1",
        solver="saga",
        l1_ratio=1.0,
        C=c,
        max_iter=200,
        tol=1e-2,
        random_state=seed,
    )
