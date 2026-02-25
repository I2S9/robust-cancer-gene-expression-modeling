from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


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
        C=c,
        max_iter=200,
        tol=1e-2,
        random_state=seed,
    )


def make_linear_svm(seed: int = 42, c: float = 1.0) -> LinearSVC:
    """Build a linear SVM baseline classifier."""
    return LinearSVC(C=c, random_state=seed, max_iter=5000)
