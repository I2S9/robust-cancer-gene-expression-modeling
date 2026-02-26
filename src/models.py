from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
        solver="saga",
        l1_ratio=1.0,
        C=c,
        max_iter=200,
        tol=1e-2,
        random_state=seed,
    )


def make_linear_svm(seed: int = 42, c: float = 1.0) -> LinearSVC:
    """Build a linear SVM baseline classifier."""
    return LinearSVC(C=c, random_state=seed, max_iter=5000)


def make_random_forest(seed: int = 42, n_estimators: int = 60) -> RandomForestClassifier:
    """Build a random forest baseline classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        random_state=seed,
        n_jobs=-1,
    )


def make_gradient_boosting(seed: int = 42) -> GradientBoostingClassifier:
    """Build a gradient boosting baseline classifier."""
    return GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=2,
        random_state=seed,
    )
