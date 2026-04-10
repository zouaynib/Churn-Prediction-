from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def build_model(random_state: int = 42) -> AdaBoostClassifier:
    base = DecisionTreeClassifier(
        max_depth=3,
        class_weight="balanced",
        random_state=random_state,
    )
    model = AdaBoostClassifier(
        estimator=base,
        n_estimators=300,
        learning_rate=0.5,
        algorithm="SAMME",
        random_state=random_state,
    )

    return model
