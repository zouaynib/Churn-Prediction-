from sklearn.ensemble import AdaBoostClassifier


def build_model(random_state: int = 42) -> AdaBoostClassifier:
    model = AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.5,
        algorithm="SAMME",
        random_state=random_state,
    )

    return model
