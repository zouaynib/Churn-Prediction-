from __future__ import annotations

from sklearn.neural_network import MLPClassifier


def build_model(
    hidden_layer_sizes: tuple = (64, 32),
    activation: str = "relu",
    learning_rate_init: float = 0.001,
    max_iter: int = 500,
    early_stopping: bool = True,
    random_state: int = 42,
) -> MLPClassifier:

    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=0.1,
        random_state=random_state,
    )
