from sklearn.ensemble import RandomForestClassifier

def build_model_RF(random_state : int=42) :
    model = RandomForestClassifier(
        n_estimators = 300,
        criterion = "gini",
        max_depth = 8,
        max_features = "sqrt",
        max_samples = 0.8
        class_weight = "balanced",
        n_jobs = -1,
        min_samples_split = 10,
        min_samples_leaf = 5,
        bootstrap = True,
        oob_score = True,
        random_state = random_state

    )


    return model