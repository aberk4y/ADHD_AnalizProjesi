from sklearn.ensemble import GradientBoostingClassifier

def get_model():
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    return model
