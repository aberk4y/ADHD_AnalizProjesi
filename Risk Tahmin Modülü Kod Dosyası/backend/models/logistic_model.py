from sklearn.linear_model import LogisticRegression

def get_model():
    model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs"
    )
    return model
