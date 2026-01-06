from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Veri %{test_size*100} test olacak sekilde bolundu.")
    return X_train, X_test, y_train, y_test
