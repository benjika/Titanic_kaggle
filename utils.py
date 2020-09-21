from sklearn.preprocessing import OrdinalEncoder


def get_title(row):
    return row['Name'].split(', ', 1)


def prepare_input(X_train_column, X_test_coumn):
    oe = OrdinalEncoder()
