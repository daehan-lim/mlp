def x_y_split(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset['mortality'].values
    return X, y
