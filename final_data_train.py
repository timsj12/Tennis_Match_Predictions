import s3
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


def test_train_data():

    df = s3.load_file('atp_clean_data.pkl', '/Data')

    y =  df['match_winner'].astype(int)
    X = df.drop(columns=['match_winner'])

    scaler = MinMaxScaler()
    df_transform = scaler.fit_transform(X)
    df_transform = pd.DataFrame(columns=X.columns, data=df_transform)

    X_train, X_test, y_train, y_test = xy_train_test(df_transform, y)

    s3.write_file(X_train, 'X_train.pkl', '/Data')
    s3.write_file(X_test, 'X_test.pkl', '/Data')
    s3.write_file(y_train, 'y_train.pkl', '/Data')
    s3.write_file(y_test, 'y_test.pkl', '/Data')

def xy_train_test(X, y):

    test_size = math.floor(X.shape[0] * .05)

    random_rows = np.random.choice(len(X), test_size, replace=False)

    X_train = pd.DataFrame(np.delete(X, random_rows, axis=0))
    X_train.columns = X.columns

    X_test = X.iloc[random_rows]
    X_test.columns = X.columns

    y_train = np.delete(y, random_rows, axis=0)
    y_train = y_train.ravel()

    y_test = y.iloc[random_rows]
    y_test = y_test.ravel()

    return X_train, X_test, y_train, y_test

