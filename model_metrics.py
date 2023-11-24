import s3
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time
import ML_Models


def compare_models():

    df = s3.load_file('atp_clean_data.pkl', '/Data')

    y =  df['match_winner'].astype(int)
    X = df.drop(columns=['match_winner'])

    scaler = MinMaxScaler()
    df_transform = scaler.fit_transform(X)
    df_transform = pd.DataFrame(columns=X.columns, data=df_transform)

    kf = KFold(n_splits=5, shuffle=True)

    metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Time'])

    for ml_model in ML_Models.getModels():
        precision = []
        recall = []
        f1 = []
        accuracy = []

        start =time.time()
        for train_index, test_index in kf.split(df_transform):
            X_train, X_test = df_transform.iloc[train_index], df_transform.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            ml_model.model.fit(X_train, y_train)

            y_model = ml_model.model.predict(X_test)

        # Calculate Average Precision, Recall, F1 and Accuracy
            precision.append(precision_score(y_test, y_model))
            recall.append(recall_score(y_test, y_model))
            f1.append(f1_score(y_test, y_model))
            accuracy.append(ml_model.model.score(X_test, y_test))

        time_spent = time.time() - start

        ml_model.precision = np.mean(precision)
        ml_model.recall = np.mean(recall)
        ml_model.f1 = np.mean(f1)
        ml_model.accuracy = np.mean(accuracy)

        metrics_df.loc[len(metrics_df)] = [ml_model.name, ml_model.accuracy, ml_model.precision, ml_model.recall,
                                           ml_model.f1, time_spent]

        print(ml_model.name)
        print("Accuracy:", ml_model.accuracy)
        print("Precision: ", ml_model.precision)
        print("Recall: ", ml_model.recall)
        print("F1: ", ml_model.f1)
        print("Time Spent: ", time_spent)
        print("-----------------------------------------------")

    metrics_df.sort_values(by='Accuracy', ascending=False, inplace=True)

    s3.write_file(metrics_df, 'atp_ml_model_metrics.pkl', '/Metrics')