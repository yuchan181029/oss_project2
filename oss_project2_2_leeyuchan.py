import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def sort_dataset(dataset_df):
    dataset_df = dataset_df.sort_values(by='year', ascending=True)
    return dataset_df

# TODO: Implement this function

def split_dataset(dataset_df):
    dataset_df['salary'] *= 0.001

    X_train = dataset_df[:1718].drop(columns="salary", axis=1)
    Y_train = dataset_df[:1718]['salary']

    X_test = dataset_df[1718:].drop(columns="salary", axis=1)
    Y_test = dataset_df[1718:]['salary']

    return X_train, X_test, Y_train, Y_test

# TODO: Implement this function

def extract_numerical_cols(dataset_df):
    numerical_df = dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]

    return numerical_df

# TODO: Implement this function

def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_cls = DecisionTreeRegressor()
    dt_cls.fit(X_train, Y_train)

    return dt_cls.predict(X_test)

# TODO: Implement this function

def train_predict_random_forest(X_train, Y_train, X_test):
    rf_cls = RandomForestRegressor()
    rf_cls.fit(X_train, Y_train)

    return rf_cls.predict(X_test)

# TODO: Implement this function

def train_predict_svm(X_train, Y_train, X_test):
    svm_pipe = make_pipeline(
        StandardScaler(),
        SVR()
    )
    svm_pipe.fit(X_train, Y_train)

    return svm_pipe.predict(X_test)

# TODO: Implement this function

def calculate_RMSE(labels, predictions):
    return np.sqrt(np.mean((predictions - labels) ** 2))

# TODO: Implement this function

if __name__ == '__main__':
    # DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))