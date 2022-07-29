import xlwings as xw
import pandas as pd
import numpy as np
from Info import Name
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import sklearn.externals
import joblib
from sklearn.metrics import confusion_matrix


def ready_data():
    excel = xw.Book(r'rawdata.xlsx')

    columns_num = len(Name.full_name)

    df_total = pd.DataFrame()

    for i in range(columns_num):
        df = excel.sheets(1).range(1, 3 * i + 1).options(pd.DataFrame,
                                                         index=True,
                                                         expand='table',
                                                         header=False).value
        df.columns = [df.iloc[0, 0]]
        df.index.name = df.index[1]
        df = df.iloc[2:, :]

        df_total = pd.concat([df_total, df], axis=1)

    df_total.dropna(inplace=True)

    df_total.sort_index(ascending=True, inplace=True)

    data_num = len(df_total)
    sig_3Y = [np.nan]
    sig_10Y = [np.nan]

    for i in range(1, data_num):
        if df_total['국채 3Y'].diff(1)[i] < -0.1:
            sig_3Y.append("하락")
        elif (df_total['국채 3Y'].diff(1)[i] > -0.1) and (df_total['국채 3Y'].diff(1)[i] < 0.1):
            sig_3Y.append("보합")
        elif df_total['국채 3Y'].diff(1)[i] > 0.1:
            sig_3Y.append("상승")

    for i in range(data_num):
        if df_total['국채 10Y'].diff(1)[i] < -0.1:
            sig_10Y.append("하락")
        elif (df_total['국채 10Y'].diff(1)[i] > -0.1) and (df_total['국채 10Y'].diff(1)[i] < 0.1):
            sig_10Y.append("보합")
        elif df_total['국채 10Y'].diff(1)[i] > 0.1:
            sig_10Y.append("상승")

    df_total['sig 3Y'] = sig_3Y
    df_total['sig 10Y'] = sig_10Y

    return df_total


if __name__ == "__main__":
    df = ready_data()
    df = df.iloc[1:, :]

    x = df[Name.X_name]
    y_3Y = df['sig 3Y']
    y_10Y = df['sig 10Y']


    # 1. 데이터 분할
    x_3Y_train, x_3Y_test, y_3Y_train, y_3Y_test = train_test_split(x, y_3Y,
                                                                    test_size=0.2,
                                                                    stratify=y_3Y,
                                                                    random_state=42)

    # x_10Y_train, x_10Y_test, y_10Y_train, y_10Y_test = train_test_split(x,
    #                                                                     y_10Y,
    #                                                                     test_size=0.2,
    #                                                                     stratify=y_10Y,
    #                                                                     random_state=42)



    # 2. 최적 하이퍼 파라미터 결정
    rnd_clf = RandomForestClassifier()

    param_dist_rf = {
        'n_estimators': [50, 100, 500],
        'max_leaf_nodes': [20, 30, 40, 50],
        'max_features': [2, 4, 6, 8]
    }

    rnd_search = RandomizedSearchCV(rnd_clf, param_dist_rf, cv=10, random_state=42)
    rnd_search.fit(x_3Y_train, y_3Y_train)
    n_estimators = rnd_search.best_params_['n_estimators']
    max_leaf_nodes = rnd_search.best_params_['max_leaf_nodes']
    max_features = rnd_search.best_params_['max_features']



    # 3. 학습 및 K-fold cross validation 평가
    rnd_clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,
                                     max_features=max_features, n_jobs=-1,
                                     random_state=42)
    rnd_scores = cross_val_score(rnd_clf, x_3Y_train, y_3Y_train, scoring="accuracy", cv=20)
    print("\n<10-fold cross-validation>")
    print("accuracy score mean: ", rnd_scores.mean())



    # 4. 최종 모델 학습
    rnd_clf.fit(x_3Y_train, y_3Y_train)
    print("\n<AI model: machine learning done >")
    print("accuracy_score of train data(0.8 of sample): ", rnd_clf.score(x_3Y_train, y_3Y_train))



    # 5. test data 확인
    print("accuracy_score of test data(0.2 of sample): ", rnd_clf.score(x_3Y_test, y_3Y_test))
    y_3Y_test_pred = rnd_clf.predict(x_3Y_test)
    print("accuracy_score of test data: ", mt.accuracy_score(y_3Y_test, y_3Y_test_pred))



    # 6. confusion matrix 확인
    cm1 = confusion_matrix(y_3Y_test, y_3Y_test_pred, labels=["상승", "보합", "하락"])
    print("\n<Confusion matrix>")
    print("(of test)")
    print("상승", "보합", "하락")
    print(cm1)
    cm2 = confusion_matrix(y_3Y, rnd_clf.predict(x), labels=["상승", "보합", "하락"])
    print("(of all)")
    print("상승", "보합", "하락")
    print(cm2)



    # 7. 변수 중요도 체크
    print("\n<Feature importance>")
    for name, score in zip(x.columns, rnd_clf.feature_importances_):
        print(name, ": ", score)