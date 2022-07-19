import time
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import cross_validate, KFold, RandomizedSearchCV, RepeatedStratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1010)


def scores(model):
    model_predicts = model.predict(test_x)
    return accuracy_score(test_y, model_predicts), recall_score(test_y, model_predicts), \
           precision_score(test_y, model_predicts), f1_score(test_y, model_predicts), ((accuracy_score(test_y, model_predicts) + recall_score(test_y, model_predicts) + \
           precision_score(test_y, model_predicts) + f1_score(test_y, model_predicts))/4)


def all_scores(predicts, test_y):
    print(f" Accuracy Score: {accuracy_score(test_y, predicts):.3f}\n"
          f" Recall Score: {recall_score(test_y, predicts):.3f}\n"
          f" Precision Score: {precision_score(test_y, predicts):.3f}\n"
          f" F1 Score: {f1_score(test_y, predicts):.3f}\n")


def pd_profiling_report(data):
    data_profile = ProfileReport(data)
    data_profile.to_file('pandas_profiling Reports/data_report.html')


def train_test_spliting(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y)
    return train_x, test_x, train_y, test_y


def randomForest_model():
    time_on_creation = time.time()
    print('Creating Random Forest Model...')
    random_forest_classifier = RandomForestClassifier(bootstrap=False, max_depth=86, min_samples_leaf=2,
                                                      min_samples_split=10, n_estimators=10)
    print('Fitting data into model for training...')
    random_forest_classifier.fit(train_x, train_y)
    print('Predicting on test data.')
    rfc_predicts = random_forest_classifier.predict(test_x)
    time_to_finish = time.time() - time_on_creation
    print(f'Done!\n'
          f'Time to create, train model and predict: {time_to_finish:.2f} seconds\n'
          f'Overall Scores:')
    all_scores(rfc_predicts, test_y)
    return random_forest_classifier


def logisticRegression_model():
    time_on_creation = time.time()
    print('Creating Logistic Regression Model...')
    log_regression = LogisticRegression(max_iter=20000, penalty='l2', solver='newton-cg')
    print('Fitting data into model for training...')
    log_regression.fit(train_x, train_y)
    print('Predicting on test data.')
    predicts_lr = log_regression.predict(test_x)
    time_to_finish = time.time() - time_on_creation
    print(f'Done!\n'
          f'Time to create, train model and predict: {time_to_finish:.2f} seconds\n'
          f'Overall Scores:')
    all_scores(predicts_lr, test_y)
    return log_regression


def multilayerPerceptron_model():
    time_on_creation = time.time()
    print('Creating MultiLayer Perceptron Model...')
    mlp_classifier = MLPClassifier()
    print('Fitting data into model for training...')
    mlp_classifier.fit(train_x, train_y)
    print('Predicting on test data.')
    predicts_mlp = mlp_classifier.predict(test_x)
    time_to_finish = time.time() - time_on_creation
    print(f'Done!\n'
          f'Time to create, train model and predict: {time_to_finish:.2f} seconds\n'
          f'Overall Scores:')
    all_scores(predicts_mlp, test_y)
    return mlp_classifier

def scores_modelos():
    model_scores = {scores(model_rfc), scores(model_lg), scores(model_mlp)}
    df_model_scores = pd.DataFrame(model_scores,
                                   columns=['Accuracy Score', 'Recall Score', 'Precision Score', 'F1 Score','Scores Mean'],
                                   index=['Random Forest', 'Logistic Regression', 'Multilayer Perceptron'])
    pd.set_option('display.max_columns', 5)
    print(df_model_scores)
    pd.reset_option('display.max_columns')


data = pd.read_csv('dataset_customer_churn.csv', sep='^')
is_NAN = data[data.isna().any(axis=1)]

data.drop(labels=['A006_REGISTRO_ANS', 'CODIGO_BENEFICIARIO', 'CLIENTE', 'CD_USUARIO', 'CODIGO_FORMA_PGTO_MENSALIDADE',
                  'A006_NM_PLANO', 'DIAS_ATE_REALIZAR_ALTO_CUSTO', 'CD_ASSOCIADO', 'ESTADO_CIVIL'], axis=1,
          inplace=True)
data.drop(is_NAN.index, axis=0, inplace=True)
data.drop(labels=182212, axis=0, inplace=True)

dict_replace = {
    "SIM": 1,
    "NAO": 0,
    'F': 0,
    'M': 1,
    'DESATIVADO': 1,
    'ATIVO': 0,

}
data.replace(dict_replace, inplace=True)

data_dummified = pd.get_dummies(data)

x = data_dummified.drop(labels=['SITUACAO'], axis=1)
y = data_dummified['SITUACAO']
train_x, test_x, train_y, test_y = train_test_spliting(x, y)

model_rfc = randomForest_model()
model_lg = logisticRegression_model()
model_mlp = multilayerPerceptron_model()

if __name__ == '__main__':
    pd_profiling_report(data)
    scores_modelos()
