import time
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from scipy.stats import randint
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import cross_validate, KFold, RandomizedSearchCV, RepeatedStratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(1010)

MODELS = []


def scores(model):
    model_predicts = model.predict(test_x)
    return accuracy_score(test_y, model_predicts), recall_score(test_y, model_predicts), \
           precision_score(test_y, model_predicts), f1_score(test_y, model_predicts), \
           ((accuracy_score(test_y, model_predicts) + recall_score(test_y, model_predicts) + \
             precision_score(test_y, model_predicts) + f1_score(test_y, model_predicts)) / 4)


def all_scores(predicts, test_y):
    print(f" Accuracy Score: {accuracy_score(test_y, predicts):.3f}\n"
          f" Recall Score: {recall_score(test_y, predicts):.3f}\n"
          f" Precision Score: {precision_score(test_y, predicts):.3f}\n"
          f" F1 Score: {f1_score(test_y, predicts):.3f}\n")


def pd_profiling_report(data):
    data_profile = ProfileReport(data)
    data_profile.to_file('pandas_profiling Reports/data_report.html')


def train_test_spliting(x, y):
    x = data_dummified.drop(labels=['SITUACAO'], axis=1)
    y = data_dummified['SITUACAO']
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
          f'Time to create, train model and predict: {time_to_finish:.2f} seconds')
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
          f'Time to create, train model and predict: {time_to_finish:.2f} seconds')
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
          f'Time to create, train model and predict: {time_to_finish:.2f} seconds')
    return mlp_classifier


def model_scores():
    model_scores = []
    for model in MODELS:
        model_scores.append(scores(model))
    df_model_scores = pd.DataFrame(model_scores,
                                   columns=['Accuracy Score', 'Recall Score', 'Precision Score', 'F1 Score',
                                            'Scores Mean'],
                                   index=[model.__class__.__name__ for model in MODELS])
    pd.set_option('display.max_columns', 5)
    print(df_model_scores)
    pd.reset_option('display.max_columns')


def menu():
    print("Choose an option:")
    print("1 - Model creation and fitting")
    print("2 - Show model scores")
    print("3 - Model cross-validation")
    print("4 - Data Profiling through pandas_profiling")
    print("5 - Show all models and their parameters")
    print("6 - Data Clustering")
    print("404 - Exit")


def model_creation():
    print("What kind of model do you want to create?")
    print("1 - Random Forest Classifier")
    print("2 - Logistic Regression")
    print("3 - Multilayer Perceptron Classifier")
    model_choice = input("Choose your model:")
    if model_choice == "1":
        MODELS.append(randomForest_model())
    elif model_choice == "2":
        MODELS.append(logisticRegression_model())
    elif model_choice == "3":
        MODELS.append(multilayerPerceptron_model())
    else:
        print("Invalid option.")


def cross_validation():
    model_list = []
    for model in MODELS:
        model_list.append(model.__class__.__name__)
    print(pd.Series(model_list))

    model_choice = int(input("Which model do you want to cross validate?: "))
    splits_choice = int(input("How many splits do you want in your KFold?: "))
    if input("Do you want shuffling? (Recommended) [y / n]:") == "n":
        cv_shuffling = False
    else:
        cv_shuffling = True
    print("OK!")
    print(f"Doing Cross validation on model {MODELS[model_choice].__class__.__name__}")
    validation_results = cross_validate(MODELS[model_choice], x, y,
                                        cv=KFold(n_splits=splits_choice, shuffle=cv_shuffling))
    print("DONE!\nHere are the results:")
    print(pd.DataFrame(validation_results))
    print("Overall fiting time, scoring time and test score (Mean):")
    print(f"Fitting time: {validation_results['fit_time'].mean():.3f} Seconds \n"
          f"Scoring time: {validation_results['score_time'].mean():.3f} Seconds \n"
          f"Test score: {validation_results['test_score'].mean():.3}")


def clustering():
    stdscaler = StandardScaler()
    data_scaled = stdscaler.fit_transform(data_dummified)
    n_clusters_choice = int(input("How much clusters do you want? (5 is the standard): "))
    print("Analizing and creating Clusters...")
    kmeans = KMeans(n_clusters=n_clusters_choice, n_init=10, max_iter=300)
    kmeans.fit(data_scaled)
    data_dummified['CLUSTER'] = kmeans.labels_
    description = data_dummified.groupby('CLUSTER')
    n_clients = description.size()
    description = description.mean()
    description['n_clients'] = n_clients
    print('Clusters overall descriptions: ')
    print(description)
    description_html = description.to_html()
    description_html_file = open("clustering description.html", "w")
    description_html_file.write(description_html)
    description_html_file.close()
    print("A HTML file has been created for better visualization on (project folder)/clustering description.html")


def hyper_parameters():
    # TODO
    pass


def standardize_data_and_split(data):
    stdscaler = StandardScaler()
    std_x = stdscaler.fit_transform(data.drop(labels=['SITUACAO'], axis=1))
    std_y = data['SITUACAO']
    std_train_x, std_test_x, train_y, test_y = train_test_spliting(std_x, std_y)
    return std_train_x, std_test_x, train_y, test_y

### SANITIZAÇÃO DE DADOS DEPOIS DE ANÁLISES PELO DATASPELL
data_file_name = 'dataset_customer_churn.csv'
print(f"Reading data from {data_file_name}")
data = pd.read_csv(data_file_name, sep='^')
is_NAN = data[data.isna().any(axis=1)]
print("Removing low importance features...")
# Certas colunas dos dados foram removidas por sua baixa importância nas análises de features, podendo assim criar
# um modelo muito mais otimizado sem sacrificar sua eficiência.
data.drop(labels=['A006_REGISTRO_ANS', 'CODIGO_BENEFICIARIO', 'CLIENTE', 'CD_USUARIO', 'CODIGO_FORMA_PGTO_MENSALIDADE',
                  'A006_NM_PLANO', 'DIAS_ATE_REALIZAR_ALTO_CUSTO', 'CD_ASSOCIADO', 'ESTADO_CIVIL'], axis=1,
          inplace=True)
print("Removing NAN's and outliers from DataFrame")
data.drop(is_NAN.index, axis=0, inplace=True)
# outlier extremo com + de 500 anos de plano ativo
data.drop(labels=182212, axis=0, inplace=True)
print("Replacing categoric features with only 2 options with 0 and 1")
dict_replace = {
    "SIM": 1,
    "NAO": 0,
    'F': 0,
    'M': 1,
    'DESATIVADO': 1,
    'ATIVO': 0,

}
data.replace(dict_replace, inplace=True)

# REDUÇÃO DE CARDINALIDADE NA TABELA 'QTDE_DIAS_ATIVO'
QTDE_DIAS_ATIVO_MENOR_QUE_365 = np.array(data['QTDE_DIAS_ATIVO'] < 365)
data['QTDE_DIAS_ATIVO_MENOR_QUE_365'] = 0
data.loc[QTDE_DIAS_ATIVO_MENOR_QUE_365, 'QTDE_DIAS_ATIVO_MENOR_QUE_365'] = 1
QTDE_DIAS_ATIVO_MENOR_QUE_1000 = np.array((data['QTDE_DIAS_ATIVO'] >= 365) & (data['QTDE_DIAS_ATIVO'] < 1000))
data['QTDE_DIAS_ATIVO_MENOR_QUE_1000'] = 0
data.loc[QTDE_DIAS_ATIVO_MENOR_QUE_1000, 'QTDE_DIAS_ATIVO_MENOR_QUE_1000'] = 1
QTDE_DIAS_ATIVO_MAIOR_QUE_1000 = np.array(data['QTDE_DIAS_ATIVO'] >= 1000)
data['QTDE_DIAS_ATIVO_MAIOR_QUE_1000'] = 0
data.loc[QTDE_DIAS_ATIVO_MAIOR_QUE_1000, 'QTDE_DIAS_ATIVO_MAIOR_QUE_1000'] = 1
data.drop(labels='QTDE_DIAS_ATIVO', inplace=True, axis=1)

data_dummified = pd.get_dummies(data)

if input("Do you want do standardize the data? [y / n]") == 'n':
    x = data_dummified.drop(labels=['SITUACAO'], axis=1)
    y = data_dummified['SITUACAO']
    print("Splitting Data between train and test")
    train_x, test_x, train_y, test_y = train_test_spliting(x, y)
else:
    print("Scaling data and splitting between train and test")
    train_x, test_x, train_y, test_y = standardize_data_and_split(data_dummified)


if __name__ == '__main__':
    while True:
        menu()
        menu_choice = input("Choose an option:")
        if menu_choice == "1":
            model_creation()
        elif menu_choice == "2":
            model_scores()
        elif menu_choice == "3":
            cross_validation()
        elif menu_choice == "4":
            pd_profiling_report(data)
        elif menu_choice == "5":
            for model in MODELS:
                print(model)
        elif menu_choice == "6":
            clustering()
        elif menu_choice == "7":
            hyper_parameters()
        elif menu_choice == "404":
            break
