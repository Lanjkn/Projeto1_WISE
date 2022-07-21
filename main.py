import time
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from scipy.stats import randint
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import cross_validate, KFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

np.random.seed(1010)

MODELS = []


def scores(model_param):
    model_predicts = model_param.predict(test_x)
    return accuracy_score(test_y, model_predicts), recall_score(test_y, model_predicts), \
           precision_score(test_y, model_predicts), f1_score(test_y, model_predicts), \
           ((accuracy_score(test_y, model_predicts) + recall_score(test_y, model_predicts) +
             precision_score(test_y, model_predicts) + f1_score(test_y, model_predicts)) / 4)


def all_scores(predicts, test_y_param):
    print(f" Accuracy Score: {accuracy_score(test_y_param, predicts):.3f}\n"
          f" Recall Score: {recall_score(test_y_param, predicts):.3f}\n"
          f" Precision Score: {precision_score(test_y_param, predicts):.3f}\n"
          f" F1 Score: {f1_score(test_y_param, predicts):.3f}\n")


def pd_profiling_report(data_param):
    data_profile = ProfileReport(data_param)
    data_profile.to_file('pandas_profiling Reports/data_report.html')


def train_test_spliting(x_param, y_param):
    train_x_, test_x_, train_y_, test_y_ = train_test_split(x_param, y_param, test_size=0.2, stratify=y_param)
    return train_x_, test_x_, train_y_, test_y_


def randomForest_model():
    time_on_creation = time.time()
    print('Creating Random Forest Model...')
    random_forest_classifier = RandomForestClassifier(n_estimators=10)
    print('Fitting data into model for training...')
    random_forest_classifier.fit(train_x, train_y)
    print('Predicting on test data.')
    time_to_finish = time.time() - time_on_creation
    print(f'Done!\n'
          f'Time to create, train model and predict: {time_to_finish:.2f} seconds')
    return random_forest_classifier


def logisticRegression_model():
    time_on_creation = time.time()
    print('Creating Logistic Regression Model...')
    log_regression = LogisticRegression(max_iter=20000)
    print('Fitting data into model for training...')
    log_regression.fit(train_x, train_y)
    print('Predicting on test data.')
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
    time_to_finish = time.time() - time_on_creation
    print(f'Done!\n'
          f'Time to create, train model and predict: {time_to_finish:.2f} seconds')
    return mlp_classifier


def model_scores():
    model_scores_ = []
    for model_ in MODELS:
        model_scores_.append(scores(model_))
    df_model_scores = pd.DataFrame(model_scores_,
                                   columns=['Accuracy Score', 'Recall Score', 'Precision Score', 'F1 Score',
                                            'Scores Mean'],
                                   index=[model_.__class__.__name__ for model_ in MODELS])
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
    print("7 - Create Model through Hyper Parameters")
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
    for model_ in MODELS:
        model_list.append(model_.__class__.__name__)
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
    print('1 - Randomized Forest Classifier')
    print('2 - Logistic Regression Classifier')
    print('3 - Multilayer Perceptron Classifier')
    model_choice = int(input("Which kind of model do you want to create with Hyper Parameters?"))
    if model_choice == 1:
        hyper_parameters_random_forest()
    elif model_choice == 2:
        hyper_parameters_logistic_regression()
    elif model_choice == 3:
        hyper_parameters_multilayer_perceptron()
    print('Model created!')


def hyper_parameters_multilayer_perceptron():
    RSCV_parameters = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive']
    }
    print(
        "Doing the Multilayer Perceptron Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
    RSCross_validation = RandomizedSearchCV(MLPClassifier(),
                                            RSCV_parameters, n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - MLPClassifier.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    print('A HTML file was created with the results dataframe, ordered by their score! (incase you want to review all the models)')
    hp_best_estimator = RSCross_validation.best_estimator_
    MODELS.append(hp_best_estimator)


def hyper_parameters_logistic_regression():
    RSCV_parameters = {
        'solvers': ['newton-cg', 'lbfgs', 'liblinear'],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'c_values': randint(1, 100)
    }
    print(
        "Doing the Logistic Regression Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
    RSCross_validation = RandomizedSearchCV(LogisticRegression(),
                                            RSCV_parameters, n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    print('Done!')
    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - LRC.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    print('A HTML file was created with the results dataframe, ordered by their score! (incase you want to review all the models)')
    hp_best_estimator = RSCross_validation.best_estimator_
    MODELS.append(hp_best_estimator)


def hyper_parameters_random_forest():
    if input("Do you want to customize hyper parameters? [y / n]: ") == "y":
        customize_hp = True
    else:
        customize_hp = False
    print("OK!")
    if customize_hp == False:
        RSCV_parameters = {
            "max_depth": randint(10, 250),
            "min_samples_split": randint(2, 16),
            "min_samples_leaf": randint(1, 16),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }
        print("Doing the Random Forest Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
        RSCross_validation = RandomizedSearchCV(RandomForestClassifier(n_estimators=10), RSCV_parameters,
                                                n_iter=10, cv=KFold(n_splits=5, shuffle=True))
        RSCross_validation.fit(train_x, train_y)
    else:
        max_depth_first_parameter = int(input("Type the first parameter for the randint for max_depth: "))
        max_depth_second_parameter = int(input("Type the second parameter for the randint for max_depth: "))
        min_samples_split_first_parameter = int(
            input("Type the first parameter for the randint for min_samples_split: "))
        min_samples_split_second_parameter = int(
            input("Type the second parameter for the randint for min_samples_split: "))
        min_samples_leaf_first_parameter = int(input("Type the first parameter for the randint for min_samples_leaf: "))
        min_samples_leaf_second_parameter = int(
            input("Type the second parameter for the randint for min_samples_leaf: "))

        n_iter_parameter = int(input("How many iterations do you want in your Randomized Search Cross Validation?: "))
        k_fold_parameter = int(input("How many splits do you want in your KFold?: "))

        RSCV_parameters = {
            "max_depth": randint(max_depth_first_parameter, max_depth_second_parameter),
            "min_samples_split": randint(min_samples_split_first_parameter, min_samples_split_second_parameter),
            "min_samples_leaf": randint(min_samples_leaf_first_parameter, min_samples_leaf_second_parameter),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }
        print("Doing the Random Forest Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
        RSCross_validation = RandomizedSearchCV(RandomForestClassifier(n_estimators=10), RSCV_parameters,
                                                n_iter=n_iter_parameter,
                                                cv=KFold(n_splits=k_fold_parameter, shuffle=True))
        RSCross_validation.fit(train_x, train_y)

    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - RFC.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    print('A HTML file was created with the results dataframe, ordered by their score! (incase you want to review all the models)')
    hp_best_estimator = RSCross_validation.best_estimator_
    MODELS.append(hp_best_estimator)


def standardize_data_and_split(data_param):
    stdscaler = StandardScaler()
    std_x = stdscaler.fit_transform(data_param.drop(labels=['SITUACAO'], axis=1))
    std_y = data_param['SITUACAO']
    std_train_x, std_test_x, train_y_, test_y_ = train_test_spliting(std_x, std_y)
    return std_train_x, std_test_x, train_y_, test_y_


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

if input("Do you want do standardize the data? [y / n]: ") == 'n':
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
