import time
import pandas as pd
from pandas_profiling import ProfileReport
import numpy as np
from flask import Flask, jsonify, request
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

flask_app = Flask(__name__)


def scores(model_param):
    model_predicts = model_param.predict(test_x)
    model_scores_dict = {
        'Accuracy score': accuracy_score(test_y, model_predicts),
        'Recall score': recall_score(test_y, model_predicts),
        'Precision score': precision_score(test_y, model_predicts),
        'F1 score': f1_score(test_y, model_predicts),
        'Overall score': ((accuracy_score(test_y, model_predicts) + recall_score(test_y, model_predicts) +
                           precision_score(test_y, model_predicts) + f1_score(test_y, model_predicts)) / 4)
    }
    return model_scores_dict


def all_scores(predicts, test_y_param):
    print(f" Accuracy Score: {accuracy_score(test_y_param, predicts):.3f}\n"
          f" Recall Score: {recall_score(test_y_param, predicts):.3f}\n"
          f" Precision Score: {precision_score(test_y_param, predicts):.3f}\n"
          f" F1 Score: {f1_score(test_y_param, predicts):.3f}\n")


def pd_profiling_report(data_param):
    data_profile = ProfileReport(data_param)
    data_profile.to_file('data_report.html')


def train_test_spliting(x_param, y_param):
    train_x_, test_x_, train_y_, test_y_ = train_test_split(x_param, y_param, test_size=0.2, stratify=y_param)
    return train_x_, test_x_, train_y_, test_y_


def randomForest_model():
    time_on_creation = time.time()
    print('Creating Random Forest Model...')
    random_forest_classifier = RandomForestClassifier(n_estimators=10)
    print('Fitting data into model for training...')
    random_forest_classifier.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print('Done!')

    model_info = {
        'model_type': random_forest_classifier.__class__.__name__,
        'model_parameters': random_forest_classifier.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(random_forest_classifier)
    }

    return {'model_info': model_info, 'model': random_forest_classifier}


def logisticRegression_model():
    time_on_creation = time.time()
    print('Creating Logistic Regression Model...')
    log_regression = LogisticRegression(max_iter=20000)
    print('Fitting data into model for training...')
    log_regression.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print('Done!')
    model_info = {
        'model_type': log_regression.__class__.__name__,
        'model_parameters': log_regression.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(log_regression)
    }

    return {'model_info': model_info, 'model': log_regression}


def multilayerPerceptron_model():
    time_on_creation = time.time()
    print('Creating MultiLayer Perceptron Model...')
    mlp_classifier = MLPClassifier()
    print('Fitting data into model for training...')
    mlp_classifier.fit(train_x, train_y)
    time_to_finish = time.time() - time_on_creation
    print('Done!')
    model_info = {
        'model_type': mlp_classifier.__class__.__name__,
        'model_parameters': mlp_classifier.get_params(),
        'seconds_to_create_model': round(time_to_finish, 2),
        'model_scores': scores(mlp_classifier)
    }

    return {'model_info': model_info, 'model': mlp_classifier}


def model_scores():
    model_scores_dict = {}
    index_model = 0
    for model_ in MODELS:
        model_scores_dict[str(index_model) + " - " + model_.__class__.__name__] = scores(model_)
        index_model += 1
    return model_scores_dict


def menu():
    print("Choose an option:")
    print("1 - Model creation")
    print("2 - View and validate models")
    print("3 - Data Clustering and Profiling")
    print("4 - Randomized Forest feature importances")
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
    if not model_list:
        print('No models created!')
        return
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
    kmeans = KMeans(n_clusters=n_clusters_choice, n_init=10, max_iter=300, random_state=1010)
    kmeans.fit(data_scaled)
    data_dummified['CLUSTER'] = kmeans.labels_
    description = data_dummified.groupby('CLUSTER')
    n_clients = description.size()
    description = description.mean()
    description['n_clients'] = n_clients
    print('Clusters overall descriptions: ')
    print(description[['SITUACAO', 'n_clients']])
    description_html = description.to_html()
    description_html_file = open("clustering description.html", "w")
    description_html_file.write(description_html)
    description_html_file.close()
    print("A HTML file has been created for better visualization on (project folder)/clustering description.html")


def hyper_parameters():
    print('1 - Randomized Forest Classifier')
    print('2 - Logistic Regression Classifier')
    print('3 - Multilayer Perceptron Classifier')
    model_choice = int(input("Which kind of model do you want to create with Hyper Parameters?: "))
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
        'learning_rate': ['constant', 'adaptive']
    }
    print(
        "Doing the Multilayer Perceptron Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
    RSCross_validation = RandomizedSearchCV(MLPClassifier(),
                                            RSCV_parameters, n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    RSCross_validation.fit(train_x, train_y)
    print('Done!')
    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - MLPClassifier.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    print(
        'A HTML file was created with the results dataframe, ordered by their score! (incase you want to review all the models)')
    hp_best_estimator = RSCross_validation.best_estimator_
    MODELS.append(hp_best_estimator)


def hyper_parameters_logistic_regression():
    RSCV_parameters = {
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'penalty': ['l2'],
        'C': [100, 10, 1.0, 0.1, 0.01]
    }
    print(
        "Doing the Logistic Regression Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
    RSCross_validation = RandomizedSearchCV(LogisticRegression(max_iter=20000),
                                            RSCV_parameters, n_iter=10, cv=KFold(n_splits=5, shuffle=True))
    RSCross_validation.fit(train_x, train_y)
    print('Done!')
    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - LRC.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    print(
        'A HTML file was created with the results dataframe, ordered by their score! (incase you want to review all the models)')
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
        print(
            "Doing the Random Forest Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
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
        print(
            "Doing the Random Forest Classifier Hyper Parameters through Randomized Search Cross Validation... (this might take a while)")
        RSCross_validation = RandomizedSearchCV(RandomForestClassifier(n_estimators=10), RSCV_parameters,
                                                n_iter=n_iter_parameter,
                                                cv=KFold(n_splits=k_fold_parameter, shuffle=True))
        RSCross_validation.fit(train_x, train_y)

    results = pd.DataFrame(RSCross_validation.cv_results_)
    results_html = results.sort_values(by='rank_test_score').to_html()
    results_html_file = open("hyper parameters results - RFC.html", "w")
    results_html_file.write(results_html)
    results_html_file.close()
    print(
        'A HTML file was created with the results dataframe, ordered by their score! (incase you want to review all the models)')
    hp_best_estimator = RSCross_validation.best_estimator_
    MODELS.append(hp_best_estimator)


def standardize_data_and_split(x_param, y_param):
    stdscaler = StandardScaler()
    std_x = stdscaler.fit_transform(x_param)
    std_train_x, std_test_x, train_y_, test_y_ = train_test_spliting(std_x, y_param)
    return std_train_x, std_test_x, train_y_, test_y_


def RFC_feature_importance():
    RFC_list = []
    for model_ in MODELS:
        if model_.__class__.__name__ == 'RandomForestClassifier':
            RFC_list.append(model_.__class__.__name__)
    if not RFC_list:
        print('No Random Forest Classifiers created!')
        return
    print(pd.Series(RFC_list))
    rfc_choice = int(input('Which model do you want to see the feature importances?: '))
    feature_importances_series = pd.Series(RFC_list[rfc_choice].feature_importances_,
                                           index=pd.Series([col for col in x.columns]))
    print(feature_importances_series.sort_values(ascending=False) * 100)


@flask_app.route('/')
def home():
    welcome_message = "Welcome to iml api!\nHere, you can a whole lot of stuff, that i will sometime later add" \
                      "\n the api is focused on Machine Learning and model creation!"
    return welcome_message


@flask_app.route('/create_model/RFC')
def api_create_model_RFC():
    model = randomForest_model()
    MODELS.append(model['model'])
    return model['model_info']


@flask_app.route('/create_model/LRC')
def api_create_model_LRC():
    model = logisticRegression_model()
    MODELS.append(model['model'])
    return model['model_info']


@flask_app.route('/create_model/MLP')
def api_create_model_MLP():
    model = multilayerPerceptron_model()
    MODELS.append(model['model'])
    return model['model_info']


@flask_app.route('/model_visualization/')
def api_model_visualization():
    models_dict = {}
    index_test = 0
    for model in MODELS:
        models_dict[str(index_test) + " - " + model.__class__.__name__] = model.get_params()
        index_test += 1
    return jsonify(models_dict)


@flask_app.route('/set_data/', methods=["POST"])
def api_set_data():
    file_loc_json = request.get_json()
    data_dummified = pd.read_csv(file_loc_json['file_loc'])
    x = data_dummified.drop(labels=[file_loc_json['y']], axis=1)
    y = data_dummified[file_loc_json['y']]
    train_x, test_x, train_y, test_y = standardize_data_and_split(x, y)
    return 'new data was set!'


@flask_app.route('/model_score/')
def api_model_scores():
    return jsonify(model_scores())


data_dummified = pd.read_csv('clean_dataset_customer_churn.csv')

x = data_dummified.drop(labels=['SITUACAO'], axis=1)
y = data_dummified['SITUACAO']

train_x, test_x, train_y, test_y = standardize_data_and_split(x, y)

if __name__ == '__main__':
    flask_app.run(debug=True)
    # while True:
    #     menu()
    #     menu_choice = input("Choose an option: ")
    #     if menu_choice == "1":
    #         print("1 - Create basic model")
    #         print("2 - Create model through hyper-parameters")
    #         creation_choice = input("Choose an option: ")
    #         if creation_choice == '1':
    #             model_creation()
    #         elif creation_choice == '2':
    #             hyper_parameters()
    #         else:
    #             print('Invalid Choice.')
    #
    #     elif menu_choice == "2":
    #         print('1 - Show model scores')
    #         print('2 - Show all models and their parameters')
    #         print('3 - Model cross-validation')
    #         view_choice = input("Choose an option: ")
    #         if view_choice == '1':
    #             model_scores()
    #         elif view_choice == '2':
    #             for model in MODELS:
    #                 print(model)
    #         elif view_choice == '3':
    #             cross_validation()
    #         else:
    #             print('Invalid choice.')
    #     elif menu_choice == "3":
    #         print('1 - Data Clustering')
    #         print('2 - Data Profiling')
    #         data_choice = input("Choose an option: ")
    #         if data_choice == '1':
    #             clustering()
    #         elif data_choice == '2':
    #             pd_profiling_report(data)
    #         else:
    #             print('Invalid choice.')
    #     elif menu_choice == '4':
    #         RFC_feature_importance()
    #     elif menu_choice == "404":
    #         break
