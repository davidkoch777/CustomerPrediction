# Import libraries necessary for this project
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import PyQt5
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def get_preprocessor():
    numeric_features = ['Mitarbeiteranzahl', 'Umsatz', 'Wachstum']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['Land_ID', 'Branche_ID']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor

def run_classification_model(data, *data2, classifier_type):

    if(classifier_type == 'decisiontree'):
        classifier = DecisionTreeRegressor()

    if(classifier_type == 'neuralnetwork'):
        classifier = MLPClassifier(hidden_layer_sizes=(5,), random_state=42)

    if(classifier_type == 'naivebayes'):
        classifier = GaussianNB()

    if(classifier_type == 'gauss'):
        classifier = GaussianProcessClassifier(n_restarts_optimizer=10)

    if(classifier_type == 'neighbors'):
        classifier = KNeighborsClassifier()

    if(classifier_type == 'logistic'):
        classifier = LogisticRegression(solver='lbfgs')

    if(classifier_type == 'randomforest'):
        classifier = RandomForestClassifier(n_estimators=100)

    preprocessor = get_preprocessor()
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', classifier)])

    X = data.drop(['IstKunde'], axis=1)
    y = data['IstKunde']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if(classifier_type == 'randomforest'):
        y_train = y_train.values.ravel()

    clf.fit(X_train, y_train)

    print(classifier_type + " classification model score: %.3f" % clf.score(X_test, y_test))


def plot_clustering_by_features(data):
    sns.set(style="ticks")
    sns.pairplot(data, vars=('Land_ID', 'Branche_ID', 'Mitarbeiteranzahl', 'Umsatz', 'Wachstum'), hue="IstKunde")
    plt.show()


def compare_clustering(data, storetofile):
    preprocessor = get_preprocessor()
    kmeans = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', KMeans(n_clusters=2, init='random', algorithm='full', random_state=42))])

    spectral = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=42))])

    gaussian = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', GaussianMixture(n_components=2, n_init=10, init_params='random', random_state=42))])


    X = data.drop(['IstKunde'], axis=1)
    y_kmeans = kmeans.fit_predict(X)
    y_spectral = spectral.fit_predict(X)
    y_gaussian = gaussian.fit_predict(X)

    data_kmeans = data.copy()
    data_kmeans['IstKunde'] = y_kmeans
    data_spectral = data.copy()
    data_spectral['IstKunde'] = y_spectral
    data_gaussian = data.copy()
    data_gaussian['IstKunde'] = y_gaussian

    sns.set(style="ticks")
    f1 = sns.pairplot(data_kmeans, vars=('Land_ID', 'Branche_ID', 'Mitarbeiteranzahl', 'Umsatz', 'Wachstum'), hue="IstKunde")
    f1.fig.canvas.set_window_title('Kmeans-Clustering Scattermatrix')
    f2 = sns.pairplot(data_spectral, vars=('Land_ID', 'Branche_ID', 'Mitarbeiteranzahl', 'Umsatz', 'Wachstum'), hue="IstKunde")
    f2.fig.canvas.set_window_title('Spectral-Clustering Scattermatrix')
    f3 = sns.pairplot(data_gaussian, vars=('Land_ID', 'Branche_ID', 'Mitarbeiteranzahl', 'Umsatz', 'Wachstum'), hue="IstKunde")
    f3.fig.canvas.set_window_title('Gaussian-Mixture-Clustering Scattermatrix')

#    pca = Pipeline(steps=[('preprocessor', preprocessor), ('pca', PCA(n_components=2))])
    pca = PCA(n_components=2)
    arr_2d = pca.fit_transform(X)
    plt.figure(figsize=(15,8))
    colors = ['red', 'navy']
    target_names = ['KeinKunde', 'IstKunde']
    lw = 2
    plt.title('PCA of Customer dataset: Cluster Comparison')
    plt.subplot(2,2,1, title='Kmeans')
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(arr_2d[y_kmeans == i, 0], arr_2d[y_kmeans == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)

    plt.subplot(2,2,2, title='Spectral')
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(arr_2d[y_spectral == i, 0], arr_2d[y_spectral == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)

    plt.subplot(2,2,3, title='Gaussian')
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(arr_2d[y_gaussian == i, 0], arr_2d[y_gaussian == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)

    print("Kmeans-Clustering Information grouped IstKunde:")
    print(data_kmeans.groupby(['IstKunde', 'Land_ID', 'Branche_ID']).count())
    print("Spectral-Clustering Information grouped IstKunde:")
    print(data_spectral.groupby(['IstKunde', 'Land_ID', 'Branche_ID']).count())
    print("Gaussian-Mixture-Clustering Information grouped IstKunde:")
    print(data_gaussian.groupby(['IstKunde', 'Land_ID', 'Branche_ID']).count())

    if storetofile:
        pd.DataFrame.to_csv(data_kmeans, "C:/Users/dakoch/Downloads/CustomerClustering/customer_cluster_kmeans.csv", float_format="%.2f")
        pd.DataFrame.to_csv(data_spectral, "C:/Users/dakoch/Downloads/CustomerClustering/customer_cluster_spectral.csv", float_format="%.2f")
        pd.DataFrame.to_csv(data_gaussian, "C:/Users/dakoch/Downloads/CustomerClustering/customer_cluster_gaussian.csv", float_format="%.2f")

    plt.show()

 #
# Load the wholesale customers dataset
try:
#    data = pd.read_csv("C:/Users/dakoch/Downloads/customer_dataset.csv", float_precision='round_trip', dtype={'Land_ID':str, 'Branche_ID':str})
    data = pd.read_csv("C:/Users/dakoch/Downloads/CustomerClustering/customer_dataset.csv", float_precision='round_trip')
    data.drop(["ID"], axis=1, inplace=True)
    print("Customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")

# Display a description of the dataset
stats = data.describe()
display(stats)
# print("Information about the values in the clusters (IstKunde):")
# print(data.groupby(['IstKunde', 'Land_ID', 'Branche_ID']).count())

compare_clustering(data, storetofile=0)

# run_classification_model(data, classifier_type='randomforest')
# run_classification_model(data, classifier_type='logistic')
# run_classification_model(data, classifier_type='decisiontree')
# run_classification_model(data, classifier_type='neighbors')
# run_classification_model(data, classifier_type='gauss')
# run_classification_model(data, classifier_type='naivebayes')
# run_classification_model(data, classifier_type='neuralnetwork')

#plot_clustering_by_features(data)
