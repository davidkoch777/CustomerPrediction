# Import libraries necessary for this project
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import numpy as np
import PyQt5
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
# from sklearn.impute import SimpleImputer <-- New version
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


def run_classification_model(data):
    numeric_features = ['Mitarbeiteranzahl', 'Umsatz', 'Wachstum']
    numeric_transformer = Pipeline(steps=[
        ('imputer', Imputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['Land_ID', 'Branche_ID']
    categorical_transformer = Pipeline(steps=[
        ('imputer', Imputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(solver='lbfgs'))])

    X = data.drop('IstKunde', axis=1)
    y = data['IstKunde']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train, y_train)
    print("Linear regression model score: %.3f" % clf.score(X_test, y_test))


def run_regression_analysis(data):
    # Create list to loop through
    dep_vars = list(data.columns)

    # Create loop to test each feature as a dependent variable
    for var in dep_vars:

        # TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
        new_data = data.drop([var], axis = 1)
        # Confirm drop
        # display(new_data.head(2))

        # Create feature Series (Vector)
        new_feature = pd.DataFrame(data.loc[:, var])
        # Confirm creation of new feature
        # display(new_feature.head(2))

        # TODO: Split the data into training and testing sets using the given feature as the target
        X_train, X_test, y_train, y_test = train_test_split(new_data, new_feature, test_size=0.25, random_state=42)

        # TODO: Create a decision tree regressor and fit it to the training set
        # Instantiate
        # dtr = DecisionTreeRegressor(random_state=42)
        # Fit
        # dtr.fit(X_train, y_train)
        # TODO: Create a random forest regressor and fit it to the training set
        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
        rfr.fit(X_train, y_train.values.ravel())

        # TODO: Report the score of the prediction using the testing set
        # Returns R^2
        score = rfr.score(X_test, y_test)
        print('R2 score for {} as dependent variable: {}'.format(var, score))

    # Produce a scatter matrix for each pair of features in the data
    pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(14, 8), diagonal='kde')
    plt.show()


# Load the wholesale customers dataset
try:
    data = pd.read_csv("C:/Users/dakoch/Downloads/customer_dataset.csv")
    data.drop(["ID"], axis = 1, inplace = True)
    print("Customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")


# Display a description of the dataset
stats = data.describe()
display(stats)
run_classification_model(data)
# clusters = KMeans(n_clusters=4).fit(data)
# y_kmeans = clusters.predict(data)
# print(len(y_kmeans))
# plt.scatter(data.values[:, 0], data.values[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = clusters.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
# plt.show()
