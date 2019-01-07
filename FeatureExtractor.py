from bs4 import BeautifulSoup
import os
import numpy as np
import collections
import math
import urllib.request as request
import requests
import pandas as pd
import collections
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier

SEPERATE = '\n' + 100*'-' + '\n'
# random.seed(22)
print(SEPERATE)

def extractNumericalFeatures(features, data, feat_pd):
    for feature in features:
        feat_pd[feature] = data[feature]
    return feat_pd


def extractConferenceFeatures(data, feat_pd, conferences):
    for conf in conferences:
        feat_pd.loc[data['Conference'] == conf, conf] = 1
    return feat_pd


def featureExtractor(features, data, games, conferences):
    feat_pd = pd.DataFrame(index=games, columns=features)
    feat_pd = extractNumericalFeatures(features, data, feat_pd)
    feat_pd = extractConferenceFeatures(data, feat_pd, conferences)

    # Clean and Save Feature Template
    feat_pd.fillna(value=0, inplace=True)
    feat_pd.to_csv('FeatureTemplate.csv')
    return feat_pd


def loadTrials(filename):
    return pd.read_csv(filename, index_col=0)


def cleanDataPanda(data, remove):
    games = list(data.index)
    columns = list(data.columns)
    for category in remove:
        columns.remove(category)
    conferences = list(set(data['Conference']))
    conferences.sort()
    features = columns
    return games, features, conferences


def differentialFeatures(feat_pd, numfeatures):
    numfeatures.remove('Team')
    (ngames, nfeatures) = feat_pd.shape
    allgames = list(feat_pd.index)
    games = [re.sub('T\d+20', '20', val) for val in allgames[::2]]

    featureDataFrame = pd.DataFrame(columns=numfeatures)
    teamDataFrame = pd.DataFrame(index=games, columns=['Team 1', 'Team 2'])
    y = []

    for i in range(len(games)):
        order = random.randint(1, 2)
        t1 = feat_pd.at[allgames[2*i], 'Team']
        t2 = feat_pd.at[allgames[2*i+1], 'Team']
        g1 = allgames[2*i]
        g2 = allgames[2*i+1]
        line1 = feat_pd.loc[g1]
        line2 = feat_pd.loc[g2]
        slice1 = line1[numfeatures]
        slice2 = line2[numfeatures]
        if order == 1:
            y.append(1)
            featureDataFrame.loc[games[i]] = slice1 - slice2
            teamDataFrame.loc[games[i]] = [t1, t2]
        else:
            y.append(0)
            featureDataFrame.loc[games[i]] = slice2 - slice1
            teamDataFrame.loc[games[i]] = [t2, t1]
    # featureDataFrame.to_csv('FinalFeatures.csv')
    return featureDataFrame, teamDataFrame, y


def modelCrossValidate(models, x, y):
    for name, model in models:
        scores = cross_val_score(model, x, y, cv=50)
        print('Cross Validation Accuracy for {}: {} (+/-) {}'.format(name.upper(), scores.mean(), scores.std()))
    print(SEPERATE)

def modelFit(models, x, y):
    for name, model in models:
        model.fit(x, y)
        print('{} Model Fit'.format(name.upper()))
    print(SEPERATE)
    return models


def modelSetup(models, x, y):
    modelCrossValidate(models, x, y)
    models = modelFit(models, x, y)
    return models


def predictValues(models, X, Y, teams):
    predicted_2018 = dict()
    for name, model in models:
        predicted_2018[name] = list(model.predict(X))
    resultsCompile(Y, predicted_2018, teams, models)


def resultsCompile(y, prediction, teams, m):
    writer = pd.ExcelWriter('Results.xlsx')

    models = list(prediction.keys())
    columns = ['Team 1', 'Team 2', 'Winner'] + models + ['Combined']
    results = pd.DataFrame(index=teams.index, columns=columns)
    results.ix[:, 2] = y
    for i, team in enumerate(teams.values):
        results.ix[i, 0] = team[0]
        results.ix[i, 1] = team[1]

    print('PREDICTING 2018 BOWL GAMES:')
    for i, model in enumerate(prediction.keys()):
        for j, value in enumerate(prediction[model]):
            results.ix[j, i+3] = value
        compare = np.where((results[model] == results['Winner']), 1, 0)
        print('{} Model Prediction Accuracy for 2018 Bowls: {}'.format(m[i][0].upper(), compare.mean()))

    print(SEPERATE)
    results.to_excel(writer, sheet_name='Summary')
    print('Saved Results')
    print(SEPERATE)
    return results


def getFeatures(filename, removeColumns=[]):
    data = loadTrials(filename)
    games, features, conferences = cleanDataPanda(data, removeColumns)
    feat_pd = featureExtractor(features, data, games, conferences)
    featureDataFrame, teamDataFrame, y = differentialFeatures(feat_pd, features)
    return featureDataFrame, teamDataFrame, y


def main():
    # ---------------------------------------------------------------------------------------------

    # EXTRACT FEATURES
    # Extract features from training data needed for model fitting. Store in panda dataframe
    x, teamDataFrame, y = getFeatures("TrialCases.csv", removeColumns=['Link', 'g', 'Conference'])

    # ---------------------------------------------------------------------------------------------

    # MACHINE LEARNING MODEL FITTING
    # Specify all models for fitting/Cross Validation. Enter them into the model setup function
    models = [
             ('logistic', LogisticRegression(solver='liblinear')),
             ('random forest', RandomForestClassifier(n_estimators=20, class_weight='balanced')),
             ('gaussian naive bayes', GaussianNB()),
             ('SVM linear', svm.LinearSVC(C=5)),
             ('SVM polynomial', svm.SVC(kernel='poly', degree=3, C=5, gamma='scale')),
             ('Neural Network', MLPClassifier(activation='logistic', solver='lbfgs', alpha=5))
              ]
    fit_models = modelSetup(models, x, y)



    # ---------------------------------------------------------------------------------------------

    # PREDICT BOWLS
    # Use 2018 Bowl data to predict the winners for this year
    X, teams, Y = getFeatures('2018_Data.csv', removeColumns=['Link', 'g', 'Conference'])
    predictValues(fit_models, X, Y, teams)

    # ---------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
