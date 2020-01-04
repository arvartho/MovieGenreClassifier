import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from utility_functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_selection import SelectPercentile, chi2

# Loading the input files

trainData = pd.read_csv('../data/movieDB_clean_train.csv')
testData = pd.read_csv('../data/movieDB_clean_test.csv')

removeStopwords(testData,'title')
removeStopwords(testData,'plot')
removeStopwords(trainData,'title')
removeStopwords(trainData,'plot')

# split train and test sets

train_X, train_y = trainData['new_plot'], trainData.drop(['new_title', 'new_plot'], axis=1)
test_X, test_y = testData['new_plot'], testData.drop(['new_title', 'new_plot'], axis=1)

categories = train_y.columns

""" Binary Relevance:

The Baseline approach, called the binary relevance method, amounts to independently training one binary classifier for each label.
Given an unseen sample, the combined model then predicts all labels for this sample for which the respective classifiers predict a positive result.
We use the inbuilt sklearn OneVsRestClassifier function to achieve this multi-label classification.
OnevsRestClassifier is commonly used for multi-class classification but it can also be used for multi-label classification.

Probability threshold used for classifying each genres is based on the frequency of its occurence

"""

probThresh = getProbThresh(trainData[categories], threshSel=1)

# Bayes
print("")
print("Starting Bayes classifiers...")
print("")

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(MultinomialNB(class_prior=None)))
            ])

parameters = [
            {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__min_df': (1,2),
            'tfidf__ngram_range': [(1, 1), (1, 2),(1, 3)],
            'clf__estimator__alpha': (0.001, 0.01, 0.4, 0.8, 1),
            'clf__estimator__fit_prior': (True,False)
            }
         ]

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)
grid_search_cv.fit(train_X, train_y)

best_clf = grid_search_cv.best_estimator_
y_pred = best_clf.predict(test_X)
printScores('BAYES', test_y, y_pred)

# Linear SVM
print("")
print("Starting Linear SVM classifiers...")
print("")

pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=-1)),
            ])

# In order to obtain the parameters we used pipeline.get_params().keys()
parameters = [
            {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__min_df': (1,2),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__estimator__class_weight': ['balanced'],
            }
         ]

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)
grid_search_cv.fit(train_X, train_y)

# measuring performance on test set
best_clf = grid_search_cv.best_estimator_
y_pred = best_clf.predict(test_X)
printScores('Linear SVM', test_y, y_pred)

# RBF SVM
from sklearn.svm import SVC
print("")
print("Starting RBF SVM classifiers...")
print("")
pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', OneVsRestClassifier(SVC(gamma=2, C=1,probability=True), n_jobs=-1)),
            ])
parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__min_df': (1,2),
            'tfidf__ngram_range': [(1, 1), (1, 3)],
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)
grid_search_cv.fit(train_X, train_y)

# measuring performance on test set
best_clf = grid_search_cv.best_estimator_
prob, y_pred = multiLabelPredict(best_clf, test_X, probThresh, categories)
printScores('RBF SVM', test_y, y_pred)

# Neural Net
from sklearn.neural_network import MLPClassifier
print("")
print("Starting Neural Net classifiers...")
print("")
pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('select_features', SelectPercentile(chi2, percentile = 40)),
            # NN Clasifier will need much resources so it will not be optimal to try different parameters
            ('clf', OneVsRestClassifier(MLPClassifier(solver='lbfgs', tol=0.00001, activation='relu', hidden_layer_sizes=(25,25,25), max_iter = 500), n_jobs=6)),
            ])
parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__min_df': (1,2),
            'tfidf__ngram_range': [(1, 1), (1, 3)],
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)
grid_search_cv.fit(train_X, train_y)

# measuring performance on test set
best_clf = grid_search_cv.best_estimator_
prob, y_pred = multiLabelPredict(best_clf, test_X, probThresh, categories)
printScores('Neural Net', test_y, y_pred)
