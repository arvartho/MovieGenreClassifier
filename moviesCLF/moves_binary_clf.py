import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from utility_functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Loading the input files

trainData = pd.read_csv('../data/movieDB_clean_train.csv')
testData = pd.read_csv('../data/movieDB_clean_test.csv')

# split train and test sets

train_X, train_y = trainData['plot'], trainData.drop(['title', 'plot'], axis=1)
test_X, test_y = testData['plot'], testData.drop(['title', 'plot'], axis=1)

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
                ('tfidf', TfidfVectorizer(max_df=0.5, ngram_range=(1, 2))),
                ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.01, fit_prior=True, class_prior=None)))
            ])
pipeline.fit(train_X, train_y)

prob, y_pred = multiLabelPredict(pipeline, test_X, probThresh, categories)
printScores('BAYES', test_y, y_pred)

# SVM
print("")
print("Starting SVM classifiers...")
print("")

pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])

# In order to obtain the parameters we used pipeline.get_params().keys()
parameters = {
            'tfidf__max_df': (0.25, 0.5),
            'tfidf__min_df': (1, 3),
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            "clf__estimator__C": [1, 10],
            "clf__estimator__class_weight": ['balanced'],
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=4)
grid_search_cv.fit(train_X, train_y)

# measuring performance on test set
best_clf = grid_search_cv.best_estimator_
y_pred = best_clf.predict(test_X)
printScores('SVM', test_y, y_pred)

# LogisticRegression
print("")
print("Starting LogisticRegression classifiers...")
print("")
pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),
            ])
parameters = {
            'tfidf__max_df': [0.25, 0.5, 0.75],
            'tfidf__min_df': [1, 2],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            "clf__estimator__C": [0.1, 1],
            }

grid_search_cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3)
grid_search_cv.fit(train_X, train_y)

# measuring performance on test set
best_clf = grid_search_cv.best_estimator_
prob, y_pred = multiLabelPredict(best_clf, test_X, probThresh, categories)
printScores('LogisticRegression', test_y, y_pred)
