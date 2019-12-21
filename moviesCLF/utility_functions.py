import pandas as pd
from sklearn import metrics
from sklearn.svm import LinearSVC

def getProbThresh(mydata, threshSel=1, threshOffset=0):
    '''
    The probability threshold to be used for making classification decisions
    threshSel = 1 : Default 0.5 probability threshold
    threshSel = 2 : max(0.5, Fraction of genre occurence + threshOffset)
    '''
    numGenres = mydata.shape[1]
    probThresh = []
    if threshSel == 1:
        probThresh = [0.5] * numGenres
    elif threshSel == 2:
        sumGenre = mydata.sum()
        probThresh = (sumGenre / mydata.shape[0] + threshOffset).clip(upper=0.5)
    return probThresh

def multiLabelPredict(clf, X_test, probThresh, categories):
    '''
    Multi-label prediction based on probability threshold.
    Prediction is made based on Binary Relevance where each genre has a separate classifier
    '''
    categoryColumns = categories
    
    y_pred = pd.DataFrame(columns=categoryColumns)
    
    prob = clf.predict_proba(X_test)
    for idx, col in enumerate(categoryColumns):
        y_pred[col] = prob[:,idx] > probThresh[idx]
    prob = pd.DataFrame(prob, columns=categoryColumns)
    return prob, y_pred

def printScores(methodName, y_test, y_pred):
    '''
    Printing on StdOut precision, recall, accuracy and F1-Measure
    '''
    accuracyScore = metrics.accuracy_score(y_test, y_pred)
    recallScore = metrics.recall_score(y_test, y_pred, average = 'macro')
    precisionScore = metrics.precision_score(y_test, y_pred, average = 'macro')
    accuracyScore = metrics.accuracy_score(y_test, y_pred)
    f1Score = metrics.f1_score(y_test, y_pred, average = 'macro')
    print("###################################################################")
    print( methodName + " ACCURACY: %0.2f%%" % (accuracyScore*100))
    print( methodName + " RECALL: %0.2f%%" % (recallScore*100))
    print( methodName + " PRECISION: %0.2f%%" % (precisionScore*100))
    print( methodName + " F1-Measure: %0.2f%%" % (f1Score*100))
    print("###################################################################")