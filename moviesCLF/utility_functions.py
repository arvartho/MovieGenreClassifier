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
    result = pd.DataFrame(columns=['Precision', 'Recall', 'F1-Score', 'Support'])
    tp, fp, fn, total = 0, 0, 0, 0
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = pd.DataFrame(y_pred, columns=y_test.columns)
    for col in y_test.columns:
        support = y_test[col].sum()
        tp = ((y_test[col]==1) & (y_pred[col]==1)).sum()
        fp = ((y_test[col]==0) & (y_pred[col]==1)).sum()
        fn = ((y_test[col]==1) & (y_pred[col]==0)).sum()
        
        precision = 0 if (tp+fp==0) else tp/(tp+fp)
        recall = 0 if (tp+fn==0) else tp/(tp+fn)
        f1_score = 0 if (precision==0 and recall==0) else 2*precision*recall/(precision+recall)
        
        result.loc[col] = [precision, recall, f1_score, support]
    
    avg_precision = (result['Precision']*result['Support']).sum()/result['Support'].sum()
    avg_recall = (result['Recall']*result['Support']).sum()/result['Support'].sum()
    avg_f1_score = (result['F1-Score']*result['Support']).sum()/result['Support'].sum()
    result.loc['Avg/Total'] = [avg_precision, avg_recall, avg_f1_score, result['Support'].sum()]
    
    print(str(round(result, 2)))


def removeStopwords(df,column):
    from spacy.lang.el import Greek
    from spacy.lang.el.stop_words import STOP_WORDS
    pd.options.mode.chained_assignment = None
    from greek_stemmer import GreekStemmer

    # Load Greek tokenizer, tagger, parser, NER and word vectors
    nlp = Greek()
    # "nlp" Object is used to create documents with linguistic annotations.
    sentences = []
    
    for i, data in df.iterrows():
        sent = nlp(data[column])

        # Create list of word tokens
        token_list = []
        for token in sent:
            token_list.append(token.lemma_)

        # Create list of word tokens after removing stopwords
        filtered_sentence =[]

        for word in token_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word.lower())
        sentence = ' '.join(map(str, filtered_sentence))
        sentences.append(sentence)
    col_name = "new_" + str(column)
    df[col_name] = sentences
    del df[column]
