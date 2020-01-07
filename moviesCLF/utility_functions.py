import pandas as pd
import numpy as np
import multiprocessing
import sys
import re
import spacy
from spacy.lang.el.stop_words import STOP_WORDS
import nltk
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook
from datetime import datetime
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE

# Uncomment to get nltk packages
# nltk.download()
nlp = spacy.load("el_core_news_sm")

# UTILITY FUNCTIONS
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
    result = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support'])
    tp, fp, fn, tn, total = 0, 0, 0, 0, 0
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = pd.DataFrame(y_pred, columns=y_test.columns)
    for col in y_test.columns:
        support = y_test[col].sum()
        tp = ((y_test[col]==1) & (y_pred[col]==1)).sum()
        fp = ((y_test[col]==0) & (y_pred[col]==1)).sum()
        fn = ((y_test[col]==1) & (y_pred[col]==0)).sum()
        tn = ((y_test[col]==0) & (y_pred[col]==0)).sum()
        
        accuracy = 0 if (tp+fp==0) else (tp+tn)/(tp+fp+fn+tn)
        precision = 0 if (tp+fp==0) else tp/(tp+fp)
        recall = 0 if (tp+fn==0) else tp/(tp+fn)
        f1_score = 0 if (precision==0 and recall==0) else 2*precision*recall/(precision+recall)
        
        result.loc[col] = [accuracy, precision, recall, f1_score, support]
    
    avg_accuracy = (result['Accuracy']*result['Support']).sum()/result['Support'].sum()
    avg_precision = (result['Precision']*result['Support']).sum()/result['Support'].sum()
    avg_recall = (result['Recall']*result['Support']).sum()/result['Support'].sum()
    avg_f1_score = (result['F1-Score']*result['Support']).sum()/result['Support'].sum()
    result.loc['Avg/Total'] = [avg_accuracy, avg_precision, avg_recall, avg_f1_score, result['Support'].sum()]
    
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


def plot_vocabulary(w2v_model):
    output_notebook()

    model_plot = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

    # getting a list of word vectors. limit to 10000. each is of 200 dimensions
    word_vectors = [w2v_model[w] for w in list(w2v_model.wv.vocab.keys())[:5000]]

    # dimensionality reduction. converting the vectors to 2d vectors
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_w2v = tsne_model.fit_transform(word_vectors)

    # putting everything in a dataframe
    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    tsne_df['words'] = list(w2v_model.wv.vocab.keys())[:5000]

    # plotting. the corresponding word appears when you hover on the data point.
    model_plot.scatter(x='x', y='y', source=tsne_df)
    hover = model_plot.select(dict(type=HoverTool))
    hover.tooltips={"word": "@words"}
    show(model_plot)


def df_preprocessing(train_df, test_df):   
   train_df['feature_str'] = train_df['title'].str.replace("'", " ") + train_df['plot']
   test_df['feature_str'] = test_df['title'].str.replace("'", " ") + test_df['plot']

   # Vectorize training set
   train_df['tokenized_feature'] = train_df['feature_str'].apply(word_preprocessing)
   test_df['tokenized_feature'] = test_df['feature_str'].apply(word_preprocessing)   
   return train_df, test_df


def word_preprocessing(doc): 
   return list(filter(None, 
                     [re.sub(r'\s+', '', token.lemma_.lower()) 
                        for token in nlp(doc) if not token.is_punct 
                           and not token.is_stop
                           and not token.is_ascii
                           and not token.is_currency
                           and token.lemma_ not in STOP_WORDS])
                           )


def build_w2v_features(tokens, w2v_model):
   for token in tokens:
      try:
         t = w2v_model.wv[token]
      except KeyError: # handling the case where the token is not in the vocabulary
         continue
   return t 


def build_weighted_w2v_features(tokens, w2v_model, tfidf, size=200):
   # Build Word2Vec feature vector by compiling a frequency-based weighted average for each token
   vec = np.zeros((1, size))
   for token in tokens:
      try:
         vec += w2v_model.wv[token].reshape((1, size)) * tfidf[token]
      except KeyError: # handling the case where the token is not in the vocabulary
         continue
   vec /= len(token)
   return vec  


def tfidf_vectorizer(series):
   '''
   Conducts simple TF-IDF feature selection with predefined parameters
   '''
   start = datetime.now()
   print('Started TF-IDF vectorization...')
   vectorizer = TfidfVectorizer(analyzer='word', 
                                min_df=2, 
                                ngram_range=(1, 3),
                                norm='l2', 
                                strip_accents=None)
   vectorizer.fit(series)
   print('Finished TF-IDF vectorizing in: ', datetime.now()-start)
   return vectorizer
   
# FEATURE SELECTION TECHNIQUES
def w2v_feature_selection(train_df, test_df, w2v_model):
   start = datetime.now()
   print('Started Word2Vec feature selection...')

   # Feature preprocessing
   train_df, test_df = df_preprocessing(train_df, test_df)

   # Build Word2Vec features:
   train_df['w2v_feature'] = train_df['tokenized_feature'].apply(lambda x:build_w2v_features(x, w2v_model))
   test_df['w2v_feature'] = test_df['tokenized_feature'].apply(lambda x:build_w2v_features(x, w2v_model))

   x_train = scale(np.array([x for x in train_df['w2v_feature'].values]))   
   x_test = scale(np.array([x for x in test_df['w2v_feature'].values]))  

   print('Finished Word2Vec feature selection in: ', datetime.now()-start) 
   return x_train, x_test 


def weighted_w2v_feature_selection(train_df, test_df, w2v_model):
   start = datetime.now()
   print('Started Weighted Word2Vec feature selection...')

   # Feature preprocessing
   train_df, test_df = df_preprocessing(train_df, test_df)

   # TF-IDF vectorization
   train_vectorizer = tfidf_vectorizer(train_df['tokenized_feature'].apply(lambda x: ' '.join(x)))   
   train_tfidf_vector = dict(zip(train_vectorizer.get_feature_names(), train_vectorizer.idf_))
   test_vectorizer = tfidf_vectorizer(test_df['tokenized_feature'].apply(lambda x: ' '.join(x)))   
   test_tfidf_vector = dict(zip(test_vectorizer.get_feature_names(), test_vectorizer.idf_))

   # Build Word2Vec features:
   train_df['weighted_w2v_feature'] = train_df['tokenized_feature'].apply(lambda x: 
         build_weighted_w2v_features(x, w2v_model, train_tfidf_vector))
   test_df['weighted_w2v_feature'] = test_df['tokenized_feature'].apply(lambda x: 
         build_weighted_w2v_features(x, w2v_model, test_tfidf_vector))

   x_train = scale(np.array([x[0] for x in train_df['weighted_w2v_feature'].values]))   
   x_test = scale(np.array([x[0] for x in test_df['weighted_w2v_feature'].values]))   

   print('Finished Weighted Word2Vec feature selection in: ', datetime.now()-start)
   return x_train, x_test                                                                                           


def tfidf_feature_selection(train_df, test_df):
   '''
   Conducts simple TF-IDF feature selection with predefined parameters
   '''
   print('Started TF-IDF feature selection')
   start = datetime.now()

   # Feature preprocessing
   train_df, test_df = df_preprocessing(train_df, test_df)
   
   # Call vectorizer
   vectorizer = tfidf_vectorizer(train_df['tokenized_feature'].apply(lambda x: ' '.join(x)))   

   x_train = vectorizer.transform(train_df['tokenized_feature'].apply(lambda x: ' '.join(x))).toarray()
   x_test = vectorizer.transform(test_df['tokenized_feature'].apply(lambda x: ' '.join(x))).toarray()

   print('Finished TF-IDF feature selection in: ', datetime.now()-start)
   return x_train, x_test


def lda_feature_selection(train_df, test_df, n_topics):
   print('Simple Latent Dirichlet Allocation...')
   start = datetime.now()

   # TF-IDF Vectorization
   x_train_tfidf, x_test_tfidf = tfidf_feature_selection(train_df, test_df)

   # LDA preprocessing
   train_df, test_df = df_preprocessing(train_df, test_df)
   
   # LDA pipeline
   lda_pipeline = Pipeline([
               ('count_vec', CountVectorizer(min_df=5,
                                             max_features=100000,
                                             analyzer='word',
                                             ngram_range = (1,3))),
               ('lda', LatentDirichletAllocation(n_components = n_topics,
                                                learning_method = 'online',
                                                batch_size = 128,
                                                max_iter = 20,
                                                random_state = 999,
                                                n_jobs = -1)),
               ])

   x_train_lda = lda_pipeline.fit_transform(train_df['tokenized_feature'].apply(lambda x: ' '.join(x)))
   x_test_lda = lda_pipeline.transform(test_df['tokenized_feature'].apply(lambda x: ' '.join(x)))

   # Combining TD-IDF features with LDA features
   x_train = np.hstack((x_train_lda, x_train_tfidf))
   x_test = np.hstack((x_test_lda, x_test_tfidf))

   print('Finished Latent Dirichlet Allocation in: ', datetime.now()-start)
   return x_train, x_test
