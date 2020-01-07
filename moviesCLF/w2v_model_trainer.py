'''
This python script performs the feature selection from the plot for feeding 
the models. Following is the descriptions of the feature selection methods 
implemented:
   1. Create embedings of ordered word frequency occurances, for example:
      the value 3 in the embeding vector, refers to the third most common
      word in the vocabulary. 
   2. Same as 1 but prepend the movie title in the plot summary
   3. TF-IDF embedings
   4. Word2vec embedings
'''

USAGE = '''
Usage:
   python feature_selector.py  <corpus_file_path>  <model_output_file>  <method> 
   
   arguments:
      <corpus_file_path>: import corpus for training
      <model_output_file>: trained Word2Vec model export path
'''
import pandas as pd
import numpy as np
import multiprocessing
import sys
from gensim.models import Word2Vec
from datetime import datetime
from utility_functions import word_preprocessing

def w2v_training(corpus, model_output_file):
   '''
    Train Word2Vec model from the plot corpus
   '''
   # Prepare model
   start = datetime.now()
   print('Started Word2Vec training...')

   cores = multiprocessing.cpu_count()
   w2v_model = Word2Vec(min_count=3,                        
                        window=5,
                        size=200,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores)
   #BUILD_VOCAB()
   w2v_model.build_vocab(corpus, progress_per=1000)

   #TRAIN()   
   w2v_model.train(corpus, total_examples=w2v_model.corpus_count, epochs=10000, report_delay=1)
   w2v_model.save(model_output_file)
   print('Finished Word2Vec training in : ', datetime.now()-start) 
   return w2v_model


if __name__ == '__main__':
   if len(sys.argv)==3:
      corpus_file_path = sys.argv[1]
      model_output_file = sys.argv[2]
      plot_corpus = pd.read_csv(corpus_file_path)
      plot_corpus['feature_str'] = plot_corpus['title'].str.replace("'", " ") + plot_corpus['plot']
      plot_corpus['tokenized_feature'] = plot_corpus['feature_str'].apply(word_preprocessing)
      corpus = plot_corpus['tokenized_feature']
      w2v_model = w2v_training(corpus, model_output_file)
         
         
