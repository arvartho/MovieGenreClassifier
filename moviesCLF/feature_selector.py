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
   python feature_selector.py  <input_file_path>  <output_file_path>  <feature_selection_method> 
   
   arguments:
      <file_path>: import csv file
      <output_file_path>: export feature csv file
      <feature_selection_method>: 
         'ordered_occurance' : method 1
         'title_ordered_occurance': method 2
         'tf-idf': method 3
         'w2v': method 4
'''
import pandas as pd
import numpy as np
import sys

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from greek_stemmer import GreekStemmer

from sklearn.model_selection import train_test_split

# Uncomment to get nltk packages
# nltk.download()

def tokenize(doc):
    words = []
    for sentence in sent_tokenize(doc):
        tokens = [stemmer.stem(t.lower()) for t in tokenizer.tokenize(sentence) if t.lower() not in stop_words]
        words += tokens
    return ','.join(words)

def custom_feature_selection(method):

   if method=='ordered_occurance':
      movies_df['feature_str'] = movies_df['plot']
   elif method=='title_ordered_occurance':
      movies_df['feature_str'] = movies_df['title'].str.replace("'", " ") + movies_df['plot']
   
   print('Starting text processing')
   # Add 'plot_tokenized' field in movie dataframe
   movies_df['tokenized_feature'] = movies_df['feature_str'].apply(lambda x: tokenize(str(x)))

   # Create bag of words
   word_bag = [ token.lower() for plot in movies_df['tokenized_feature'].values for token in plot.split(',') ]

   # Create vocabulary 
   vocabulary = set(word_bag)

   print('Vocabulary count: %s' % len(vocabulary))
   print('Total word count: %s' % len(word_bag))
   print('Starting feature selection...')
   # Create word count dictionary
   # word:count(word)
   wc_dict = { word:word_bag.count(word)  for word in vocabulary }
   
   # Order values according to number of occurances
   ordered_values = list(np.argsort(np.asarray(list(wc_dict.values())))[::-1])

   # Order keys according to number of occurances
   ordered_keys = [list(wc_dict.keys())[val] for val in ordered_values]

   # Create dictionary with the ordered number of occurances as values
   # word:occurance_order(word)
   ordered_dict = dict(zip(ordered_keys, range(1, len(ordered_keys)+1)))

   # Add word mapping field in movie dataframe that maps each word in the plot to the ordered dictionary
   movies_df['feature'] = movies_df['tokenized_feature'].apply(lambda x: ','.join([str(ordered_dict[word]) for word in str(x).split(',')]))
   movies_df['feature'].to_csv(output_file_path, header =False)

if __name__ == '__main__':

   if len(sys.argv)==4:
      input_file_path = sys.argv[1]
      output_file_path = sys.argv[2]
      method = sys.argv[3]
      # Read filepath to pandas Dataframe
      movies_df = pd.read_csv(input_file_path)

      # Load stop-words
      stop_words = set(stopwords.words('greek'))

      # Initialize tokenizer
      tokenizer = RegexpTokenizer(r'\w+')

      # Initialize greek stemmer
      stemmer = GreekStemmer()

      print('Feature selection from input file %s to %s with %s' % (input_file_path, output_file_path, method))
      
      if method == 'ordered_occurance':
         custom_feature_selection('ordered_occurance')

      elif method == 'title_ordered_occurance':
         custom_feature_selection('title_ordered_occurance')

      elif method == 'tf-idf':
         pass

      elif method == 'w2v':
         pass

      else:
         print('No such method \"%s\"' % method)
         print(USAGE)

      print('Finished feature selection with "%s". Exported features are in "%s"' % (method, output_file_path))
   else:
        print(USAGE)
    
