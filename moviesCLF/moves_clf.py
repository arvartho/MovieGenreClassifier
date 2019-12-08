"""
The aim of this python script is to calculate and classify
a set of movies by genre.
The dataset used to implement this script contained features as followed:
`url, title, original_title, year, country, genre, plot`
"""

"""
Function that will loop into the genres array and remove the least common genres
In this dataset there occurencies where countries are included in the file as genres,
and by that way we exclude most of them
"""
def getGenres(genresArr):
   genreDict = {}
   for gen in genresArr['genre']:
      gen = gen.split(",")
      for g in gen:
         if g in genreDict:
            genreDict[g] += 1
         else:
            genreDict[g] = 1
   return {k: v for k, v in genreDict.items() if v > 20}

"""
Function getUniqueGenresArray(genresArr, text) will take as inputs
the genresArr as was extracted from the dataset. The genres in the dataset
contain a csv serarated list with the genres correspond to that film.

This function will filter the dataset and expand it to have ['title','plot', [genre1....genreN]]
each of the genre columns will be a binary value

The function will return the above array
"""
def getUniqueGenresArray(genresArr, text):
   
   genreArr = ['title','plot']
   genreDict = getGenres(genresArr)
   for key in genreDict.keys():
      genreArr.append(key)
   df = pd.DataFrame(columns=genreArr)
   finalGenre = []
   line = {}
   for gen, txtTitle, txtPlot in zip(genresArr['genre'], text['title'], text['plot']):
      gen = gen.split(",")
      for g in genreDict.keys():
         line[g] = 0
      for g in gen:
         if g in genreDict:
            line[g] = 1
      line['title'] = txtTitle
      line['plot'] = txtPlot
      finalGenre.append(line)
   return finalGenre

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import csv

# #############################################################################
# Load and remove some columns from dataset like URL, and original_title
# #############################################################################
moviesData = pd.read_csv("../data/movieDB.csv", sep=',', encoding='utf-8')

textData = moviesData.iloc[:,np.r_[1,6]]
genres = moviesData.iloc[:,np.r_[5]]
genresDict = getUniqueGenresArray(genres, textData)
