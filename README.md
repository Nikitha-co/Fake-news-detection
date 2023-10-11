# Fake-news-detection
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

from google.colab import files
files.upload()

df=pd.read_csv('FAKE NEWS DATASET.csv.zip')
df.head()

df.shape
df.drop_duplicates(inplace=True)
df.shape
df.isnull().sum()
df.dropna(axis=0,inplace=True)
df.shape
#combine important columns
df['combined'] =df['Author'] +'  ' +df['Title']
df.head
nltk.download('stopwords')
def process_text(text):
  nopunc  = [char for char in text if char  not in string.punctuation]
  nopunc = ''.join(nopunc)
  

  clean_words = [word for  word in nopunc.split() if word.lower() not in stopwords.words('english')]

  return clean_words
  df['combined'].head().apply(process_text)
  df['combined']
  from sklearn.feature_extraction.text import CountVectorizer
message_bow = CountVectorizer(analyzer=process_text).fit_transform(df['combined'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(message_bow,df['Label'],test_size=0.20,random_state=0)
message_bow.shape
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
print(classifier.predict(x_train))
print(y_train.values)
from sklearn.metrics import classification_report
pred = classifier.predict(x_train)
print(classification_report(y_train,pred))
from sklearn.metrics import classification_report
pred = classifier.predict(x_test)
print(classification_report(y_test,pred))
