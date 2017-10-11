import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation


train = pd.read_csv("train.csv")[:1000]

print train.isnull().sum()

stop_words = ['the','a','an','and','or','because','as','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','during','to']


def cleaner_function(text, remove_stop_words=True):
	text = re.sub(r"\b([A-Za-z]+)'re\b", '\\1 are', text)
	text = re.sub(r"\b([A-Za-z]+)'s\b", '\\1 is', text)
	text = re.sub(r"\b([A-Za-z]+)'m\b", '\\1 am', text)
	text = re.sub(r"\b([A-Za-z]+)'ve\b", '\\1 have', text)
	text = re.sub(r"\b([A-Za-z]+)'ll\b", '\\1 will', text)
	text = re.sub(r"\b([A-Za-z]+)n't\b", '\\1 not', text)
	text = re.sub(r"\b([A-Za-z]+)'d\b", '\\1 had', text)
	text = re.sub(r"[^A-Za-z0-9]", " ", text)
	if remove_stop_words:
		text = text.split()
		text = [w for w in text if not w in stop_words]
		text = " ".join(text)

	snowball_stemmer = SnowballStemmer('english')
	tx = snowball_stemmer.stem(text)
	#print tx
	return tx

def clean_text(question_list, questions, question_list_name, dataframe):
    
    for question in questions:
    	question.lower()

        question_list.append(cleaner_function(question))


train_question1 = []
clean_text(train_question1, train.question1, 'train_question1', train)

train_question2 = []
clean_text(train_question2, train.question2, 'train_question2', train)

x = 100 
for i in range(x,x+15):
    print(train_question1[i])
    print(train_question2[i])
    print train['is_duplicate'][i]
    print