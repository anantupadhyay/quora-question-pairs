import pandas as pd
import numpy as np
import re, math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.neural_network import MLPClassifier


train = pd.read_csv("edited_train.csv")[:10000]
ls=[]
for i in range(10000):
	ls.append(i)

sample = 10000
print(sample)
test_size = 500

#print (train.isnull().sum())

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
		try:
			float(question)
			question_list.append("Nan")
		except:
			question.lower()
			question_list.append(cleaner_function(question))
	return

train_question1 = []
test_question1 = []
clean_text(train_question1, train.question1, 'train_question1', train)
#clean_text(test_question1, test.question1, 'test_question1', test)

train_question2 = []
test_question2 = []
clean_text(train_question2, train.question2, 'train_question2', train)

noun_tags=['NN','NNP','NNS','NNPS']
verb_tags=['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjective_tags=['JJ','JJR','JJS']
adverb_tags=['RB','RBR','RBS']
Wh_word_tags=['WDT', 'WP', 'WP$', 'WRB']
tags=[noun_tags, verb_tags,adjective_tags ,adverb_tags ]

def same_pos(question1, question2, tag):
	pos1 = [word for word,pos in question1 if pos in tag]
	pos2 = [word for word,pos in question2 if pos in tag]
	if len(pos1)==0 and len(pos2)==0:
		return 1
	if len(pos1)==0 or len(pos2)==0:
		return 0
	for pos in pos1:
		if pos in pos2:
			return 1
	return 0


same_noun=[]
same_verb=[]
same_adjective=[]
same_adverb=[]
same_wh_word=[]

for x in range(sample):
	t1 = train_question1[x]
	t2 = train_question2[x]
	words1 = TextBlob(t1).tags
	words2 = TextBlob(t2).tags
	same_noun.append(same_pos(words1, words2, noun_tags))
	same_verb.append(same_pos(words1, words2, verb_tags))
	same_adjective.append(same_pos(words1, words2, adjective_tags))
	same_adverb.append(same_pos(words1, words2, adverb_tags))
	same_wh_word.append(same_pos(words1, words2, Wh_word_tags))

train['same_noun'] = same_noun
train['same_verb'] = same_verb
train['same_adjective'] = same_adjective
train['same_adverb'] = same_adverb
train['same_wh_word'] = same_wh_word

print ("Train data prepared")


random.shuffle(ls)
train_cc=train.loc[ls[:9500],:]
test_cc=train.loc[ls[9500:],:]
X_train = train_cc[['same_noun', 'same_verb', 'same_adjective', 'same_adverb', 'same_wh_word']]
y_train = train_cc[['is_duplicate']]

X_test = test_cc[['same_noun', 'same_verb', 'same_adjective', 'same_adverb', 'same_wh_word']]


clf = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (100,))
clf = clf.fit(X_train, y_train)
print ("Classifier is Trained")
Y_pred = clf.predict(X_test)

print ("Starting to write the results...")
print(i)
submission = pd.DataFrame({
	"is_duplicate": Y_pred,
	"S.no.": ls[9500:]
	})
st='predicted_class_nn.csv'
submission.to_csv(st, index=False)	