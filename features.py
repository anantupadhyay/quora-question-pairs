import pandas as pd
import numpy as np
import re, math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from textblob import TextBlob


train = pd.read_csv("train.csv")[:1000]

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
	return

train_question1 = []
clean_text(train_question1, train.question1, 'train_question1', train)

train_question2 = []
clean_text(train_question2, train.question2, 'train_question2', train)

x = 0 
'''
for i in range(x,x+1000):
    print(train_question1[i])
    print(train_question2[i])
    print (train['is_duplicate'][i])
'''
dict = {}
for i in range(x, x+1000):
	if train['is_duplicate'][i]==1:
		words1 = TextBlob(train_question1[i]).tags
		words2 = TextBlob(train_question2[i]).tags
		nn=[]
		nnp=[]
		nns=[]
		nnps=[]
		nn1=[]
		nnp1=[]
		nns1=[]
		nnps1=[]
		for word,pos in words1:
			if pos=='NN':
				nn.append(word)
			if pos=='NNS':
				nns.append(word)
			if pos=='NNP':
				nnp.append(word)
			if pos=='NNPS':
				nnps.append(word)
		for word,pos in words2:
			if pos=='NN':
				nn1.append(word)
			if pos=='NNS':
				nns1.append(word)
			if pos=='NNP':
				nnp1.append(word)
			if pos=='NNPS':
				nnps1.append(word)
		#print("......................................", i)
		#print(nn)
		#print(nn1)
		#print(nns)
		#print(nns1)
		#print(nnp)
		#print(nnp1)
		#print(nnps)
		#print(nnps1)
		if len(nn)!=0 and len(nn1)!=0:
			for each in nn:
				if not each in nn1:
					for each2 in nn1:
						if not each2 in nn:
							dict[each]=each2
							nn.append(each2)
		if len(nnp)!=0 and len(nnp1)!=0:
			for each in nnp:
				if not each in nnp1:
					for each2 in nnp1:
						if not each2 in nnp:
							dict[each]=each2
							nnp.append(each2)
		if len(nns)!=0 and len(nns1)!=0:
			for each in nns:
				if not each in nns1:
					for each2 in nns1:
						if not each2 in nns:
							dict[each]=each2
							nns.append(each2)
		if len(nnps)!=0 and len(nnps1)!=0:
			for each in nnps:
				if not each in nnps1:
					for each2 in nnps1:
						if not each2 in nnps:
							dict[each]=each2
							nnps.append(each2)
'''
for i in range(x, x+1000):
	print(train_question1[i])
	print(train_question2[i])
print(len(dict))
'''
similar_word = []
def word_match(question1, question2):
	q1words = {}
	q2words = {}
	for word in str(question1).split():
		q1words[word] = 1

	for word in str(question2).split():
		q2words[word] = 1

	sharedwords_q1 = [w for w in q1words.keys() if w in q2words]
	sharedwords_q2 = [w for w in q2words.keys() if w in q1words]

	ln = (len(sharedwords_q1) + len(sharedwords_q2))/(len(q1words) + len(q2words)*1.0)
	similar_word.append(ln)


WORD = re.compile(r'\w+')
cos_sim = []
def cosine_sim(question1, question2):
	word1 = WORD.findall(question1)
	word2 = WORD.findall(question2)

	vec1 = Counter(word1)
	vec2 = Counter(word2)
	intersec = set(vec1.keys()) & set(vec2.keys())
	nume = sum([vec1[x] * vec2[x] for x in intersec])
	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	deno = math.sqrt(sum1) * math.sqrt(sum2)
	if not deno:
		cos_sim.append(0.0)
	else:
		cos_sim.append(float(nume) / deno)

#  Only considering first 1000 samples from train set
for x in xrange(1000):
	word_match(train['question1'][x], train['question2'][x])
	cosine_sim(train['question1'][x], train['question2'][x])

#print (cos_sim)
#print (similar_word)

train['similar_word'] = similar_word
train['cos_sim'] = cos_sim

print (train.describe())