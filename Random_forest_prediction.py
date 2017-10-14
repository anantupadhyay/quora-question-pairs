'''
	Tested on 100 rows of test data, it was trained on data-set of 10000 rows and only 3 decesion trees were used
	for estimation. 
	Training and testing on whole dataset was taking a lot of time.
'''
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

train = pd.read_csv("train.csv")[:10000]
test = pd.read_csv("test.csv")[:100]
sample = 10000
print(sample)
test_size = 100

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
clean_text(test_question1, test.question1, 'test_question1', test)

train_question2 = []
test_question2 = []
clean_text(train_question2, train.question2, 'train_question2', train)
clean_text(test_question2, test.question2, 'test_question2', test)


'''
for i in range(x,x+1000):
    print(train_question1[i])
    print(train_question2[i])
    print (train['is_duplicate'][i])
'''

dict = {}
def make_dict(tags, train_question1, train_question2, train):
	for i in range(sample):
		if train['is_duplicate'][i]==1:
			words1 = TextBlob(train_question1[i]).tags
			words2 = TextBlob(train_question2[i]).tags
			ls=[[] for j in range(len(tags))]
			ls1=[[] for j in range(len(tags))]
			for word,pos in words1:
				for t in range(len(tags)):
					if pos in tags[t]:
						ls[t].append(word)
			for word,pos in words2:
				for t in range(len(tags)):
					if pos in tags[t]:
						ls1[t].append(word)
			for t in range(len(tags)):
				if len(ls[t])!=0 and len(ls1[t])!=0:
					for each in ls[t]:
						if not each in ls1[t]:
							for each2 in ls1[t]:
								if not each2 in ls[t]:
									dict[each]=each2
									dict[each2] = each
									ls[t].append(each2)
	return
#noun, verb, adjective, adverb
noun_tags=['NN','NNP','NNS','NNPS']
verb_tags=['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjective_tags=['JJ','JJR','JJS']
adverb_tags=['RB','RBR','RBS']
Wh_word_tags=['WDT', 'WP', 'WP$', 'WRB']
tags=[noun_tags, verb_tags,adjective_tags ,adverb_tags ]
make_dict(tags, train_question1, train_question2, train)

#print(dict)
print(len(dict))
'''
for i in range(x, x+1000):
	print(train_question1[i])
	print(train_question2[i])
print(len(dict))
'''
similar_word = []
similar_word_test = []
def word_match(question1, question2, flag):
	#------------FLAG IS ZERO FOR TRAIN DATA AND '1' FOR TEST DATA----------#
	q1words = {}
	q2words = {}
	for word in str(question1).split():
		q1words[word] = 1

	for word in str(question2).split():
		q2words[word] = 1

	sharedwords_q1 = [w for w in q1words.keys() if w in q2words or (w in dict.keys() and dict[w] in q2words)]
	sharedwords_q2 = [w for w in q2words.keys() if w in q1words or (w in dict.keys() and dict[w] in q1words)]
	#print sharedwords_q1
	#print sharedwords_q2

	ln = (len(sharedwords_q1) + len(sharedwords_q2))/(len(q1words) + len(q2words)*1.0)
	if(flag==0):
		similar_word.append(ln)
	elif(flag==1):
		similar_word_test.append(ln)


WORD = re.compile(r'\w+')
cos_sim = []
cos_sim_test = []
def cosine_sim(question1, question2, flag):
	#---------FLAG IS ZERO '0' FOR TRAIN DATA & ONE '1' FOR TEST DATA-----------#
	word1 = WORD.findall(question1)
	word2 = WORD.findall(question2)
	for w in word1:
		if w not in word2:
			if w in dict.keys():
				w = dict[w]

	for w in word2:
		if w not in word1:
			if w in dict.keys():
				w = dict[w]
	vec1 = Counter(word1)
	vec2 = Counter(word2)
	intersec = set(vec1.keys()) & set(vec2.keys())
	nume = sum([vec1[x] * vec2[x] for x in intersec])
	sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	deno = math.sqrt(sum1) * math.sqrt(sum2)
	if not deno:
		if(flag==0):
			cos_sim.append(0.0)
		elif(flag==1):
			cos_sim_test.append(0.0)
	else:
		if(flag==0):
			cos_sim.append(float(nume) / deno)
		elif(flag==1):
			cos_sim_test.append(float(nume) / deno)

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
		if pos in dict.keys():
			if dict[pos] in pos2:
				return 1
	return 0

#--------------TRAINING ON FULL TRAIN SET--------------#
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
	word_match(t1, t2, 0)
	cosine_sim(t1, t2, 0)
	same_noun.append(same_pos(words1, words2, noun_tags))
	same_verb.append(same_pos(words1, words2, verb_tags))
	same_adjective.append(same_pos(words1, words2, adjective_tags))
	same_adverb.append(same_pos(words1, words2, adverb_tags))
	same_wh_word.append(same_pos(words1, words2, Wh_word_tags))


train['similar_word'] = similar_word
train['cos_sim'] = cos_sim
train['same_noun'] = same_noun
train['same_verb'] = same_verb
train['same_adjective'] = same_adjective
train['same_adverb'] = same_adverb
train['same_wh_word'] = same_wh_word

print ("Train data prepared")

#---------------REPEATING THE PROCEDURE OF CREATING FEATURES FOR TESTING DATA--------------#
same_noun_test=[]
same_verb_test=[]
same_adjective_test=[]
same_adverb_test=[]
same_wh_word_test=[]
for x in range(test_size):
	t1 = test_question1[x]
	t2 = test_question2[x]
	words1 = TextBlob(t1).tags
	words2 = TextBlob(t2).tags
	word_match(t1, t2, 1)
	cosine_sim(t1, t2, 1)
	same_noun_test.append(same_pos(words1, words2, noun_tags))
	same_verb_test.append(same_pos(words1, words2, verb_tags))
	same_adjective_test.append(same_pos(words1, words2, adjective_tags))
	same_adverb_test.append(same_pos(words1, words2, adverb_tags))
	same_wh_word_test.append(same_pos(words1, words2, Wh_word_tags))


test['similar_word'] = similar_word_test
test['cos_sim'] = cos_sim_test
test['same_noun'] = same_noun_test
test['same_verb'] = same_verb_test
test['same_adjective'] = same_adjective_test
test['same_adverb'] = same_adverb_test
test['same_wh_word'] = same_wh_word_test

print ("Test data also prepared\n")

X_train = train[['similar_word', 'cos_sim', 'same_noun', 'same_verb', 'same_adjective', 'same_adverb', 'same_wh_word']]
y_train = train[['is_duplicate']]

X_test = test[['similar_word', 'cos_sim', 'same_noun', 'same_verb', 'same_adjective', 'same_adverb', 'same_wh_word']]

clf = RandomForestClassifier(n_estimators=3)
clf = clf.fit(X_train, y_train)
print ("Classifier is Trained")
Y_pred = clf.predict(X_test)

print ("Starting to write the results...")
submission = pd.DataFrame({
        "test_id": test["test_id"],
        "is_duplicate": Y_pred
    })
submission.to_csv('submission2.csv', index=False)
