import numpy as np
import pandas as pd
import math
import random
import re
import gensim
import nltk
from nltk.corpus import stopwords
from collections import Counter
from fuzzywuzzy import fuzz
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def jaccard(q1, q2):
	wic = set(q1).intersection(set(q2))
	uw = set(q2).union(q2)
	if len(uw) == 0:
		uw = [1]
	return (len(wic) / len(uw))

def common_words(q1, q2):
	return len(set(q1).intersection(set(q2)))

def wc_diff(q1, q2):
	return abs(len(q1) - len(q2))

def wc_ratio_unique(q1, q2):
	l1 = len(set(q1)) * 1.0
	l2 = len(set(q2))
	if l2 == 0:
		return np.nan
	if l1 / l2:
		return (l2 / l1)*1.0
	else:
		return (l1 / l2)*1.0

cos_sim = []
WORD = re.compile(r'\w+')
def cosine_sim(question1, question2, flag):
	#---------FLAG IS ZERO '0' FOR TRAIN DATA & ONE '1' FOR TEST DATA-----------#
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
		if(flag==0):
			cos_sim.append(0.0)
		elif(flag==1):
			cos_sim_test.append(0.0)
	else:
		if(flag==0):
			cos_sim.append(float(nume) / deno)
		elif(flag==1):
			cos_sim_test.append(float(nume) / deno)


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

	lemm = WordNetLemmatizer()
	if type(text) is not float:
		text = [lemm.lemmatize(i) for i in text.split()]
		text = ' '.join(text)
	else:
		return ""

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

#   ************************ 	POS - FEATURES STARTS FROM HERE		***************************************
dict = {}
def make_dict(tags, train_question1, train_question2, train):
	for i in range(sample):
		if train['is_duplicate'][i]==1:
			words1 = nltk.pos_tag(nltk.word_tokenize(train_question1[i]))
			words2 = nltk.pos_tag(nltk.word_tokenize(train_question2[i]))
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
tags=[noun_tags, verb_tags,adjective_tags ,adverb_tags]

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

#*************************************************************************************************************

train = pd.read_csv("edited_train.csv")[:10000]
ls=[]
for i in range(10000):
	ls.append(i)

sample = 10000
print ('\n')
print("Size of the sample set is ", sample)
test_size = 500

train_question1 = []
clean_text(train_question1, train.question1, 'train_question1', train)

train_question2 = []
clean_text(train_question2, train.question2, 'train_question2', train)

make_dict(tags, train_question1, train_question2, train)
normalized_wmd = []
jack = []
common_word = []
wc_dif = []
wc_unique = []
same_noun=[]
same_verb=[]
same_adjective=[]
same_adverb=[]
same_wh_word=[]

for x in range(sample):
	t1 = train_question1[x]
	t2 = train_question2[x]
	cosine_sim(t1, t2, 0)
	jack.append(jaccard(t1, t2))
	common_word.append(common_words(t1, t2))
	wc_dif.append(wc_diff(t1, t2))
	wc_unique.append(wc_ratio_unique(t1, t2))
'''
	words1 = nltk.pos_tag(nltk.word_tokenize(t1))
	words2 = nltk.pos_tag(nltk.word_tokenize(t2))
	same_noun.append(same_pos(words1, words2, noun_tags))
	same_verb.append(same_pos(words1, words2, verb_tags))
	same_adjective.append(same_pos(words1, words2, adjective_tags))
	same_adverb.append(same_pos(words1, words2, adverb_tags))
	same_wh_word.append(same_pos(words1, words2, Wh_word_tags))
'''
train['cos_sim'] = cos_sim
train['jackard'] = jack
train['common_words'] = common_word
train['wc_diff'] = wc_dif
train['wc_unique'] = wc_unique
train['fuzz_qratio'] = train.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
train['fuzz_WRatio'] = train.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
train['fuzz_partial_ratio'] = train.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
'''
train['same_noun'] = same_noun
train['same_verb'] = same_verb
train['same_adjective'] = same_adjective
train['same_adverb'] = same_adverb
train['same_wh_word'] = same_wh_word
'''

print ("Every thing done till now.\n")

random.shuffle(ls)
train_cc=train.loc[ls[:9500],:]
test_cc=train.loc[ls[9500:],:]
X_train = train_cc[['cos_sim', 'jackard', 'common_words', 'wc_diff', 'wc_unique', 'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio']]
y_train = train_cc[['is_duplicate']]

X_test = test_cc[['cos_sim', 'jackard', 'common_words', 'wc_diff', 'wc_unique', 'fuzz_qratio', 'fuzz_WRatio', 'fuzz_partial_ratio']]
y_test = test_cc[['is_duplicate']]

clf = MLPClassifier(solver = 'adam', alpha = 1e-5, hidden_layer_sizes = (100,))
clf = clf.fit(X_train, y_train)
print ("Classifier is Trained")
Y_pred = clf.predict(X_test)
print (accuracy_score(y_test, Y_pred))


df = pd.DataFrame(train_cc)
df.to_csv("train_param.csv")