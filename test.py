from collections import Counter
import pandas as pd 
import numpy as np
import re, math
from textblob import TextBlob

inp = "I got a 3.8 GPA. Is it enough to get into top universities like Harvard?"
inp.lower()
words = inp.split()
words1 = TextBlob(inp).tags
print words1
wordCount = Counter(words)
print len(wordCount)

train = pd.read_csv("train.csv")[:100]
similar_word = []
def word_match(question1, question2):
	q1words = {}
	q2words = {}
	for word in str(question1).split():
		q1words[word] = 1

	for word in str(question2).split():
		q2words[word] = 1

	sharedwords_q1 = [w for w in q1words.keys() if w in q2words or (w in dict.keys() and dict[w] in q2words)]
	sharedwords_q2 = [w for w in q2words.keys() if w in q1words or (w in dict.keys() and dict[w] in q1words)]
	print sharedwords_q1
	print sharedwords_q2

	ln = (len(sharedwords_q1) + len(sharedwords_q2))/(len(q1words) + len(q2words)*1.0)
	similar_word.append(ln)
