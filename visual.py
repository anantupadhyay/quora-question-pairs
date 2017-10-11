import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

df = pd.read_csv('train.csv')
#print df.head()

#print df['is_duplicate'].value_counts()

#print df.isnull().sum()

df = df[df.question2.notnull()]
#print df.describe()

#print df.isnull().sum()
df['len1'] = df['question1'].str.len()
df['len2'] = df['question2'].str.len()

print df.describe()
'''
plt.scatter(df['len1'][:2500], df['len2'][:2500], label='skitscat', color='k', s=25, marker="o")

plt.xlabel('Question 1 length')
plt.ylabel('Question 2 length')
plt.show()
'''

q1 = df['len1'][:100].value_counts()
sns.barplot(q1.index, q1.values, alpha=0.8)
plt.xlabel('Number of Questions')
plt.ylabel('Length of question1')
plt.xticks(rotation='vertical')
plt.show()