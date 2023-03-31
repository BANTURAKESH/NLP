#!/usr/bin/env python
# coding: utf-8

# # Natural Language Toolkit
# 
# * NLTK is a leading platform for building Python programs to work with human language data.
# 
# * It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.
# 
# 

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme()
import sklearn

import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


import pickle


# In[2]:


df = pd.read_csv("spam.csv")
df.head(5)


# In[3]:


df.shape


# # Data Preprocessing
# * Data cleaning 
# * Exploratory data analysis (EDA)
# * Text Preprocessing
# * Modeling Building
# * Model Evaluation
# * Improvement
# * Website
# * Deploy

# # 1.Data cleaning 

# In[4]:


df.info()


# In[5]:


# renaming my columns
df.rename(columns={'Category':'target','Message':'text'},inplace=True)
df.head(5)


# In[6]:


# Converting the target text to numeric values by using LabelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[7]:


df['target'] = encoder.fit_transform(df['target'])


# In[8]:


df.head(5)


# In[9]:


# checking missing values
df.isnull().sum()


# In[10]:


# checking for duplicate values
df.duplicated().sum()


# In[11]:


# remove duplicate values
df = df.drop_duplicates(keep='first')


# In[12]:


df.duplicated().sum()


# In[13]:


df.shape


# # Exploratory data analysis (EDA)

# In[14]:


# checking the value counts of both ham and spam 
df['target'].value_counts()


# In[15]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()


# In[16]:


df["target"].value_counts().plot(kind="bar",color=["salmon", "blue"])


# # 0-----> ham

# # 1-----> spam

# In[17]:


# Data is imbalanced
import nltk


# In[18]:


nltk.download('punkt')


# In[19]:


#number of characters
df['num_characters'] = df['text'].apply(len)


# In[20]:


df.head()


# In[21]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[22]:


df.head()


# In[23]:


# number of sentences
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[24]:


df.head()


# In[25]:


df[['num_characters','num_words','num_sentences']].describe()


# In[26]:


#ham
df[df['target']==0][['num_characters','num_words','num_sentences']].describe()


# In[27]:


#spam
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[28]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'],color='red')
sns.histplot(df[df['target']==1]['num_characters'],color='green')


# In[29]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'],color='red')
sns.histplot(df[df['target']==1]['num_words'],color='green')


# In[30]:


sns.pairplot(df,hue='target')


# In[31]:


sns.heatmap(df.corr(),annot=True)


# # 3.Data Preprocessing
# * Lower case
# * Tokenization
# * Removing special characters
# * Removing stop words and punctuation
# * stemming

# In[32]:


#Removing stop words
from nltk.corpus import stopwords
#stopwords.words("english")


# In[33]:


#punctuation
import string
string.punctuation


# In[34]:


#stemming
from nltk.stem.porter import PorterStemmer
ps =  PorterStemmer()
ps.stem('dancing')


# In[35]:


def transform_text(text):
    text = text.lower() #Lower case
    text = nltk.wordpunct_tokenize(text) #Tokenization
    
    y=[]  #Removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)
            
            
    text = y[:]  #Removing stop words and punctuation
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]  #stemming
    y.clear()
    
    for i in text:
         y.append(ps.stem(i))
            
    return " ".join(y)


# In[36]:


# checking the text


# In[37]:


df['text'][4]


# In[38]:


transform_text("Nah I don't think he goes to usf, he lives around here though")


# In[39]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[40]:


df.head(5)


# In[41]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[42]:


spam_wc = wc.generate(df[df['target'] ==1]['transformed_text'].str.cat(sep=" "))


# In[43]:


plt.figure(figsize=(7,5))
plt.imshow(spam_wc)


# In[44]:


ham_wc = wc.generate(df[df['target'] ==0]['transformed_text'].str.cat(sep=" "))


# In[45]:


plt.figure(figsize=(7,5))
plt.imshow(spam_wc)


# In[46]:


#checking what are the most common words in both spam and ham


# In[47]:


spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[48]:


len(spam_corpus)


# In[49]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[50]:


ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[51]:


len(ham_corpus)


# In[52]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[53]:


# Text vectorization
# using Bag of words
df.head()


# # Model Building 

# In[54]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv= CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[55]:


x = cv.fit_transform(df['transformed_text']).toarray()


# In[56]:


x.shape


# In[57]:


y= df['target'].values


# In[58]:


y.shape


# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[61]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[62]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[63]:


gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[64]:


mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[65]:


bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[66]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# 
