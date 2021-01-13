import streamlit as st
import numpy as np
import pandas as pd
import nltk
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

st.title('Fireside Chats Data')
tokenizer = RegexpTokenizer(r'\w+') 

@st.cache
def word_display(text):
    word_tokenize = tokenizer.tokenize(text)
    stop_words = stopwords.words('english')
    stop_words.append('the')
    filtered = []
    for w in word_tokenize:
        if w not in stop_words:
            filtered.append(w)
    filtered_dist = FreqDist(filtered)
    return filtered_dist



df = pd.read_csv(r"C:\Users\NH-DB\Desktop\fireside speeches v2.csv", header = 0, encoding = 'unicode_escape')
if st.sidebar.checkbox('Show initial dataframe'):
    df

text = st.sidebar.selectbox(
    'Which date do you want to see the speech from?',
    df['date'])

'The speech from the date you chose is: ', st.text(str(df['speech'][df['date']==text]))

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(df['speech'])

from nltk.stem import PorterStemmer
ps = PorterStemmer()
@st.cache
def top_3_stemmed_words(speech):
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokenize = tokenizer.tokenize(speech)
    stop_words = set(stopwords.words('english'))
    filtered = []
    stemmed = []
    for w in word_tokenize:
        if w not in stop_words:
            filtered.append(w)
    filtered_tags = nltk.pos_tag(filtered)
    for x in filtered_tags:
        if x[1] == 'PRP':
            filtered_tags.remove(x)
    final = []
    for x in filtered_tags:
        final.append(x[0])
    for w in final:
        stemmed.append(ps.stem(w))
    filtered_dist = FreqDist(stemmed)
    first_3_pairs = {k: filtered_dist[k] for k in list(filtered_dist)[:3]}
    first_3 = list(first_3_pairs.keys())
    return first_3
df['top 3 stemmed words'] = df.apply(lambda x: top_3_stemmed_words(x['speech']), axis = 1)

@st.cache
def top_3_word_counts(speech):
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokenize = tokenizer.tokenize(speech)
    stop_words = set(stopwords.words('english'))
    filtered = []
    stemmed = []
    for w in word_tokenize:
        if w not in stop_words:
            filtered.append(w)
    filtered_tags = nltk.pos_tag(filtered)
    for x in filtered_tags:
        if x[1] == 'PRP':
            filtered_tags.remove(x)
    final = []
    for x in filtered_tags:
        final.append(x[0])
    for w in final:
        stemmed.append(ps.stem(w))
    filtered_dist = FreqDist(stemmed)
    first_3_pairs = {k: filtered_dist[k] for k in list(filtered_dist)[:3]}
    return first_3_pairs

df['top 3 word counts']=  df.apply(lambda x: top_3_word_counts(x['speech']), axis = 1)

#vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
speech_matrix =  vectorizer.fit_transform([x for x in df["speech"]])

# Import cosine_similarity to calculate similarity of the speeches
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the similarity distance
similarity_distance = 1 - cosine_similarity(speech_matrix)

#make a heatmap to visually show the similarity distance
sim_df = pd.DataFrame(similarity_distance)
sim_df.index = df['date']
sim_df.columns = df['date']
fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(sim_df, ax = ax, linewidth = 0.15, cmap = "coolwarm")
if st.sidebar.checkbox('Show Similarity Distance Heatmap'):
    st.write(fig)

df2 = pd.DataFrame.from_dict(df['top 3 word counts'])
df2 = df2['top 3 word counts'].apply(pd.Series)
df2 = df2.fillna(0)

import plotly.express as px
fig2 = px.line(df2)
if st.sidebar.checkbox('Show all top 3 word occurences by date line chart'):
    st.write(fig2)

fig3 =px.imshow(df2, x = list(df2.columns), y = df['date'], color_continuous_scale= 'YlGnBu', width = 1000, height = 750)
if st.sidebar.checkbox('Show all top 3 word occurences by date(on y axis) heatmap'):
    st.write(fig3)

df2T = df2.transpose()
fig4 = px.imshow(df2T,x = df['date'], y = list(df2T.index), color_continuous_scale= 'ylorrd', width = 1000, height = 750)
if st.sidebar.checkbox('Show all top 3 word occurences by date(on x axis) heatmap'):
    st.write(fig4)

@st.cache
def top_10_word_counts(speech):
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokenize = tokenizer.tokenize(speech)
    dist = FreqDist(word_tokenize)
    stop_words = set(stopwords.words('english'))
    filtered = []
    stemmed = []
    for w in word_tokenize:
        if w not in stop_words:
            filtered.append(w)
    filtered_tags = nltk.pos_tag(filtered)
    for x in filtered_tags:
        if x[1] == 'PRP':
            filtered_tags.remove(x)
    final = []
    for x in filtered_tags:
        final.append(x[0])
    for w in final:
        stemmed.append(ps.stem(w))
    filtered_dist = FreqDist(stemmed)
    first_10_pairs = {k: filtered_dist[k] for k in list(filtered_dist)[:10]}
    return first_10_pairs

df['top 10 word counts']=  df.apply(lambda x: top_10_word_counts(x['speech']), axis = 1)

df3 = df['top 10 word counts'].apply(pd.Series).fillna(0)

fig5 = px.imshow(df3,x = list(df3.columns), y = df['date'], color_continuous_scale= 'YlGnBu', width = 1000, height = 750)
st.write(fig5)

df3T= df3.transpose()

fig6 = px.imshow(df3T,x = df['date'], y = list(df3T.index), color_continuous_scale= 'ylorrd', width = 1000, height = 750)
st.write(fig6)

if st.sidebar.checkbox('Show final dataframe'):
    df

option = st.sidebar.selectbox(
    'Which date do you want to see the top stemmed words and counts from?',
    df['date'])

option

'The top three stemmed words from the date you chose are: ', df['top 3 word counts'][df['date']==option]