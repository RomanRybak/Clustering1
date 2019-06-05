# Clustering1
#Hello! This is one of my college projects where I had to analyze student's performance in a data science class (python)

from matplotlib import style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

style.use('ggplot') 

df = pd.read_csv('cs1-scores.txt',  sep=r"\s+", comment="#")
ffs = df.columns

df['Final'].replace('#', 0, inplace=True)
df = pd.DataFrame(normalize(df, norm='max', axis=0), columns=ffs)                

before = df.dtypes
df1 = df['Final'].astype(float)
max_f = df1.max()
df['Final']=df['Final'].astype(float)/max_f
after = df.dtypes
 
Columns = ["A1", "A2", "Q1", "Q2"]

df['Q1'] = df['Q1']/df['Q1'].max()
df['Q2'] = df['Q2']/df['Q2'].max()
df['A1'] = df['A1']/df['A1'].max()
df['A2'] = df['A2']/df['A2'].max()

df['mean'] = (df['Q1'] + df['Q2'] + df['A1'] + df['A2'])/4 #finding the mean of Qs and As

df=df[['mean','Final']] #new column with mean and final grade

print(df)# Print the new Data Frame 

df_arr=np.array(df)

kmeans = KMeans(7) #diving into 7 clusters
kmeans.fit(df_arr) #fitting them into the graph

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors=["k.","b.","c.","r.","g."]
colors *= 2

for i in range(len(df_arr)):
    plt.plot(df_arr[i][0], df_arr[i][1], colors[labels[i]], markersize=9)
 
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidths=5)
plt.xlabel('Final score')
plt.ylabel('Mean Assignment score')
plt.savefig('Cluster-grades.png')
plt.show()   
