#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Data manipulation
import pandas as pd


# In[ ]:


#Necessary library components
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

#Analysis libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Load dataset
ratings_df = pd.read_csv('/Users/kaneaidan12/Downloads/archive (1)/rating.csv')


# In[ ]:


print(ratings_df.head(5))


# In[ ]:


ratings_df.info(memory_usage='deep')


# In[ ]:


ratings_df['userId'] = ratings_df['userId'].astype('int32')
ratings_df['movieId'] = ratings_df['movieId'].astype('int32')
ratings_df['rating'] = ratings_df['rating'].astype('float32')


# In[ ]:


ratings_df.info(memory_usage='deep')


# In[ ]:


ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')


# In[ ]:


ratings_matrix.head(5)


# In[ ]:


from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

sparse_matrix = csr_matrix(ratings_matrix.fillna(0).values)


# In[ ]:


svd = TruncatedSVD(n_components=20)
svd.fit(sparse_matrix)


# In[ ]:


reduced_matrix = svd.transform(sparse_matrix)


# n = 5

# In[ ]:


n = 7


# In[ ]:


del n


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
ratings_matrix_filled = ratings_matrix.fillna(0)


# In[ ]:


#compute cosine similarity
cosine_sim = cosine_similarity(ratings_matrix_filled)


# In[ ]:


#convert to df
cosine_sim_df = pd.DataFrame(cosine_sim, index=ratings_matrix.index, columns=ratings_matrix.index)


# In[ ]:


cosine_sim_df.head(5)

