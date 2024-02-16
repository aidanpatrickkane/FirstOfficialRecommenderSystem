#!/usr/bin/env python
# coding: utf-8

#Imports
import pandas as pd
from scipy.parse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load and process the dataset
def load_and_process_data(filepath):
    """Load ratings data and optimize memory usage."""
    ratings_df = pd.read_csv(filepath)
    ratings_df['userId'] = ratings_df['userId'].astype('int32')
    ratings_df['movieId'] = ratings_df['movieId'].astype('int32')
    ratings_df['rating'] = ratings_df['rating'].astype('float32')
    return ratings_df

# Create ratings matrix
def create_ratings_matrix(ratings_df):
  """Pivot ratings df to a matrix."""
  return ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Apply SVD to ratings matrix
def apply_svd(sparse_matrix, n_components=20):
  """Apply Truncated SVD to compress u-r matrix."""
  svd = TruncatedSVD(n_components=n_components)
  return svd.fit_transform(sparse_matrix)

#Calculate cosine similarity
def calculate_cosine_similarity(matrix):
  return cosine_similarity(matrix)

# Fun part -- recommend movies
def recommend_movies(user_id, cosine_sim_df, ratings_df, top_n=5):
  """Recommend movies for a user based on user similarity."""
  similar_users = cosine_sim_df[user_id].sort_values(ascending=False).index[1:top_n+1]
  similar_users_ratings = ratings_df[ratings_df['userId'].isin(similar_users)]
  recommended_movies = similar_users_ratings.groupby('movieId').mean()['rating'].sort_values(ascending=False).head(top_n)
  return recommended_movies

# Main script execution
if __name__ == "__main__":
    # Load and process data
    ratings_df = load_and_process_data('/Users/kaneaidan12/Downloads/archive (1)/rating.csv')
    
    # Create sparse matrix
    ratings_matrix = create_ratings_matrix(ratings_df)
    sparse_matrix = csr_matrix(ratings_matrix.values)
    
    # Apply SVD
    reduced_matrix = apply_svd(sparse_matrix)
    
    # Calculate cosine similarity
    cosine_sim = calculate_cosine_similarity(reduced_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=ratings_matrix.index, columns=ratings_matrix.index)
    
    # Example: Recommend movies for user 1
    recommendations = recommend_movies(1, cosine_sim_df, ratings_df)
    print(recommendations)
