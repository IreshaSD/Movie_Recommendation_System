# Importing necessary libraries
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading data from a CSV file into a Pandas DataFrame, handling potential parsing errors
movies_data = pd.read_csv('G:/ML_Projects/Movie_Recommendation_System/movies.csv', encoding='UTF-8')

# Printing the first 5 rows of the dataframe
movies_data.head()

# Checking the number of rows and columns in the data frame
movies_data.shape

# Selecting relevant features for recommendation
selected_features = ["genres", "keywords", "overview", "tagline", "cast", "director"]

# Replacing null values with empty strings in selected features
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining selected features into a single column
combined_feature = movies_data["genres"] + ' ' + movies_data['keywords'] + ' ' + movies_data['overview'] + ' ' + movies_data["tagline"] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Converting text data to a feature vector using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_feature)

# Calculating cosine similarity scores using feature vectors
similarity = cosine_similarity(feature_vectors)

# Getting user input for a favorite movie
movie_name = input('Enter your favourite movie name: ')

# Converting all movie titles to strings and finding a close match for the user's input
list_of_all_titles = movies_data['title'].tolist()
list_of_all_titles = [str(title) for title in list_of_all_titles]
movie_name = str(movie_name)
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

# Retrieving the indices of the close match movies from the DataFrame
title_column = 'title'
for movie_title in find_close_match:
    matching_indices = movies_data[movies_data[title_column] == movie_title].index.tolist()
    print(f"Indices of '{movie_title}': {matching_indices}")

# Displaying the top 3 close match movies and their indices
close_match = find_close_match[0]
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)

# ... [additional code for handling similar movies based on selected indices]

# Finding similar movies based on the provided movie name
movie_name = input('Enter your favourite movie name: ')
list_of_all_titles = movies_data['title'].tolist()
list_of_all_titles = [str(title) for title in list_of_all_titles]
movie_name = str(movie_name)
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_of_the_movie]))
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# Displaying top 30 suggested movies based on similarity scores
print("Movies suggested for you: \n ")
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i <= 30:
        print(i, ".", title_from_index)
        i += 1
