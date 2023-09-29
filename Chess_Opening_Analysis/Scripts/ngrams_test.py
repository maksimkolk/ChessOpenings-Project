# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:10:01 2021

@author: Jonathan Meeng
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

# import for extra images
from scipy.cluster.hierarchy import linkage, dendrogram
#%% Read data
def read_data(file_name):
    """
    First, opens the file.
    Than, skips the non-data in the first line.
    Next, splits the data where each column is a added in a list.
    It returns a list of all de data of each header.
    """
    input_file = open(file_name, 'r')
    count_line = 0
    # headers  = ["id", "rated", "created_at", "last_move_at"	"turns", "victory_status", "winner", "increment_code", "white_id", "white_rating", "black_id"	"black_rating", "moves", "opening_eco", "opening_name", "opening_ply"]
    Id = []
    Rated = []
    created_at = []
    last_move_at = []
    turns = []
    victory_status = []
    winner = []
    increment_code = []
    white_id = []
    white_rating = []
    black_id = []
    black_rating = []
    moves = []
    opening_eco	= []
    opening_name = []
    opening_ply = []

    for line in input_file:
        count_line += 1
        if count_line > 1:
            split_data = line.split(';')
            Id.append(split_data[0])
            Rated.append(split_data[1])
            created_at.append(split_data[2])
            last_move_at.append(split_data[3])
            turns.append(split_data[4])
            victory_status.append(split_data[5])
            winner.append(split_data[6])
            increment_code.append(split_data[7])
            white_id.append(split_data[8])
            white_rating.append(split_data[9])
            black_id.append(split_data[10])
            black_rating.append(split_data[11])
            moves.append(split_data[12])
            opening_eco.append(split_data[13])
            opening_name.append(split_data[14])
            opening_ply.append(split_data[15])


    input_file.close()
    return Id, Rated, created_at, last_move_at, turns, victory_status, winner, increment_code, white_id, white_rating, black_id, black_rating, moves, opening_eco, opening_name, opening_ply

_, _, _, _, _, _, _, _, _, _, _, _, moves, _, opening_name, _ = read_data("Data\Games\games_lichess.csv")

#%% N-grmas

def ngram_frequencies(text, n):

    # splitting the moves of each game and taking the first 10
    word_list = text.split()
    word_list = word_list[:10]
    n_gram_list = []
    
    # taking the n-th move in the movelist and putting it in a list
    for i in range(len(word_list)):
        n_gram = word_list[i:i+n]
        n_gram = " ".join(n_gram)
        n_gram_list.append(n_gram)
    n_gram_dictonary = {}
    
    for n_gram in n_gram_list:

        # add the move to the dictonary and set the move count to 1.
        if n_gram_dictonary.get(n_gram, 0) == 0:
            n_gram_dictonary[n_gram] = 1
        else:

            # add 1 to te move count if the move has already occured.
            n_gram_dictonary[n_gram] += 1
    return n_gram_dictonary

n = 2

# makes the n-gram creating a move counter in the form of a dictonary of a game.
n_grams_moves = [ngram_frequencies(moves[i], n) for i in range(len(moves))]

# creating a list of unique openings
openings = set(opening_name)


# Create a dictonary of n-grmas per game for each opening.
dataset = {}
for i in range(len(opening_name)):
    opening = opening_name[i]
    if dataset.get(opening, 0) == 0:
        dataset[opening] = [n_grams_moves[i]]
    else:
        dataset[opening].append(n_grams_moves[i])

#%% 1 / Cosine similarity
# Function returning the dot product using dictionaries with ngram frequencies
def dot_prod_ngram(vec1, vec2):
    dot_prod = 0
    keys = set(vec1.keys()) & set(vec2.keys())
    for key in keys:
        dot_prod += vec1[key] * vec2[key]
    return dot_prod

# Function returning the similarity matrix of all games using the cosine similarity
def distance_matrix(games):
    distance_matrix = []
    for i, game1 in enumerate(games):
        row = [inverse_cosine_sim(game1, game2) for game2 in games]
        distance_matrix.append(row)

    return distance_matrix

# Function returning the cosine similarity between two chess games
def inverse_cosine_sim(game1, game2):

    dot_prod = dot_prod_ngram(game1, game2)
    game1_squared = [game1[key]**2 for key in game1]
    game2_squared = [game2[key]**2 for key in game2]

    game1_normalized = np.sqrt(sum(game1_squared))
    game2_normalized = np.sqrt(sum(game2_squared))

    cos_sim = dot_prod / (game1_normalized * game2_normalized)
    
    # calculate the distance between the vectors by deviding by the cosine simulatrity
    if cos_sim != 0:
        return 1 / cos_sim
    else:
        return 0
#%% use all the openings with 200 plus games (10 biggest opeings)
count = 0
index_2D = []
opening_game = []
for key in dataset.keys():
    if count == 0:
        if len(dataset[key]) > 200:
            
            # get moves of games of a specific opening
            cluster_dataset = dataset[key][:]
            
            # store opening name in list
            opening_game.append(key)
            
            # store the number of games of the specific openings in list for colour plotting later
            index_2D.append([0, len(dataset[key])])
            count = 1
    else:
        if len(dataset[key]) > 200:
            
            # add moves of games of a specific opening to dataset
            cluster_dataset = cluster_dataset + dataset[key][:]
            
            # store the index of the beginning and end of the added number of games in list for colour plotting later
            index_2D.append([index_2D[-1][-1], index_2D[-1][-1] + len(dataset[key])])
            opening_game.append(key)
            
#%% 10 biggest openings (usaing specific openings)
# keys = ["Van't Kruijs Opening", "Sicilian Defense", "Sicilian Defense: Bowdler Attack", 
#         "French Defense: Knight Variation", "Scotch Game", 
#         "Scandinavian Defense: Mieses-Kotroc Variation", 
#         "Queen's Pawn Game: Mason Attack", "Queen's Pawn Game: Chigorin Variation", 
#         "Scandinavian Defense", "Horwitz Defense"]
# count = 0
# index_2D = []
# opening_game = []
# for key in keys:
#     if count == 0:
#             cluster_dataset = dataset[key][:]
#             opening_game.append(key)
#             index_2D.append([0, len(dataset[key])])
#             count = 1
#     else:
#             cluster_dataset = cluster_dataset + dataset[key][:]
#             index_2D.append([index_2D[-1][-1], index_2D[-1][-1] + len(dataset[key])])
#             opening_game.append(key)
#%% calculate the distance matrix
distance_matrix = distance_matrix(cluster_dataset)
distance_matrix = np.array(distance_matrix)

#%% Calculate the new 2D space for clustering with metric MDS
embedding = MDS(n_components = 2, random_state = 100)

# Transform the distance matrix of N by N games to N by 2.
distance_matrix_transformed = embedding.fit_transform(distance_matrix)

# plot the distances between openinngs in the new 2D space
for i in range(len(index_2D)):
    plt.scatter(distance_matrix_transformed[index_2D[i][0]:index_2D[i][1],0], 
                distance_matrix_transformed[index_2D[i][0]:index_2D[i][1],1], 
                label = opening_game[i])
    
plt.legend(bbox_to_anchor=(1, -0.05), loc='upper right', ncol = int(len(index_2D) / 2))
plt.show()

#%%  Extra images and analysis not part of the presentation.

#Cosine similarity

# Function returning the similarity matrix of all games using the cosine similarity
def cossine_similarity_matrix(games):
    similarity_matrix = []
    for i, game1 in enumerate(games):
        row = [cosine_sim(game1, game2) for game2 in games]
        similarity_matrix.append(row)

    return similarity_matrix

# Function returning the cosine similarity between two chess games
def cosine_sim(game1, game2):

    dot_prod = dot_prod_ngram(game1, game2)
    game1_squared = [game1[key]**2 for key in game1]
    game2_squared = [game2[key]**2 for key in game2]

    game1_normalized = np.sqrt(sum(game1_squared))
    game2_normalized = np.sqrt(sum(game2_squared))

    cos_sim = dot_prod / (game1_normalized * game2_normalized)
    
    return cos_sim


# get cosine simularity for all the games in the cluster_dataset
similarity_matrix = cossine_similarity_matrix(cluster_dataset)
similarity_matrix = np.array(similarity_matrix)

# plot the consine simularity matrix
plt.imshow(similarity_matrix)
plt.colorbar()


# plot dendogram using linkage function form scipy
plt.figure()
link = linkage(similarity_matrix)
dendrogram(link)#, labels = set1_names)
plt.xticks(rotation=90)
plt.ylabel("Height")

# plot the distibution of the cosine simularity
plt.figure()
unique_values = np.unique(similarity_matrix)
plt.hist(unique_values, bins = len(unique_values))
plt.show()
