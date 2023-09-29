# This script contains various functions used to
# analyse the raw chess data

from multiprocessing import Pool
import numpy as np
import concurrent.futures
import data_processing as dpr

ngram_length = 1

# returns game as list of moves
def get_game_moves(game, stop_at=100):
    moves = game.split()
    return moves[0:stop_at]

# Function returning the dot product using dictionaries with ngram frequencies
def dot_prod_ngram(vec1, vec2):
    dot_prod = 0
    keys = set(vec1.keys()) & set(vec2.keys())
    for key in keys:
        dot_prod += vec1[key] * vec2[key]
    return dot_prod

# Function returning the cosine similarity between two chess games
def cosine_sim(game1, game2):

    dot_prod = dot_prod_ngram(game1, game2)
    game1_squared = [game1[key]**2 for key in game1]
    game2_squared = [game2[key]**2 for key in game2]

    game1_normalized = np.sqrt(sum(game1_squared))
    game2_normalized = np.sqrt(sum(game2_squared))

    return dot_prod / (game1_normalized * game2_normalized)

def cos_sim(games):
    game1 = games[0]
    game2 = games[1]

    dot_prod = dot_prod_ngram(game1, game2)
    game1_squared = [game1[key]**2 for key in game1]
    game2_squared = [game2[key]**2 for key in game2]

    game1_normalized = np.sqrt(sum(game1_squared))
    game2_normalized = np.sqrt(sum(game2_squared))

    return dot_prod / (game1_normalized * game2_normalized)

# Function returning the similarity matrix of all games using the cosine similarity
def similarity_matrix(games):
    similarity_matrix = []
    for i, game1 in enumerate(games):
        row = [cosine_sim(game1, game2) for game2 in games]
        similarity_matrix.append(row)

    return similarity_matrix

# Returns the similarities between every vector in a list
def get_similarities(vectors):
    similarities = [cos_sim([vec1, vec2]) for i, vec1 in enumerate(vectors, 1) for vec2 in vectors[i:]]
    return similarities

# Function returning the frequencies of ngrams in a dictionary
def ngram_frequencies(text, n):
    # splitting the moves of each game
    word_list = text.split()
    word_list = word_list[:10]
    n_gram_list = []
    
    for i in range(len(word_list)):
        n_gram = word_list[i:i+n]
        n_gram = " ".join(n_gram)
        n_gram_list.append(n_gram)
    n_gram_dictonary = {}
    
    for n_gram in n_gram_list:
        if n_gram_dictonary.get(n_gram, 0) == 0:
            n_gram_dictonary[n_gram] = 1
        else:
            n_gram_dictonary[n_gram] += 1
    return n_gram_dictonary

def ngram_freqs(moves, n):
    dict = {}
    for move in moves:
        if move not in dict: dict[move] = 1
        else: dict[move] += 1
    return dict

# Function returning the mean similarity in a similarity matrix
def matrix_mean_similarity(matrix):
    similarities = [matrix[i][ii] for i in range(1, len(matrix[0])) for ii in range(0, i + 1)]

    mean = np.mean(similarities)

    return mean

def get_similarity_matrix_per_move(games, move_count):
    games_i = [[game_i[0:i] for game_i in games] for i in range(1, move_count + 1)]
    move_frequencies = [[ngram_frequencies(game, ngram_length)\
                        for game in games_ii] for games_ii in games_i]
    matrices = [similarity_matrix(freqs) for freqs in move_frequencies]
    return matrices

def get_mean_similarity_over_time(max_sample_size, games):
    move_count = 200
    if len(games) >= max_sample_size:
        new_games = dpr.get_random_sample(games, max_sample_size)
        games = new_games
    # games = [get_game_moves(game) for game in games]

    means = [matrix_mean_similarity(matrix) for matrix in get_similarity_matrix_per_move(games, move_count)]
    return means
