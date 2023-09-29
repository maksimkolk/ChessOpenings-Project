# Statistical Data Analysis Research
# December 2021
# Jonas Jansen 13955594
# Jonathan Meeng 14074036
# Maksim Kolk 14054310
# Roshan Baldewsing 12188328

import pandas
import matplotlib.pyplot as plt
import numpy as np
import csv
import analysis
import visualisation
import random

n = 2
max_sample_size = 100

def init_dataframe(csv_filename):
    return pandas.read_csv(csv_filename, sep=',')

def get_moves_from_dataframe(dataframe, column_name):
    moves = [get_game_moves(game) for game in dataframe[column_name]]
    return moves

def get_game_moves(game):
    moves = game.split()
    return moves

def plot_opening_count():
    filename = 'cleaned_data.csv'
    df = init_dataframe(filename)
    counts = {}
    for opening in df['opening_name']:
        if not opening in counts:
            counts[opening] = 1
        else: counts[opening] += 1
    counts = [counts[key] for key in counts]
    counts.sort(reverse=True)
    hist_data = counts
    plt.plot(hist_data)
    plt.yscale('log')
    plt.show()

def get_game_lengths(games):
    return [len(get_game_moves(game)) for game in games]

df = init_dataframe('Data/cleaned data/2016_CvC_cleaned.csv')
min_elo = 2000
max_elo = 2500
min_time = 900

df = df[(df['White Elo'] >= min_elo) & (df['Black Elo'] >= min_elo) & (df['White Elo'] <= max_elo) & (df['Black Elo'] <= max_elo) & (df['TimeControl'] >= min_time)]
games = df['Moves']
if len(games) >= max_sample_size:
    games = games[0: max_sample_size]

# games = [' e4 e5 e6 e7 ', 'd4 d5 d6 d7' ]

#visualisation.animate_similarity_matrix_development(games, 0.02, 200)
#ing the list of plays per game

def n_gram_bootstrap(text, n):
    n_gram_list = text.split()
    # boostrapped value
    bootstrapped_data_list = []


    bootstrapped_values = np.random.choice(n_gram_list, replace = False, size= len(n_gram_list))
    bootstrapped_values = " ".join(bootstrapped_values)


    return bootstrapped_values

# run
scrambled_moves_order = [n_gram_bootstrap(game,1) for game in games]


#visualisation.graph_similarities_for_game_length(scrambled_moves_order,'value')


# visualisation.graph_similarities_for_game_length(scrambled_moves_order,'Bootstrapped without resample games')
#
# params = f'Minimum Elo: {min_elo}, minimum game time: {min_time}, sample size: {max_sample_size}'
# visualisation.graph_similarities_for_game_length(games, params)

visualisation.graph_similarities_for_game_length_two_plots(scrambled_moves_order,games, 'Bootstrapped without resample')


# game lengths to create distribution of game lengths
# lengthlist = get_game_lengths(games)
# visualisation.distribution_game_length(lengthlist, params)

# graph mean similarities for multiple elo values
elovalues = [(2000, 2250), (2250, 2500), (2500, 2750), (2750, 3000), (2000, 3000)]
dframe = init_dataframe('Data/cleaned data/2016_CvC_cleaned.csv')
paramgroups = f"mean similarity after x amount of moves for different elo groups"
visualisation.multiple_lines(elovalues, dframe, min_time, max_sample_size, paramgroups)
