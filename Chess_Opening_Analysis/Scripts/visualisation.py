# This script contains various functions
# used to visualize the chess data

import matplotlib.pyplot as plt
import analysis
import numpy as np
from multiprocessing import Pool
from functools import partial
import data_processing as dpr

ngram_length = 1
max_processes = 12

def get_similarity_matrix_per_move(games, move_count):
    games_i = [[game_i[0:i] for game_i in games] for i in range(1, move_count + 1)]
    move_frequencies = [[analysis.ngram_frequencies(game, ngram_length)\
                        for game in games_ii] for games_ii in games_i]
    matrices = [analysis.similarity_matrix(freqs) for freqs in move_frequencies]

    return matrices

def animate_similarity_matrix_development(games, time_step, move_count):
    matrices = get_similarity_matrix_per_move(games, move_count)
    image = plt.imshow(matrices[0])
    plt.colorbar()
    plt.draw()
    plt.pause(5)

    for i, matrix in enumerate(matrices, 1):
        image.set_data(matrix)
        plt.draw()
        plt.title(f'Similarities after {i} moves')
        plt.pause(time_step)
    plt.show()

def graph_similarities_for_game_length(games, params):
    x_vals = np.linspace(1, 400, num = 399)
    matrices = get_similarity_matrix_per_move(games, 399)


    means = []
    mins = []
    maxs = []

    for matrix in matrices:
        mean = analysis.matrix_mean_similarity(matrix)
        means.append(mean)

    print(means[0])

    plt.plot(x_vals, means)
    plt.xlabel('Number of moves played')
    plt.ylabel('Mean similarity between games')
    plt.title(f'Mean similarity after x amount of moves\n Parameters: {params}')
    plt.xlim(0, 200)
    plt.show()
    #plt.fill_between(mins, maxs, color = 'orange')

def distribution_game_length(gamelist, params):
    plt.hist(gamelist, bins=50)
    plt.title(f"Distribution of gamelengths\n Parameters: {params}")
    plt.ylabel("frequency")
    plt.xlabel("game length")
    plt.show()

def multiple_lines(values, dataframe, min_time, max_sample_size, paramgroups):

    for x in range(len(values)):
        df = dataframe
        minval = values[x][0]
        maxval = values[x][1]

        df = df[(df['White Elo'] >= minval) & (df['White Elo'] < maxval) & (df['Black Elo'] >= minval) & (df['Black Elo'] < maxval) & (df['TimeControl'] >= min_time)]
        games = df['Moves']
        games = games[0: max_sample_size - 1]

        x_vals = np.linspace(1, 200, num = 199)
        matrices = get_similarity_matrix_per_move(games, 199)

        means = []
        mins = []
        maxs = []

        for matrix in matrices:
            mean, min, max = analysis.matrix_mean_similarity(matrix)
            means.append(mean)
            mins.append(min)
            maxs.append(max)

        print(means[0])

        if minval == 2000 and maxval == 3000:
            plt.plot(x_vals, means, label = f"{minval}, {maxval}", alpha = 1)
        else:
            plt.plot(x_vals, means, label = f"{minval}, {maxval}", alpha = 0.2)

    plt.legend()
    plt.title(paramgroups)
    plt.xlabel('Number of moves played')
    plt.ylabel('Mean similarity between games')
    plt.xlim(0, 200)
    plt.show()

def get_mean_similarity_over_time(max_sample_size, games):
    move_count = 400
    if len(games) >= max_sample_size:
        new_games = dpr.get_random_sample(games, max_sample_size)
    means = [analysis.matrix_mean_similarity(matrix) for matrix in analysis.get_similarity_matrix_per_move(games, move_count - 1)]
    return means

def compare_data_subsets(dataset, filters, move_count, max_sample_size):
    subsets_df = [dpr.filter_dataframe(dataset, filter) for filter in filters]
    subsets_games = [list(subset['Moves']) for subset in subsets_df]
    x_vals = np.arange(move_count)
    subset_means = []
    part = partial(analysis.get_mean_similarity_over_time, max_sample_size)
    results = []

    with Pool(max_processes) as p:
        results = p.map(part, subsets_games)

    for i, result in enumerate(results):
        elorange = (filters[i]['White Elo'][0], filters[i]['White Elo'][1])
        lbl = f'Elo range: {elorange}'
        plt.plot(x_vals, result, alpha = 0.5, label = lbl)
    plt.title(f'Mean similarity over {max_sample_size} samples')
    plt.xlim(0, 200)
    plt.legend()
    plt.show()

def test(games, i):
    freqs = []
    for game in games:
        moves = analysis.get_game_moves(game)
        if(i >= len(moves)): moves = [' ']
        else: moves = moves[i:]
        freqs.append(analysis.ngram_freqs(moves, ngram_length))
    similarities = analysis.get_similarities(freqs)
    return np.mean(similarities)

def graph_opening_significance(games, params):
    means = []

    games_i = []
    starting_moves = np.arange(0, 99)
    test2 = partial(test, games)

    with Pool(max_processes) as p:
        results = p.map(test2, starting_moves)
    means = results

    # Deprecated implementation without multiprocessing
    # for i in range(0, 100):
    #     freqs = []
    #     for game in games:
    #         moves = analysis.get_game_moves(game)
    #         if(i >= len(moves)): moves = [' ']
    #         else: moves = moves[i:]
    #
    #         freqs.append(analysis.ngram_freqs(moves, ngram_length))
    #
    #     similarities = analysis.get_similarities(freqs)
    #     means.append(np.mean(similarities))


    x_vals = np.linspace(1, len(means) + 1, num = len(means))

    plt.plot(x_vals, means)
    plt.xlabel('Starting move')
    plt.ylabel('Mean similarity between games')
    plt.title(f'Mean similarity with the game starting at move x\n Parameters: {params}')
    plt.xlim(1, len(means))
    # plt.savefig('Graphs/new_img.png')
    plt.show()

def graph_elo_diff(df, diffs, max_sample_size):
    move_count = 200
    part = partial(analysis.get_mean_similarity_over_time, max_sample_size)
    gamesets = []
    results = []
    x_vals = np.arange(move_count)

    df['diff'] = df['White Elo'] - df['Black Elo']

    for diff in diffs:
        games = list(df[(df['diff'] >= diff)]['Moves'])
        if len(games) > max_sample_size:
            games = dpr.get_random_sample(games, max_sample_size)
        gamesets.append(games)

    with Pool(max_processes) as p:
        results = p.map(part, gamesets)

    for i, result in enumerate(results):
        lbl = f'Min Elo difference: {diffs[i]}'
        plt.plot(x_vals, result, alpha = 0.5, label = lbl)
    plt.title(f'Mean similarity over {max_sample_size} samples')
    plt.xlim(0, 200)
    plt.legend()
    plt.show()

def graph_starting_moves(games):
    moves = [analysis.get_game_moves(game) for game in games]
    starting_moves = [moves_i[0] for moves_i in moves]
    for i, move in enumerate(starting_moves):
        if len(move) == 1:
            starting_moves.remove(move)

    plt.hist(starting_moves, align = 'left')
    plt.title('Distribution of opening moves', fontsize = 18)
    plt.ylabel('Frequency', fontsize = 18)
    plt.yscale('log')
    plt.show()

def graph_similarities_for_game_length_two_plots(game1,game2, params):
    x_vals = np.arange(199)
    matrice1 = get_similarity_matrix_per_move(game1, 199)
    matrice2 = get_similarity_matrix_per_move(game2, 199)


    means = []
    means2 = []
    mins = []
    maxs = []

    for matrix in matrice1:
        mean = analysis.matrix_mean_similarity(matrix)
        means.append(mean)

    for matrix2 in matrice2:
        mean2 = analysis.matrix_mean_similarity(matrix2)
        means2.append(mean2)


    print(means[0])

    plt.plot(x_vals, means)
    plt.plot(x_vals, means2)
    plt.xlabel('Number of moves played')
    plt.ylabel('Mean similarity between games')
    plt.title(f'Mean similarity after x amount of moves\n Parameters: {params}')
    plt.xlim(0, 199)
    plt.show()
    #plt.fill_between(mins, maxs, color = 'orange')
