# Main script, call functions to show results here

import threading
from functools import partial
import pandas
import matplotlib.pyplot as plt
import numpy as np

# Personal imports
import debugger as dbg
import analysis
import visualisation as vis
import data_processing as dpr

def n_gram_bootstrap(text, n):
    n_gram_list = text.split()
    bootstrapped_values = np.random.choice(n_gram_list, replace = False, size= len(n_gram_list))
    bootstrapped_values = " ".join(bootstrapped_values)
    return bootstrapped_values


if __name__ == '__main__':
    max_sample_size = 10**2
    df = dpr.init_dataframe('Data/cleaned data/2016_CvC_cleaned.csv')
    max_time = df['TimeControl'].max()

    filters = [{'White Elo': (1800, 2400), 'Black Elo': (1800, 2400), 'TimeControl': (900, max_time)},
              {'White Elo': (2400, 2600), 'Black Elo': (2400, 2600), 'TimeControl': (900, max_time)},
              {'White Elo': (2600, 2800), 'Black Elo': (2600, 2800), 'TimeControl': (900, max_time)},
              {'White Elo': (2800, 2900), 'Black Elo': (2800, 2900), 'TimeControl': (900, max_time)},
              {'White Elo': (2900, 3600), 'Black Elo': (2900, 3600), 'TimeControl': (900, max_time)}]

    diffs = [50, 100, 200, 300, 400, 500, 750, 1000]
    #df = dpr.filter_dataframe(df, filters)
    #
    # df = df[(df['White Elo'] >= min_elo) & (df['Black Elo'] >= min_elo)\
    # & (df['White Elo'] <= max_elo) & (df['Black Elo'] <= max_elo)\
    # & (df['TimeControl'] >= min_time)]
    #
    games = list(df['Moves'])
    params = ' '
    if len(games) >= max_sample_size:
        games = dpr.get_random_sample(games, max_sample_size)
    # means = vis.get_mean_similarity_over_time(max_sample_size, games)
    games2 = [n_gram_bootstrap(game, 1) for game in games]
    # means2 = vis.get_mean_similarity_over_time(max_sample_size, games2)
    #
    # plt.plot(np.arange(1, len(means) + 1), means, label = 'Games with moves in order')
    # plt.plot(np.arange(1, len(means2) + 1), means2, label = 'Games with moves scrambled')
    # plt.xlabel('Number of moves played', fontsize = 18)
    # plt.ylabel('Mean similarity between games', fontsize = 18)
    # plt.legend()
    # plt.title(f'Mean similarity comparison with scrambled move\n Random sample of {max_sample_size} games', fontsize = 18)
    # plt.xlim(1, len(means) + 1)
    # plt.show()
    # params = f'Elo between: {min_elo} and {max_elo}, minimum game time: {min_time},\n sample size: {len(games)}'

    # test_func = partial(vis.graph_opening_significance, games2, 'Scrambled moves')
    # dbg.execution_time(test_func)

    vis.animate_similarity_matrix_development(games, 0.05, 200)

    # vis.graph_elo_diff(df, diffs, max_sample_size)
    # vis.compare_data_subsets(df, filters, 200, max_sample_size )
    # vis.graph_starting_moves(list(df['Moves']))
