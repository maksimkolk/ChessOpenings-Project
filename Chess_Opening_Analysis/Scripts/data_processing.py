# Importing data from csv into Python
# and processing the raw data

import pandas
import numpy as np

def init_dataframe(filename):
    return pandas.read_csv(filename, sep=',')

# Filters argument formatted as follows:
# {filter1: [min_val, max_val], filter2: [min_val, max_val]}
def filter_dataframe(og_df, filters, columns = ''):
    filtered_df = og_df.copy(deep=True)

    # Grab the correct columns
    if columns != '':
        filtered_df = filtered_df[filters['columns']]

    # Apply filters
    for filter in filters:
        if filter in filtered_df:
            filtered_df = filtered_df[(filtered_df[filter] >= filters[filter][0]) \
            & (filtered_df[filter] <= filters[filter][1])]
    print(len(filtered_df))
    return filtered_df

def get_random_sample(og_sample, sample_size):
    indices = []
    while len(indices) < sample_size:
        rand_index = np.random.randint(0, len(og_sample))
        if rand_index not in indices:
            indices.append(rand_index)
    random_sample = [og_sample[i] for i in indices]
    return random_sample
