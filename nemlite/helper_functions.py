import numpy as np
import  pandas as pd


def save_index(dataframe, new_col_name, offset=0):
    # Save the indexes of the data frame as an np array.
    index_list = np.array(dataframe.index.values)
    # Add an offset to each element of the array.
    offset_index_list = index_list + offset
    # Add the list of indexes as a column to the data frame.
    dataframe[new_col_name] = offset_index_list
    return dataframe


def max_constraint_index(newest_variable_data):
    # Find the maximum constraint index already in use in the constraint matrix.
    max_index = newest_variable_data['ROWINDEX'].max()
    return max_index


def stack_columns(data_in, cols_to_keep, cols_to_stack, type_name, value_name):
    # Wrapping pd.melt to make it easier to use in nemlite context.
    stacked_data = pd.melt(data_in, id_vars=cols_to_keep, value_vars=cols_to_stack,
                           var_name=type_name, value_name=value_name)
    return stacked_data