import pandas as pd
from nemlite import helper_functions as hf


def create_bidding_contribution_to_constraint_matrix(capacity_bids):
    bids_and_indexes = create_bidding_index(capacity_bids.copy())
    unit_max_energy_constraints = create_constraints(bids_and_indexes.copy(), capacity_bids.copy(),
                                                     bid_types=['ENERGY'], max_row_index=-1, rhs_col='MAXENERGY',
                                                     direction='<=')
    max_con_index = hf.max_constraint_index(unit_max_energy_constraints)
    unit_min_energy_constraints = create_constraints(bids_and_indexes.copy(), capacity_bids.copy(),
                                                     bid_types=['ENERGY'], max_row_index=max_con_index,
                                                     rhs_col='MINENERGY', direction='>=')
    bids_and_data = pd.concat([unit_max_energy_constraints, unit_min_energy_constraints], sort=False)
    max_con_index = hf.max_constraint_index(bids_and_data)
    unit_max_fcas_constraints = create_constraints(bids_and_indexes.copy(), capacity_bids.copy(),
                                                   bid_types=['LOWER5MIN', 'LOWER60SEC', 'LOWER6SEC', 'RAISE5MIN',
                                                              'RAISE60SEC', 'RAISE6SEC', 'LOWERREG', 'RAISEREG'],
                                                   max_row_index=max_con_index, rhs_col='MAXAVAIL', direction='<=')
    bids_and_cons = pd.concat([bids_and_data, unit_max_fcas_constraints], sort=False)
    return bids_and_cons, bids_and_indexes


def create_bidding_index(capacity_bids):
    # Stack all the columns that represent an individual variable.
    stacked_bids = hf.stack_columns(capacity_bids, cols_to_keep=['DUID', 'BIDTYPE'],
                                    cols_to_stack=['BANDAVAIL1', 'BANDAVAIL2', 'BANDAVAIL3', 'BANDAVAIL4', 'BANDAVAIL5',
                                                   'BANDAVAIL6', 'BANDAVAIL7', 'BANDAVAIL8', 'BANDAVAIL9',
                                                   'BANDAVAIL10'],
                                    type_name='CAPACITYBAND', value_name='BID')
    # Remove the rows where the fcas bid is equal to zero.
    stacked_bids = remove_gen_bids_with_zero_avail(stacked_bids)

    # Save the index of each bid.
    stacked_bids = stacked_bids.sort_values(['DUID', 'CAPACITYBAND'])
    stacked_bids = stacked_bids.reset_index(drop=True)
    stacked_bids_index = hf.save_index(stacked_bids, 'INDEX')
    return stacked_bids_index


def remove_gen_bids_with_zero_avail(gen_bidding_data):
    # Remove the rows where the max avail is equal to zero.
    gen_bidding_data = gen_bidding_data[(gen_bidding_data['BID'] > 0.01)].copy()
    return gen_bidding_data


def create_constraints(bidding_indexes, capacity_bids, bid_types, max_row_index, rhs_col, direction):
    # Only need energy bid data.
    constraint_rows = capacity_bids[capacity_bids['BIDTYPE'].isin(bid_types)]
    # Create constraint row indexes for each unit.
    constraint_rows = hf.save_index(constraint_rows.reset_index(drop=True), 'ROWINDEX', max_row_index + 1)
    constraint_rows = constraint_rows.loc[:, ['DUID', 'ROWINDEX', rhs_col, 'BIDTYPE']]
    # Merge in bidding data to constraint row data.
    indexes_and_constraints_rows = pd.merge(constraint_rows, bidding_indexes[bidding_indexes['BIDTYPE'].isin(bid_types)],
                                            how='inner', on=['DUID', 'BIDTYPE'])
    # Set enablement type to constraint RHS and LHS is calculated correctly.
    indexes_and_constraints_rows['ENABLEMENTTYPE'] = rhs_col
    # Select just relevant data.
    indexes_and_constraints_rows['RHSCONSTANT'] = indexes_and_constraints_rows[rhs_col]
    indexes_and_constraints_rows = indexes_and_constraints_rows.loc[:, ['INDEX', 'ROWINDEX', 'DUID', 'ENABLEMENTTYPE',
                                                                        'BIDTYPE', 'RHSCONSTANT']]
    indexes_and_constraints_rows['LHSCOEFFICIENTS'] = 1
    indexes_and_constraints_rows['CONSTRAINTTYPE'] = direction
    return indexes_and_constraints_rows

