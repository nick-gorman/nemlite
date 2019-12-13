import numpy as np
import pandas as pd
from nemlite import helper_functions as hf


def create_bidding_contribution_to_constraint_matrix(capacity_bids, ns):
    # Create a unique variable index for each bid going into the constraint matrix.
    bids_and_indexes = create_bidding_index(capacity_bids.copy(), ns)

    # Define the row indexes of the constraints arising from the bids data.
    bids_and_data = create_constraint_row_indexes(bids_and_indexes.copy(), capacity_bids, ns)  # type: pd.DataFrame
    # Add additional constraints based on units maximum energy.
    max_con_index = hf.max_constraint_index(bids_and_data)
    unit_max_energy_constraints = create_max_unit_energy_constraints(bids_and_indexes.copy(), capacity_bids.copy(),
                                                                     max_con_index, ns)
    # Combine bidding constraints and maximum energy constraints.
    bids_and_data = pd.concat([bids_and_data, unit_max_energy_constraints], sort=False)  # type: pd.DataFrame
    # Add additional constraints based on unit minimum energy (ramp down rates).
    max_con_index = hf.max_constraint_index(bids_and_data)
    unit_min_energy_constraints = create_min_unit_energy_constraints(bids_and_indexes.copy(), capacity_bids.copy(),
                                                                     max_con_index, ns)
    bids_and_data = pd.concat([bids_and_data, unit_min_energy_constraints], sort=False)  # type: pd.DataFrame

    bids_and_data = pd.merge(bids_and_data,
                             capacity_bids.loc[:, ('DUID', 'BIDTYPE', 'MAXENERGY', 'MINENERGY')],
                             'left', ['DUID', 'BIDTYPE'])

    # For each type of bid variable and type of constraint set the constraint matrix coefficient value.
    bids_coefficients = add_lhs_coefficients(bids_and_data, ns)
    # For each type of constraint set the constraint matrix right hand side value.
    bids_all_data = add_rhs_constant(bids_coefficients, ns)
    return bids_all_data, bids_and_indexes


def create_bidding_index(capacity_bids, ns):
    # Add an additional column that represents the integer variable associated with the FCAS on off decision
    # capacity_bids = insert_col_fcas_integer_variable(capacity_bids, ns.col_fcas_integer_variable)

    # Stack all the columns that represent an individual variable.
    cols_to_keep = [ns.col_unit_name, ns.col_bid_type]
    cols_to_stack = ns.cols_bid_cap_name_list.copy()
    # cols_to_stack.append(ns.col_fcas_integer_variable)
    type_name = ns.col_capacity_band_number
    value_name = ns.col_bid_value
    stacked_bids = hf.stack_columns(capacity_bids, cols_to_keep, cols_to_stack, type_name, value_name)

    # Remove the rows where the fcas bid is equal to zero.
    stacked_bids = remove_gen_bids_with_zero_avail(stacked_bids, ns.col_bid_value)

    # Remove rows where the band number is the FCAS integer variable but the type is energy. These do not exist in
    # reality and are a by product of the way the variable was added to the data set.
    stacked_bids = \
        stacked_bids[(stacked_bids[ns.col_bid_type] != 'ENERGY')
                     | ((stacked_bids[ns.col_bid_type] == 'ENERGY')
                        & (stacked_bids[ns.col_capacity_band_number] != ns.col_fcas_integer_variable))].copy()

    # Save the index of each bid.
    stacked_bids = stacked_bids.reset_index(drop=True)
    new_col_name = ns.col_variable_index
    stacked_bids_index = hf.save_index(stacked_bids, new_col_name)
    return stacked_bids_index


def create_constraint_row_indexes(bidding_indexes, raw_data, ns) -> pd:
    # Create the constraint rows needed to model the interaction between FCAS bids and energy bids. This is effectively
    # creating the space in the constraint matrix for the inequality that make up the FCAS availability trapezium.

    # Just use the FCAS bid data as only FCAS bids generate rows in the constraint matrix.
    just_fcas = raw_data[raw_data[ns.col_bid_type] != 'ENERGY']
    just_info_for_rows = just_fcas.loc[:, (ns.col_unit_name, ns.col_bid_type, ns.col_unit_max_output,
                                           ns.col_low_break_point, ns.col_high_break_point, ns.col_enablement_min,
                                           ns.col_enablement_max)]

    # Stack the unit bid info based on enablement such that an index that corresponds to the constraint matrix row can
    # be generated.
    # cols_to_keep = [ns.col_unit_name, ns.col_bid_type, ns.col_unit_max_output, ns.col_low_break_point,
    #                 ns.col_high_break_point]
    # cols_to_stack = [ns.col_enablement_min, ns.col_enablement_max, ns.col_fcas_integer_variable]
    # type_name = ns.col_enablement_type
    # value_name = ns.col_enablement_value
    # just_fcas_stacked_enablement = stack_columns(just_fcas, cols_to_keep, cols_to_stack, type_name, value_name)

    # Create a new column to store the constraint row index.
    new_col_name = ns.col_constraint_row_index
    row_indexes = hf.save_index(just_info_for_rows, new_col_name)
    # row_indexes = row_indexes.drop(ns.col_unit_max_output, axis=1)

    # Merge the row index data with fcas and energy index data such that information associated with each bid can be
    # mapped to an exact place in the lp constraint matrix.
    bid_data_fcas_row_index = pd.merge(bidding_indexes[bidding_indexes[ns.col_bid_type] != "ENERGY"],
                                       row_indexes, how='left',
                                       on=[ns.col_unit_name, ns.col_bid_type], sort=False)

    # bid_data_energy_row_index = pd.merge(bidding_indexes[bidding_indexes[ns.col_bid_type] == "ENERGY"],
    #                                      just_fcas_stacked_enablement.drop(ns.col_bid_type, axis=1),
    #                                      how='inner',
    #                                      on=[ns.col_unit_name], sort=False)
    #
    # bid_data_fcas_energy_row_index = pd.concat([bid_data_fcas_row_index, bid_data_energy_row_index])

    return bid_data_fcas_row_index


def create_max_unit_energy_constraints(bidding_indexes, raw_data, max_row_index, ns):
    # Only need energy bid data.
    constraint_rows = raw_data[raw_data[ns.col_bid_type] == ns.type_energy].copy()
    # Create a constraint row for each generator bidding into the market.
    constraint_rows = hf.save_index(constraint_rows, ns.col_constraint_row_index, max_row_index + 1)
    constraint_rows = constraint_rows.loc[:, (ns.col_unit_name, ns.col_constraint_row_index)]
    # Merge constraint row indexes with generator bid indexes and max energy data.
    indexes_and_constraints_rows = pd.merge(constraint_rows,
                                            bidding_indexes[bidding_indexes[ns.col_bid_type] == ns.type_energy],
                                            'inner', ns.col_unit_name)
    # Set enablement type, as this is used to flag which calculation to use when constructing the lhs and rhs
    # coefficients of the constraint matrix.
    indexes_and_constraints_rows[ns.col_enablement_type] = ns.col_max_unit_energy
    return indexes_and_constraints_rows


def remove_gen_bids_with_zero_avail(gen_bidding_data, avail_col):
    # Remove the rows where the max avail is equal to zero.
    gen_bidding_data = gen_bidding_data[(gen_bidding_data[avail_col] > 0.01)].copy()
    return gen_bidding_data


def over_ride_max_energy_for_fast_start_plants(original_max_energy_constraints, unit_solutions, fast_start_defs):
    # Merge in the unit solution which provides fast start mode info.
    max_energy_constraints = \
        pd.merge(original_max_energy_constraints, unit_solutions.loc[:, ('DUID', 'DISPATCHMODE')], 'left', 'DUID')
    # Merge in fast start definitions which defines which plants are fast start plants.
    max_energy_constraints = pd.merge(max_energy_constraints, fast_start_defs, 'left', 'DUID')
    # If a fast start plant is operating in mode 1 then set its max energy to 0, else leave as is.
    max_energy_constraints['MAXAVAIL'] = np.where(max_energy_constraints['DISPATCHMODE'] == 1, 0,
                                                  max_energy_constraints['MAXAVAIL'])
    return max_energy_constraints


def create_min_unit_energy_constraints(bidding_indexes, raw_data, max_row_index, ns):
    # Only need energy bid data.
    constraint_rows = raw_data[raw_data[ns.col_bid_type] == ns.type_energy].copy()
    # Create constraint row indexes for each unit.
    constraint_rows = hf.save_index(constraint_rows, ns.col_constraint_row_index, max_row_index + 1)
    constraint_rows = constraint_rows.loc[:, (ns.col_unit_name, ns.col_constraint_row_index)]
    # Merge in bidding data to constraint row data.
    indexes_and_constraints_rows = pd.merge(constraint_rows,
                                            bidding_indexes[bidding_indexes[ns.col_bid_type] == ns.type_energy],
                                            'inner', ns.col_unit_name)
    # Set enablement type to constraint RHS and LHS is calculated correctly.
    indexes_and_constraints_rows[ns.col_enablement_type] = 'MINENERGY'
    # Select just relevant data.
    indexes_and_constraints_rows = indexes_and_constraints_rows.loc[:, ('INDEX', 'ROWINDEX', 'DUID', 'ENABLEMENTTYPE'
                                                                        , 'BIDTYPE')]
    return indexes_and_constraints_rows


def add_lhs_coefficients(bids_data_by_row, ns):
    # For each type of unit bid variable determine the corresponding lhs coefficients in the constraint matrix.

    # LHS for lower enablement limit for the fcas variables
    # bids_data_by_row[ns.col_lhs_coefficients] = np.where(
    #     (bids_data_by_row[ns.col_enablement_type] == ns.col_enablement_min) &
    #     (bids_data_by_row[ns.col_bid_type] != ns.type_energy),
    #     lhs_fcas_enable_min(bids_data_by_row[ns.col_low_break_point],
    #                         bids_data_by_row[ns.col_enablement_value],
    #                         bids_data_by_row[ns.col_unit_max_output])
    #     , 0)
    # LHS for high enablement limit for the fcas variables
    # bids_data_by_row[ns.col_lhs_coefficients] = np.where(
    #     (bids_data_by_row[ns.col_enablement_type] == ns.col_enablement_max)
    #     & (bids_data_by_row[ns.col_bid_type] != ns.type_energy),
    #     lhs_fcas_enable_max(bids_data_by_row[ns.col_high_break_point],
    #                         bids_data_by_row[ns.col_enablement_value],
    #                         bids_data_by_row[ns.col_unit_max_output])
    #     , bids_data_by_row[ns.col_lhs_coefficients])

    # LHS for total fcas limit for fcas variables
    bids_data_by_row[ns.col_lhs_coefficients] = np.where(
        (bids_data_by_row[ns.col_bid_type] != ns.type_energy),
        1, 0)

    # LHS for high enablement limit for energy variables
    # bids_data_by_row[ns.col_lhs_coefficients] = np.where(
    #     (bids_data_by_row[ns.col_enablement_type] == ns.col_enablement_max)
    #     & (bids_data_by_row[ns.col_bid_type] == ns.type_energy),
    #     1, bids_data_by_row[ns.col_lhs_coefficients])

    # LHS for low enablement limit for energy variables
    # bids_data_by_row[ns.col_lhs_coefficients] = np.where(
    #     (bids_data_by_row[ns.col_enablement_type] == ns.col_enablement_min)
    #     & (bids_data_by_row[ns.col_bid_type] == ns.type_energy),
    #     -1, bids_data_by_row[ns.col_lhs_coefficients])

    # LHS for total fcas limit for energy variables
    # bids_data_by_row[ns.col_lhs_coefficients] = np.where(
    #     (bids_data_by_row[ns.col_enablement_type] == ns.col_fcas_integer_variable) &
    #     (bids_data_by_row[ns.col_bid_type] == ns.type_energy),
    #     0, bids_data_by_row[ns.col_lhs_coefficients])

    # LHS for low enablement limit for fcas integer variable
    # bids_data_by_row[ns.col_lhs_coefficients] = np.where(
    #     (bids_data_by_row[ns.col_enablement_type] == ns.col_enablement_min) &
    #     (bids_data_by_row[ns.col_capacity_band_number] == ns.col_fcas_integer_variable),
    #     bids_data_by_row[ns.col_enablement_value], bids_data_by_row[ns.col_lhs_coefficients])

    # LHS for high enablement limit for fcas integer variable
    # bids_data_by_row[ns.col_lhs_coefficients] = np.where(
    #     (bids_data_by_row[ns.col_enablement_type] == ns.col_enablement_max) &
    #     (bids_data_by_row[ns.col_capacity_band_number] == ns.col_fcas_integer_variable),
    #     -1 * (bids_data_by_row[ns.col_enablement_value] - bids_data_by_row[ns.col_max_unit_energy]),
    #     bids_data_by_row[ns.col_lhs_coefficients])

    # LHS for total fcas limit for fcas integer variable
    # bids_data_by_row[ns.col_lhs_coefficients] = np.where(
    #     (bids_data_by_row[ns.col_enablement_type] == ns.col_fcas_integer_variable) &
    #     (bids_data_by_row[ns.col_capacity_band_number] == ns.col_fcas_integer_variable),
    #     -bids_data_by_row[ns.col_unit_max_output], bids_data_by_row[ns.col_lhs_coefficients])

    # LHS for max unit energy
    bids_data_by_row[ns.col_lhs_coefficients] = np.where(
        bids_data_by_row[ns.col_enablement_type] == ns.col_max_unit_energy,
        1, bids_data_by_row[ns.col_lhs_coefficients])

    # LHS for min unit energy
    bids_data_by_row[ns.col_lhs_coefficients] = np.where(
        bids_data_by_row[ns.col_enablement_type] == 'MINENERGY',
        1, bids_data_by_row[ns.col_lhs_coefficients])

    return bids_data_by_row


def add_rhs_constant(combined_bids_col, ns):
    # For each type of unit bid variable determine the corresponding rhs constant in the constraint matrix.

    combined_bids_col[ns.col_rhs_constant] = combined_bids_col['MAXAVAIL']

    # RHS for max energy constraint.
    combined_bids_col[ns.col_rhs_constant] = np.where(
        (combined_bids_col[ns.col_enablement_type] == ns.col_max_unit_energy),
        combined_bids_col[ns.col_max_unit_energy], combined_bids_col[ns.col_rhs_constant])

    # RHS for max FCAS constraint.
    # combined_bids_col[ns.col_rhs_constant] = np.where(
    #     (combined_bids_col[ns.col_enablement_type] == 'MAXAVAIL'),
    #     combined_bids_col[ns.col_max_unit_energy], combined_bids_col[ns.col_rhs_constant])

    # RHS for min energy constraint.
    combined_bids_col[ns.col_rhs_constant] = np.where(
        (combined_bids_col[ns.col_enablement_type] == 'MINENERGY'),
        combined_bids_col['MINENERGY'], combined_bids_col[ns.col_rhs_constant])

    return combined_bids_col
