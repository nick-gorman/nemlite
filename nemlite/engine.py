import pandas as pd
import numpy as np
from time import time, perf_counter
from nemlite import declare_names as dn
from nemlite import solver_interface
from nemlite import bid_constraints
from nemlite import helper_functions as hf
from nemlite import pre_process_bids
from nemlite import joint_fcas_energy_constraints
from nemlite import interconnectors


def run(dispatch_unit_information, dispatch_unit_capacity_bids, initial_conditions, regulated_interconnectors,
        regional_demand, dispatch_unit_price_bids, regulated_interconnectors_loss_model, connection_point_constraints,
        interconnector_constraints, constraint_data, region_constraints, regulated_interconnector_loss_factor_model,
        market_interconnectors, market_interconnector_price_bids, market_interconnector_capacity_bids,
        market_cap_and_floor):
    # Create an object that holds the AEMO names for data table column names.
    ns = dn.declare_names()
    # dummy
    # Initialise list to store results.
    results_datetime = []
    results_service = []
    results_state = []
    results_price = []

    regions_to_price = list(regional_demand['REGIONID'])
    # Create a linear programing problem and definitions of the problem variables.
    combined_constraints, rhs_and_inequality_types, objective_coefficients, \
    var_definitions, inter_variable_bounds, region_req_by_row, duid_and_link_data = \
        create_lp_as_dataframes(dispatch_unit_information, dispatch_unit_capacity_bids, initial_conditions,
                                regulated_interconnectors,
                                regional_demand, dispatch_unit_price_bids, regulated_interconnectors_loss_model,
                                ns, connection_point_constraints, interconnector_constraints, constraint_data,
                                region_constraints, regulated_interconnector_loss_factor_model,
                                market_interconnectors, market_interconnector_price_bids,
                                market_interconnector_capacity_bids)

    # Solve a number of variations of the base problem. These are the problem with the actual loads for each region
    # and an extra solve for each region where a marginal load of 1 MW is added. The dispatches of each
    # dispatch unit are returned for each solve, as well as the interconnector flows.
    dispatches, inter_flows = \
        solver_interface.solve_lp(var_definitions, inter_variable_bounds, combined_constraints,
                                  objective_coefficients, rhs_and_inequality_types, region_req_by_row,
                                  regions_to_price)

    # The price of energy in each region is calculated. This is done by comparing the results of the base dispatch
    # with the results of dispatches with the marginal load added.
    results_price, results_service, results_state = \
        price_regions(dispatches['BASERUN'], dispatches, duid_and_link_data, results_price, results_service,
                      results_state, regions_to_price, market_cap_and_floor)
    # Turn the results lists into a pandas dataframe, note currently only the energy service is priced.
    results_dataframe = pd.DataFrame({'State': results_state, 'Service': results_service,
                                      'Price': results_price})

    return results_dataframe, dispatches, inter_flows


def create_lp_as_dataframes(gen_info_raw, capacity_bids_raw, unit_solution_raw, inter_direct_raw,
                            region_req_raw, price_bids_raw, inter_seg_definitions,
                            ns, con_point_constraints, inter_gen_constraints, gen_con_data,
                            region_constraints, inter_demand_coefficients, mnsp_inter, mnsp_price_bids,
                            mnsp_capacity_bids):
    t1 = time()
    # Convert data formatted in the style and layout of AEMO public data files to the format of a linear program, i.e
    # return a constraint matrix, objective function, also return additional information needed to create a pulp lp
    # object.

    # Create a data frame that maps each generator to its region.
    duid_info = gen_info_raw.loc[:, (ns.col_unit_name, ns.col_region_id, ns.col_loss_factor, 'DISTRIBUTIONLOSSFACTOR',
                                     ns.col_dispatch_type, 'CONNECTIONPOINTID')]

    # Create a list of the plants operating in fast start mode.
    # fast_start_du = define_fast_start_plants(price_bids_raw.copy())
    capacity_bids_scaled = pre_process_bids.filter_and_scale(capacity_bids_raw.copy(), unit_solution_raw)
    # Create a data frame of the constraints that define generator capacity bids in the energy and FCAS markets. A
    # data frame that defines each variable used to represent the the bids in the linear program is also returned.
    bidding_constraints, bid_variable_data = \
        bid_constraints.create_bidding_contribution_to_constraint_matrix(capacity_bids_scaled.copy())

    # Find the current maximum index of the system variable, so new variable can be assigned correct unique indexes.
    max_var_index = hf.max_variable_index(bidding_constraints)

    # Find the current maximum index of the system constraints, so new constraints can be assigned correct unique
    # indexes.
    max_con_index = hf.max_constraint_index(bidding_constraints)
    joint_capacity_constraints = joint_fcas_energy_constraints.create_joint_capacity_constraints(
        bid_variable_data, capacity_bids_scaled, unit_solution_raw, max_con_index)

    # Create inter variables with indexes
    inter_variable_indexes = interconnectors.index_inter(inter_direct_raw, inter_seg_definitions, max_var_index)

    # Create an upper bound for each inter variable
    inter_bounds = interconnectors.add_inter_bounds(inter_variable_indexes, inter_seg_definitions)

    # Find the current maximum index of the system variable, so new variable can be assigned correct unique indexes.
    max_var_index = hf.max_variable_index(inter_variable_indexes)

    # Create the RHS sides of the constraints that force regional demand and FCAS requirements to be met.
    max_con_index = hf.max_constraint_index(joint_capacity_constraints)
    region_req_by_row = create_region_req_contribution_to_constraint_matrix(region_req_raw, max_con_index)

    # Create the coefficients that determine how much a generator contributes to meeting a regions requirements.
    req_row_coefficients = create_region_req_coefficients(duid_info, region_req_by_row, bid_variable_data, ns)

    # Create the coefficients that determine how much an interconnector contributes to meeting a regions requirements.
    # For each segment of an interconnector calculate its loss percentage.
    inter_segments_loss_factors = interconnectors.calculate_loss_factors_for_inter_segments(
        inter_bounds.copy(), region_req_raw, inter_demand_coefficients, inter_direct_raw)

    # For each segment of an interconnector assign it indexes such that its flows are attributed to the correct regions.
    req_row_indexes_for_inter = interconnectors.create_req_row_indexes_for_inter(inter_segments_loss_factors,
                                                                                 region_req_by_row)

    # Convert the loss percentages of interconnectors into contribution coefficients.
    req_row_indexes_coefficients_for_inter = interconnectors.convert_contribution_coefficients(
        req_row_indexes_for_inter, inter_direct_raw)

    # Split out mnsp region segments
    mnsp_to_region_segments, _ = interconnectors.match_against_inter_data(
        inter_variable_indexes, inter_direct_raw)

    # Filter out mnsp interconnectors that are not specified as type 'MNSP' in the general interconnector data.
    mnsp_inter, _ = interconnectors.match_against_inter_data(mnsp_inter, inter_direct_raw)

    # Create a set of indexes for the mnsp links.
    mnsp_link_indexes = interconnectors.create_mnsp_link_indexes(mnsp_capacity_bids, max_var_index)

    # For mnsp links to connect to the interconnector loss model
    max_con_index = hf.max_constraint_index(region_req_by_row)
    constraints_coupling_links_to_interconnector = \
        interconnectors.create_from_region_mnsp_region_requirement_constraints(
            mnsp_link_indexes.copy(), mnsp_inter.copy(), mnsp_to_region_segments.copy(), max_con_index)

    # Create mnsp objective coefficients
    mnsp_objective_coefficients = interconnectors.create_mnsp_objective_coefficients(
        mnsp_link_indexes, mnsp_price_bids, ns)

    # Create generic constraints, these are the generally the network constraints calculated by AEMO.
    max_con_index = hf.max_constraint_index(constraints_coupling_links_to_interconnector)
    generic_constraints, type_and_rhs = create_generic_constraints(con_point_constraints, inter_gen_constraints,
                                                                   gen_con_data, bid_variable_data,
                                                                   inter_bounds, duid_info, max_con_index,
                                                                   region_constraints)

    # Create the constraint matrix by combining dataframes containing the information on each type of constraint.
    coefficient_data_list = [bidding_constraints,
                             req_row_coefficients,
                             req_row_indexes_coefficients_for_inter,
                             generic_constraints,
                             joint_capacity_constraints,
                             constraints_coupling_links_to_interconnector
                             ]

    combined_constraints = combine_constraint_matrix_coefficients_data_frames(coefficient_data_list)

    # Create the objective function coefficients from price bid data
    duid_objective_coefficients = create_objective_coefficients(price_bids_raw, bid_variable_data, duid_info, ns)

    # Add region to bid index data
    objective_coefficients = pd.concat([duid_objective_coefficients,
                                        mnsp_objective_coefficients,
                                        ], sort=False)
    prices_and_indexes = objective_coefficients.loc[:, (ns.col_variable_index, ns.col_bid_value)]
    prices_and_indexes.columns = [ns.col_variable_index, ns.col_price]
    link_data_as_bid_data = create_duid_version_of_link_data(mnsp_link_indexes.copy())
    bid_variable_data = pd.concat([bid_variable_data,
                                   link_data_as_bid_data,
                                   ], sort=False)
    bid_variable_data = pd.merge(bid_variable_data, prices_and_indexes, 'inner', on=ns.col_variable_index)

    # List of inequality types.
    rhs_and_inequality_types = create_inequality_types([bidding_constraints, region_req_by_row, type_and_rhs,
                                                        joint_capacity_constraints,
                                                        constraints_coupling_links_to_interconnector
                                                        ], ns)

    link_loss_factors = mnsp_inter.loc[:, ['LINKID', 'TOREGION', 'TO_REGION_TLF']]
    link_loss_factors.columns = ['DUID', 'REGIONID', 'TRANSMISSIONLOSSFACTOR']
    link_loss_factors['DISTRIBUTIONLOSSFACTOR'] = 1.0
    link_loss_factors['DISPATCHTYPE'] = 'GENERATOR'
    duid_and_link_data = pd.concat([duid_info, link_loss_factors], sort=False)

    return combined_constraints, rhs_and_inequality_types, objective_coefficients, bid_variable_data, \
           req_row_indexes_coefficients_for_inter, region_req_by_row, duid_and_link_data


def create_duid_version_of_link_data(link_data):
    # A mnsp interconnector is modelled as two links one in each interconnected region. Each link behaves similar to
    # a generator. Nemlite treats this links as generators, this function relabels the link data as generator data
    # so it can be merged into the generator variable data frame.
    link_data = link_data.loc[:, ('LINKID', 'CAPACITYBAND', 'BIDVALUE', 'INDEX')]
    link_data.columns = ['DUID', 'CAPACITYBAND', 'BID', 'INDEX']
    link_data['BIDTYPE'] = 'ENERGY'
    return link_data


def create_generic_constraints(connection_point_constraints, inter_constraints, constraint_rhs,
                               bids_and_indexes, indexes_for_inter, gen_info, index_offset, active_region_cons):
    inter_constraints_mnsp = inter_constraints.copy()

    # Select just the data needed about the constrain right hand sides.
    constraint_rhs = constraint_rhs.loc[:, ('GENCONID', 'CONSTRAINTTYPE', 'GENERICCONSTRAINTWEIGHT', 'RHS')]
    # Select just the data needed about the constraints that are being used. Also change the naming of the constraint
    # ID to make it consistent across dataframes being used.

    # Create the set of connection point constraints data.
    # Map the generators to connection points.
    active_duid_constraints = pd.merge(connection_point_constraints, gen_info, 'inner', 'CONNECTIONPOINTID')
    # Select only the bid data needed to construct the constraints.
    bids_and_indexes = bids_and_indexes.loc[:, ('DUID', 'BIDTYPE', 'INDEX', "CAPACITYBAND")]
    # Specifically exclude the FCAS decision variable.
    bids_and_indexes = bids_and_indexes[bids_and_indexes['CAPACITYBAND'] != 'FCASINTEGER'].copy()
    # Map the constraints to specific generator bids.
    active_duid_constraints = pd.merge(active_duid_constraints, bids_and_indexes, 'inner', ['DUID', 'BIDTYPE'])
    # Map in additional information needed about the constraint.
    active_duid_constraints = pd.merge(active_duid_constraints, constraint_rhs, 'inner', ['GENCONID'])
    # Refine data to just that needed for final generic constraint preparation.
    cols_to_keep = ('DUID', 'INDEX', 'FACTOR', 'CONSTRAINTTYPE', 'RHS', 'GENCONID', 'GENERICCONSTRAINTWEIGHT')
    duid_constraints = active_duid_constraints.loc[:, cols_to_keep]

    # Create the set of interconnector constraints data.
    # Select only the interconnector data needed.
    indexes_for_inter = indexes_for_inter.loc[:, ('INTERCONNECTORID', 'BIDTYPE', 'INDEX', "DIRECTION")]
    # Apply only to interconnector energy variables.
    indexes_for_inter_energy_only = indexes_for_inter[indexes_for_inter['BIDTYPE'] == 'ENERGY']
    # Make interconnector variables to constraints.
    inter_constraints = pd.merge(inter_constraints, indexes_for_inter_energy_only, 'inner', 'INTERCONNECTORID')
    # Map in additional constraint data.
    inter_constraints = pd.merge(inter_constraints, constraint_rhs, 'inner', ['GENCONID'])
    # Shorten name for formatting.
    aic = inter_constraints
    # Give interconnector constraint coefficients the correct sign based on their direction of flow.
    inter_constraints['FACTOR'] = np.where(aic['DIRECTION'] == 'REGIONFROM', aic['FACTOR'], -1 * aic['FACTOR'])
    # Refine data to just that needed for final generic constraint preparation.
    cols_to_keep = ('INDEX', 'FACTOR', 'CONSTRAINTTYPE', 'RHS', 'GENCONID', 'GENERICCONSTRAINTWEIGHT')
    inter_constraints = inter_constraints.loc[:, cols_to_keep]

    # Create the set of mnsp interconnector constraints.
    # Map interconnector variables to constraints.
    # inter_constraints_mnsp = pd.merge(inter_constraints_mnsp, mnsp_con_data, 'inner', 'INTERCONNECTORID')
    # Map in additional constraint data.
    # inter_constraints_mnsp = pd.merge(inter_constraints_mnsp, constraint_rhs, 'inner', ['GENCONID'])
    # Shorten name for formatting.
    # aic = inter_constraints_mnsp
    # Give interconnector constraint coefficients the correct sign based on their direction of flow.
    # inter_constraints_mnsp['FACTOR'] = aic['FACTOR'] * aic['LHSFACTOR']
    # Refine data to just that needed for final generic constraint preparation.
    # cols_to_keep = ('INDEX', 'FACTOR', 'CONSTRAINTTYPE', 'RHS', 'GENCONID', 'GENERICCONSTRAINTWEIGHT')
    # inter_constraints_mnsp = inter_constraints_mnsp.loc[:, cols_to_keep]

    # Create the set of region constraints data.
    # Map generators to regions so they can contribute to regional constraints.
    bids_and_indexes_region = pd.merge(bids_and_indexes, gen_info.loc[:, ('DUID', 'REGIONID')], 'inner', 'DUID')
    # Map generators to regional constraints.
    active_region_cons = pd.merge(active_region_cons, bids_and_indexes_region, 'inner', ['REGIONID', 'BIDTYPE'])
    # Map in additional data needed about constraint.
    active_region_cons = pd.merge(active_region_cons, constraint_rhs, 'inner', ['GENCONID'])
    # Refine data to just that need for the final generic constrain preparation.
    cols_to_keep = ('INDEX', 'FACTOR', 'CONSTRAINTTYPE', 'RHS', 'GENCONID', 'GENERICCONSTRAINTWEIGHT')
    region_cons = active_region_cons.loc[:, cols_to_keep]

    # Combine connection point, interconnector and regional constraint information.
    combined_constraints = pd.concat([duid_constraints, inter_constraints, region_cons], sort=False)

    # Prepare a dataframe summarising key constrain information
    type_and_rhs = combined_constraints.loc[:, ('CONSTRAINTTYPE', 'RHS', 'GENCONID', 'GENERICCONSTRAINTWEIGHT')]
    # Make just unique constraints, giving each constraint a specific row index in the constraint matrix.
    type_and_rhs = type_and_rhs.drop_duplicates(subset='GENCONID', keep='first')
    type_and_rhs = hf.save_index(type_and_rhs.reset_index(drop=True), 'ROWINDEX', index_offset + 1)

    # Make the row index into the combined constraints dataframe.
    combined_constraints = pd.merge(combined_constraints, type_and_rhs.loc[:, ['GENCONID', 'ROWINDEX']], 'inner', 'GENCONID')
    # Keep explicit duid constraints over those implied via regional constraints.
    # just_duid_combined_cons = combined_constraints[combined_constraints['INDEX'].isin(bids_and_indexes['INDEX'])]
    # just_duid_combined_cons = just_duid_combined_cons.sort_values('DUID')
    # just_duid_combined_cons = just_duid_combined_cons.groupby(['INDEX', 'ROWINDEX'], as_index=False).first()
    # just_non_duid_combined_cons = combined_constraints[~combined_constraints['INDEX'].isin(bids_and_indexes['INDEX'])]
    # combined_constraints = pd.concat([just_duid_combined_cons, just_non_duid_combined_cons], sort=False)
    # Select just data needed for the constraint matrix.
    combined_constraints = combined_constraints.groupby(['INDEX', 'ROWINDEX'], as_index=False).aggregate({'FACTOR': 'sum'})
    combined_constraints = combined_constraints.loc[:, ('INDEX', 'FACTOR', 'ROWINDEX')]
    # Rename columns to standard constrain matrix names.
    combined_constraints.columns = ['INDEX', 'LHSCOEFFICIENTS', 'ROWINDEX']

    # Rename to standard names.
    type_and_rhs.columns = ['CONSTRAINTTYPE', 'RHSCONSTANT', 'GENCONID', 'CONSTRAINTWEIGHT', 'ROWINDEX']

    return combined_constraints, type_and_rhs


def create_region_req_contribution_to_constraint_matrix(latest_region_req, start_index, ):
    # Create a set of row indexes for each regions requirement constraints.
    region_req_by_row = index_region_constraints(latest_region_req, start_index)
    # Change the naming of columns to match those use for generator bidding constraints. Allows the constraints to
    # be processed together later on.
    region_req_by_row = change_req_naming_to_bid_naming(region_req_by_row)
    region_req_by_row['CONSTRAINTTYPE'] = '='
    return region_req_by_row


def create_objective_coefficients(just_latest_price_bids, bids_all_data, duid_info, ns):
    # Map generator loss factor data to price bid data.
    bids_all_data = pd.merge(bids_all_data, duid_info, 'inner', on=ns.col_unit_name)
    # Combine generator transmission loss factors and distribution loss factors
    bids_all_data[ns.col_loss_factor] = bids_all_data[ns.col_loss_factor] * bids_all_data['DISTRIBUTIONLOSSFACTOR']
    # Reformat price bids so they can be merged with bid variable index data.
    objective_coefficients = create_objective_coefficient_by_duid_type_number(just_latest_price_bids, ns)
    # Add the capacity band type so price bids can be matched to bid variable indexes.
    objective_coefficients = hf.add_capacity_band_type(objective_coefficients, ns)
    # Select the bid variables and just the data we need on them.
    bids_and_just_identifiers = bids_all_data.loc[:, (ns.col_unit_name, ns.col_bid_type, ns.col_capacity_band_number,
                                                      ns.col_variable_index, ns.col_dispatch_type, ns.col_loss_factor,
                                                      'DISTRIBUTIONLOSSFACTOR')]

    # Map the extra data including the indexes to the objective function coefficients.
    objective_coefficients = pd.merge(bids_and_just_identifiers, objective_coefficients, 'left',
                                      [ns.col_unit_name, ns.col_bid_type, ns.col_capacity_band_number])

    # Where an objective function coefficient belongs to a load swap its sign to a negative. As loads are revenue for
    # the market not costs.
    objective_coefficients[ns.col_bid_value] = np.where(
        (objective_coefficients[ns.col_dispatch_type] == ns.type_load) &
        (objective_coefficients[ns.col_bid_type] == ns.type_energy),
        -1 * objective_coefficients[ns.col_bid_value],
        objective_coefficients[ns.col_bid_value])

    # TODO: Check correct values for objective function
    # Increase costs of bids by their loss factors.
    objective_coefficients[ns.col_bid_value] = np.where(objective_coefficients['BIDTYPE'] == ns.type_energy,
                                                        # (
                                                        objective_coefficients[ns.col_bid_value] / \
                                                        objective_coefficients[ns.col_loss_factor],
                                                        #   * objective_coefficients['DISTRIBUTIONLOSSFACTOR'])),
                                                        objective_coefficients[ns.col_bid_value])

    return objective_coefficients


def create_lp_constraint_matrix(coefficient_data_list: list, ns: object):
    # Reformat constraint matrix data so that variable indexes are the columns, and the constraints are the rows.
    # Then convert it to a list of tuples where each tuple is a row.

    # Combine sources of constrain matrix data.
    combined_data = combine_constraint_matrix_coefficients_data_frames(coefficient_data_list, ns)
    # Pivot so indexes values are now columns and row indexes are rows.
    combined_data_matrix_format = combined_data.pivot(ns.col_constraint_row_index, ns.col_variable_index,
                                                      ns.col_lhs_coefficients)
    # Fill in nan values as zeros.
    combined_data_matrix_format = combined_data_matrix_format.fillna(0)
    # Convert to list of tuples.
    constraint_matrix = combined_data_matrix_format.values
    return constraint_matrix, combined_data_matrix_format.index.values


def lhs_fcas_enable_min(low_break_point, low_enablement_value, unit_max_output):
    # Calculate the lhs coefficients for the constraint placed on generators due to their minimum dispatch required for
    # FACAS enablement.
    lhs = (low_break_point - low_enablement_value) / unit_max_output
    return lhs


def lhs_fcas_enable_max(high_break_point: np, high_enablement_value: np, unit_max_output: np) -> np:
    # Calculate the lhs coefficients for the constraint placed on generators due to their max dispatch allowed for
    # FACAS enablement.
    lhs = (high_enablement_value - high_break_point) / unit_max_output
    return lhs


def create_variable_bounds(cap_bidding_df, ns):
    # Select just the columns needed for processing variable bounds.
    variable_bounds = \
        cap_bidding_df.loc[:, (ns.col_unit_name, ns.col_bid_type, ns.col_capacity_band_number, ns.col_bid_value)]
    # Insert a lower bound for each variable
    variable_bounds[ns.col_lower_bound] = 0
    # Rename variable in terms of bounds, rather than bids.
    variable_bounds.columns = [ns.col_unit_name, ns.col_bid_type, ns.col_capacity_band_number, ns.col_upper_bound,
                               ns.col_lower_bound]
    return variable_bounds


def create_objective_coefficient_by_duid_type_number(just_latest_bids, ns):
    # Stack all the columns that represent an individual variable.
    cols_to_keep = [ns.col_unit_name, ns.col_bid_type, 'MINIMUMLOAD']
    cols_to_stack = ns.cols_bid_price_name_list
    type_name = ns.col_price_band_number
    value_name = ns.col_bid_value
    stacked_bids = hf.stack_columns(just_latest_bids, cols_to_keep, cols_to_stack, type_name, value_name)
    return stacked_bids


def insert_col_fcas_integer_variable(gen_variables_in_cols, new_col_name):
    # Create a new column that represents the facas integer decision variable. The value is set to 1 as the variable
    # will act as a binary decision variable taking on the value of either 0 or 1.
    gen_variables_in_cols[new_col_name] = 1
    return gen_variables_in_cols


def index_region_constraints(raw_constraints, max_constraint_row_index):
    # Create constraint rows for regional based constraints.
    row_for_each_constraint = hf.stack_columns(raw_constraints, ['REGIONID'], ['TOTALDEMAND'],
                                               'CONSTRAINTTYPE', 'RHSCONSTANT')
    row_for_each_constraint_indexed = hf.save_index(row_for_each_constraint, 'ROWINDEX',
                                                    max_constraint_row_index + 1)
    return row_for_each_constraint_indexed


def change_req_naming_to_bid_naming(df_with_req_naming):
    # Change the naming conventions of regional requirements to match the naming conventions of bidding, this allows
    # bidding variables to contribute to the correct regional requirements.
    df_with_bid_naming = df_with_req_naming.copy()
    # Apply the mapping of naming defined in the given function.
    df_with_bid_naming['BIDTYPE'] = \
        df_with_bid_naming['CONSTRAINTTYPE'].apply(map_req_naming_to_bid_naming)
    df_with_bid_naming = df_with_bid_naming.drop('CONSTRAINTTYPE', axis=1)
    return df_with_bid_naming


def map_req_naming_to_bid_naming(req_name):
    # Define the mapping of requirement naming to bidding naming.
    if req_name == 'TOTALDEMAND':
        bid_name = 'ENERGY'
    else:
        characters_to_remove = len('LOCALDISPATCH')
        bid_name = req_name[:-characters_to_remove]
    return bid_name


def create_region_req_coefficients(duid_info, region_req_row_map, bids_col_indexes, ns):
    # Give bidding variable the correct constraint matrix coefficients so that it contributes to meeting regional
    # requirements. E.g. a nsw generator's energy bid contributes to meeting the demand in NSW, but not demand in other
    # states.

    # Make sure there are no duplicates.
    bids_col_indexes_no_duplicates = bids_col_indexes.drop_duplicates(subset=[ns.col_variable_index])
    # Merge in duid_info so the location of each generator is known.
    bids_and_region = pd.merge(bids_col_indexes_no_duplicates, duid_info, 'left', ns.col_unit_name)
    # Merge in the matrix rows that define the region requirements.
    bids_region_row = pd.merge(bids_and_region, region_req_row_map, 'inner', [ns.col_region_id, ns.col_bid_type])
    # Specifically define the FCAS decision variables as having no contribution to meeting region requirements.
    bids_region_row[ns.col_lhs_coefficients] = np.where(
        bids_region_row[ns.col_capacity_band_number] == ns.col_fcas_integer_variable,
        0, 1)
    # Define loads that bid into the energy market as having a negative contribution and all other bids as having a
    # positive contribution. Note coefficients are set to 1 which means loss factors have not been applied.
    # TODO: Work out why it appears AEMO does not apply loss factors to generator output.
    bids_region_row[ns.col_lhs_coefficients] = np.where(
        (bids_region_row[ns.col_dispatch_type] == ns.type_load) &
        (bids_region_row[ns.col_bid_type] == ns.type_energy),
        -1 * bids_region_row[ns.col_lhs_coefficients],
        bids_region_row[ns.col_lhs_coefficients])
    bids_region_row = bids_region_row.loc[:, (ns.col_unit_name, ns.col_variable_index, ns.col_constraint_row_index,
                                              ns.col_lhs_coefficients)]
    return bids_region_row


def combine_constraint_matrix_coefficients_data_frames(list_data_sets):
    # Combine all data frames containing constraint matrix coefficients into a single data frame. Just using the
    # index, row inex and coefficient columns.
    data_set_temp_list = [list_data_sets[0].loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS')].copy()]
    for data_set in list_data_sets[1:]:
        data_set_temp_list.append(data_set.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS')].copy())
    combined_coefficient_data = pd.concat(data_set_temp_list)
    return combined_coefficient_data


def rhs_coefficients_in_tuple(list_data_sets, ns):
    # Combine all rhs constants into a single tuple without duplicates and in ascending order.
    combined_rhs_data = list_data_sets[0].drop_duplicates(subset=[ns.col_constraint_row_index])
    for data_set in list_data_sets[1:]:
        data_set_temp = data_set.loc[:, (ns.col_constraint_row_index, ns.col_rhs_constant)].copy()
        data_set_temp = data_set_temp.drop_duplicates(subset=[ns.col_constraint_row_index])
        combined_rhs_data = pd.concat([combined_rhs_data, data_set_temp], sort=False)
    combined_rhs_data = combined_rhs_data.sort_values([ns.col_constraint_row_index], ascending=True)
    combined_rhs_data = tuple(list(combined_rhs_data[ns.col_rhs_constant]))
    return combined_rhs_data


def create_inequality_types(constraint_sets, ns):
    # For each constraint row determine the inequality type and save in a list of ascending order according to row
    # index.
    processed_sets = []
    for constraint_set in constraint_sets:
        constraint_set = constraint_set.drop_duplicates(subset=[ns.col_constraint_row_index]).copy()
        constraint_set = constraint_set.loc[:, (ns.col_constraint_row_index, 'CONSTRAINTTYPE', ns.col_rhs_constant)]
        processed_sets.append(constraint_set)
    # Combine type data, sort by row index and convert to type list.
    combined_type_data = pd.concat(processed_sets, sort=False)
    # combined_type_data = combined_type_data.sort_values([ns.col_constraint_row_index], ascending=True)
    # combined_type_data = list(combined_type_data[ns.col_enquality_type])
    return combined_type_data


def create_objective_coefficients_tuple(objective_coefficients, number_of_variables, ns):
    # pulp needs the objective coefficients in the form of an ordered tuple. This function takes a data frame
    # containing the objective coefficients and converts them into the required tuple format.
    # Create a list of zeros of the required length.
    list_of_objective_coefficients = [0 for i in range(0, int(number_of_variables))]
    # Insert the objective coefficients into the list based on their variable index. If they represent a fcas descision
    # variable then set their cost coefficient to zero.
    for coefficient, index, band_type in zip(list(objective_coefficients[ns.col_bid_value]),
                                             list(objective_coefficients[ns.col_variable_index]),
                                             list(objective_coefficients[ns.col_capacity_band_number])):
        if band_type == ns.col_fcas_integer_variable:
            list_of_objective_coefficients[int(index)] = 0
        else:
            list_of_objective_coefficients[int(index)] = coefficient
    # Convert the the list to a tuple.
    return tuple(list_of_objective_coefficients)


def define_fast_start_plants(bid_costs):
    # If a generator submits a non zero generation profile, then define it as fast start plant.
    bid_costs['ISFAST'] = np.where((bid_costs['T1'] > 0) | (bid_costs['T2'] > 0) | (bid_costs['T3'] > 0) |
                                   (bid_costs['T4'] > 0), 1, 0)
    # bid_costs = bid_costs[bid_costs['BIDTYPE'] == 'ENERGY']
    bid_costs = bid_costs.groupby(['DUID'], as_index=False).first()
    return bid_costs.loc[:, ('DUID', 'ISFAST', 'MINIMUMLOAD', 'T1', 'T2')]


def find_price(base_dispatch, increased_dispatch, gen_info_raw):
    # Select and data needed from the base case dispatch run and rename the dispatch column so the data can be joined
    # with the state marginal load dispatchs.
    base_dispatch_comp = base_dispatch.loc[:, ('DUID', 'CAPACITYBAND', 'BIDTYPE', 'DISPATCHED')]
    base_dispatch_comp.columns = ['DUID', 'CAPACITYBAND', 'BIDTYPE', 'ACTUALDISPATCHED']
    # Select the data need from the marginal load dispatch runs.
    increased_dispatch_comp = increased_dispatch.loc[:, ('DUID', 'CAPACITYBAND', 'BIDTYPE', 'DISPATCHED')]
    increased_dispatch_comp.columns = ['DUID', 'CAPACITYBAND', 'BIDTYPE', 'INCREASEDDISPATCHED']
    # Join the dispatch results so the dispacth of each DUID and bid can be compared.
    dispatch_comp = pd.merge(base_dispatch_comp, increased_dispatch_comp, 'inner', ['DUID', 'CAPACITYBAND', 'BIDTYPE'])
    # Calaculate the change in dispatch for each generator and bid.
    dispatch_comp['DISPATCHCHANGE'] = dispatch_comp['INCREASEDDISPATCHED'] - dispatch_comp['ACTUALDISPATCHED']
    # Select just the generators and bids whose dispatch changed.
    marginal_dispatch = dispatch_comp[(dispatch_comp['DISPATCHCHANGE'] > 0.0001) |
                                      (dispatch_comp['DISPATCHCHANGE'] < -0.0001)]
    # Merge in the base dispatch to provide bid pricing.
    marginal_dispatch = pd.merge(marginal_dispatch, base_dispatch, 'inner', ['DUID', 'CAPACITYBAND', 'BIDTYPE'])
    # Merge in the generator information to provide loss factors.
    marginal_dispatch = pd.merge(marginal_dispatch, gen_info_raw, 'inner', 'DUID')
    md = marginal_dispatch
    # Calculate the price after accounting for losses.
    # md['RELATIVEPRICE'] = np.where(md['BIDTYPE'] == 'ENERGY',
    #                                (md['PRICE'] / (md['TRANSMISSIONLOSSFACTOR'] * md['DISTRIBUTIONLOSSFACTOR'])),
    #                                md['PRICE'])
    md['RELATIVEPRICE'] = md['PRICE']
    # Find the marginal cost of each each marginal generator.
    md['MARGINALCOST'] = md['DISPATCHCHANGE'] * md['RELATIVEPRICE']
    # Sum the marginal cost of each generator to find the total marginal costs.
    marginal_price = md['MARGINALCOST'].sum()
    return marginal_price


def price_regions(base_case_dispatch, pricing_dispatchs, gen_info_raw, results_price, results_service,
                  results_state, regions_to_price, market_cap_and_floor):
    # For the energy service find the marginal price in each region.
    service = 'ENERGY'
    for region in regions_to_price:
        price = find_price(base_case_dispatch, pricing_dispatchs[region], gen_info_raw)
        if price < market_cap_and_floor['MARKETPRICEFLOOR'][0]:
            price = market_cap_and_floor['MARKETPRICEFLOOR'][0]
        if price > market_cap_and_floor['VOLL'][0]:
            price = market_cap_and_floor['VOLL'][0]
        results_price.append(price)
        results_service.append(service)
        results_state.append(region)

    return results_price, results_service, results_state
