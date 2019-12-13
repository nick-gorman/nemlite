import pandas as pd
import numpy as np
from time import time, perf_counter
from nemlite import declare_names as dn
from nemlite import solver_interface
from nemlite import bid_constraints
from nemlite import helper_functions as hf
from nemlite import pre_process_bids

def run(dispatch_unit_information, dispatch_unit_capacity_bids, initial_conditions, regulated_interconnectors,
        regional_demand, dispatch_unit_price_bids, regulated_interconnectors_loss_model, connection_point_constraints,
        interconnector_constraints, constraint_data, region_constraints, regulated_interconnector_loss_factor_model,
        market_interconnectors, market_interconnector_price_bids, market_interconnector_capacity_bids,
        market_cap_and_floor):
    # Create an object that holds the AEMO names for data table column names.
    ns = dn.declare_names()

    # Initialise list to store results.
    results_datetime = []
    results_service = []
    results_state = []
    results_price = []

    regions_to_price = list(regional_demand['REGIONID'])
    # Create a linear programing problem and definitions of the problem variables.
    combined_constraints, rhs_and_inequality_types, objective_coefficients, \
    var_definitions, inter_variable_bounds, region_req_by_row = \
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
        price_regions(dispatches['BASERUN'], dispatches, dispatch_unit_information, results_price, results_service,
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
        bid_constraints.create_bidding_contribution_to_constraint_matrix(capacity_bids_scaled.copy(), ns)

    # Find the current maximum index of the system variable, so new variable can be assigned correct unique indexes.
    max_var_index = max_variable_index(bidding_constraints)

    # Find the current maximum index of the system constraints, so new constraints can be assigned correct unique
    # indexes.
    max_con_index = hf.max_constraint_index(bidding_constraints)
    joint_capacity_constraints = create_joint_capacity_constraints(bid_variable_data, capacity_bids_scaled,
                                                                   unit_solution_raw, max_con_index)

    # Create inter variables with indexes
    inter_variable_indexes = index_inter(inter_direct_raw, inter_seg_definitions, max_var_index, ns)

    # Create an upper bound for each inter variable
    inter_bounds = add_inter_bounds(inter_variable_indexes, inter_seg_definitions, ns)

    # Find the current maximum index of the system variable, so new variable can be assigned correct unique indexes.
    max_var_index = max_variable_index(inter_variable_indexes)

    # Find the current maximum index of the system constraints, so new constraints can be assigned correct unique
    # indexes.
    max_con_index = hf.max_constraint_index(joint_capacity_constraints)

    # Create constraints that ensure interconnector segments are dispatched the correct order.
    inter_seg_dispatch_order_constraints = \
        create_inter_seg_dispatch_order_constraints(inter_bounds, max_con_index, max_var_index)

    # Create the RHS sides of the constraints that force regional demand and FCAS requirements to be met.
    max_con_index = hf.max_constraint_index(inter_seg_dispatch_order_constraints)
    region_req_by_row = create_region_req_contribution_to_constraint_matrix(region_req_raw, max_con_index, ns)

    # Create the coefficients that determine how much a generator contributes to meeting a regions requirements.
    req_row_coefficients = create_region_req_coefficients(duid_info, region_req_by_row, bid_variable_data, ns)

    # Create the coefficients that determine how much an interconnector contributes to meeting a regions requirements.
    # For each segment of an interconnector calculate its loss percentage.
    inter_segments_loss_factors = calculate_loss_factors_for_inter_segments(inter_bounds.copy(), region_req_raw,
                                                                            inter_demand_coefficients, inter_direct_raw,
                                                                            ns)

    # For each segment of an interconnector assign it indexes such that its flows are attributed to the correct regions.
    req_row_indexes_for_inter = create_req_row_indexes_for_inter(inter_segments_loss_factors, region_req_by_row, ns)
    # Convert the loss percentages of interconnectors into contribution coefficients.
    req_row_indexes_coefficients_for_inter = convert_contribution_coefficients(req_row_indexes_for_inter,
                                                                               inter_direct_raw, ns)

    # Filter out mnsp interconnectors that are not specified as type 'MNSP' in the general interconnector data.
    mnsp_inter = match_against_inter_data(mnsp_inter, inter_direct_raw)

    # Create a set of indexes for the mnsp links.
    # max_var_index = max_variable_index(inter_seg_dispatch_order_constraints)
    mnsp_link_indexes = create_mnsp_link_indexes(mnsp_capacity_bids, max_var_index)

    # Create contribution coefficients for mnsp link variables.
    mnsp_region_requirement_coefficients = create_mnsp_region_requirement_coefficients(mnsp_link_indexes, mnsp_inter,
                                                                                       region_req_by_row)

    # Create mnsp objective coefficients
    mnsp_objective_coefficients = create_mnsp_objective_coefficients(mnsp_link_indexes, mnsp_price_bids, ns)

    # Create mnsp generic constraint data.
    mnsp_con_data = pd.merge(mnsp_link_indexes, mnsp_inter, 'inner', 'LINKID')
    mnsp_con_data = mnsp_con_data.loc[:, ('INTERCONNECTORID', 'LHSFACTOR', 'INDEX')]

    # Create generic constraints, these are the generally the network constraints calculated by AEMO.
    max_con_index = hf.max_constraint_index(region_req_by_row)
    generic_constraints, type_and_rhs = create_generic_constraints(con_point_constraints, inter_gen_constraints,
                                                                   gen_con_data, bid_variable_data,
                                                                   inter_bounds, duid_info, max_con_index,
                                                                   region_constraints, mnsp_con_data)

    # Create the constraint matrix by combining dataframes containing the information on each type of constraint.
    coefficient_data_list = [bidding_constraints,
                             req_row_coefficients,
                             req_row_indexes_coefficients_for_inter,
                             generic_constraints,
                             #     inter_seg_dispatch_order_constraints,
                             mnsp_region_requirement_coefficients,
                             joint_capacity_constraints]

    combined_constraints = combine_constraint_matrix_coefficients_data_frames(coefficient_data_list)

    # Create the objective function coefficients from price bid data
    duid_objective_coefficients = create_objective_coefficients(price_bids_raw, bid_variable_data, duid_info, ns)

    # Add region to bid index data
    objective_coefficients = pd.concat([duid_objective_coefficients, mnsp_objective_coefficients], sort=False)
    prices_and_indexes = objective_coefficients.loc[:, (ns.col_variable_index, ns.col_bid_value)]
    prices_and_indexes.columns = [ns.col_variable_index, ns.col_price]
    link_data_as_bid_data = create_duid_version_of_link_data(mnsp_link_indexes.copy())
    bid_variable_data = pd.concat([bid_variable_data, link_data_as_bid_data], sort=False)
    bid_variable_data = pd.merge(bid_variable_data, prices_and_indexes, 'inner', on=ns.col_variable_index)


    # List of inequality types.
    rhs_and_inequality_types = create_inequality_types(bidding_constraints, region_req_by_row, type_and_rhs,
                                               inter_seg_dispatch_order_constraints,
                                               joint_capacity_constraints,
                                               ns)

    return combined_constraints, rhs_and_inequality_types, objective_coefficients, bid_variable_data, \
           req_row_indexes_coefficients_for_inter, region_req_by_row,


def create_duid_version_of_link_data(link_data):
    # A mnsp interconnector is modelled as two links one in each interconnected region. Each link behaves similar to
    # a generator. Nemlite treats this links as generators, this function relabels the link data as generator data
    # so it can be merged into the generator variable data frame.
    link_data = link_data.loc[:, ('LINKID', 'CAPACITYBAND', 'BIDVALUE', 'INDEX')]
    link_data.columns = ['DUID', 'CAPACITYBAND', 'BID', 'INDEX']
    link_data['BIDTYPE'] = 'ENERGY'
    return link_data


def create_mnsp_objective_coefficients(indexes, price_bids, ns):
    # Create the objective function coefficients for the mnsp interconnectors.
    # Stack the price band data so each one can be maped to and a variable index.
    cols_to_keep = ['LINKID']
    cols_to_stack = ['PRICEBAND1', 'PRICEBAND2', 'PRICEBAND3', 'PRICEBAND4', 'PRICEBAND5', 'PRICEBAND6',
                     'PRICEBAND7', 'PRICEBAND8', 'PRICEBAND9', 'PRICEBAND10']
    type_name = 'PRICEBAND'
    value_name = 'BID'
    stacked_bids = hf.stack_columns(price_bids, cols_to_keep, cols_to_stack, type_name, value_name)
    # Map in capacity band types so that each price band can be mapped to a variable index.
    stacked_bids = add_capacity_band_type(stacked_bids, ns)
    # Map in bid indexes.
    price_bids_and_indexes = pd.merge(stacked_bids, indexes, 'inner', ['CAPACITYBAND', 'LINKID'])
    return price_bids_and_indexes.loc[:, ('INDEX', 'BID')]


def create_mnsp_region_requirement_coefficients(var_indexes, inter_data, region_requirements):
    # Make sure link flows are attributed to the correct regions.
    # Map links to interconnectors.
    link_and_inter_data = pd.merge(inter_data, var_indexes, 'inner', 'LINKID')
    # Refine data to only columns needed for from region calculations.
    from_region_data = link_and_inter_data.loc[:, ('LINKID', 'FROMREGION', 'FROM_REGION_TLF', 'INDEX')]
    # From region flows are attributed to regions as the negative invereses of their loss factors, this represents the
    # fact that link losses cause more power to be draw from a region than actually flow down the line.
    from_region_data['FROM_REGION_TLF'] = -1 / from_region_data['FROM_REGION_TLF']
    from_region_data.columns = ['LINKID', 'REGIONID', 'LHSCOEFFICIENTS', 'INDEX']
    # Refine data to only columns needed for from region calculations
    to_region_data = link_and_inter_data.loc[:, ('LINKID', 'TOREGION', 'TO_REGION_TLF', 'INDEX')]
    # To region flows are attributed to regions as their loss factors, this represents the fact that link losses cause
    # less power to be delivered to a region than actually flow down the line.
    to_region_data.columns = ['LINKID', 'REGIONID', 'LHSCOEFFICIENTS', 'INDEX']
    # Combine to from region coefficients.
    lhs_coefficients = pd.concat([from_region_data, to_region_data])
    # Select just the region requirements for energy.
    region_requirements_just_energy = region_requirements[region_requirements['BIDTYPE'] == 'ENERGY']
    # Map the requirement constraint rows to the link coefficients based on their region.
    lhs_coefficients_and_row_index = pd.merge(lhs_coefficients, region_requirements_just_energy, 'inner', 'REGIONID')
    lhs_coefficients_and_row_index = lhs_coefficients_and_row_index.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS')]
    return lhs_coefficients_and_row_index


def create_mnsp_link_indexes(mnsp_capacity_bids, max_var_index):
    # Create variable indexes for each link bid into the energy market. This is done by stacking, reindexing and saving
    # the off set index values as the row index colum.
    cols_to_keep = ['LINKID', 'MAXAVAIL', 'RAMPUPRATE']
    cols_to_stack = ['BANDAVAIL1', 'BANDAVAIL2', 'BANDAVAIL3', 'BANDAVAIL4', 'BANDAVAIL5', 'BANDAVAIL6', 'BANDAVAIL7',
                     'BANDAVAIL8', 'BANDAVAIL9', 'BANDAVAIL10']
    type_name = 'CAPACITYBAND'
    value_name = 'BIDVALUE'
    stacked_bids = hf.stack_columns(mnsp_capacity_bids, cols_to_keep, cols_to_stack, type_name, value_name)
    stacked_bids = hf.save_index(stacked_bids, 'INDEX', max_var_index + 1)
    return stacked_bids


def match_against_inter_data(mnsp_data, all_inter_data):
    # Select mnsp data only where they are also listed as an mnsp interconnector in the combined interconnector data.
    just_mnsp = all_inter_data[all_inter_data['ICTYPE'] == 'MNSP']['INTERCONNECTORID']
    mnsp_data = mnsp_data[mnsp_data['INTERCONNECTORID'].isin(just_mnsp)]
    return mnsp_data


def max_variable_index(newest_variable_data):
    # Find the maximum variable index already in use in the constraint matrix.
    max_index = newest_variable_data['INDEX'].max()
    return max_index


def create_generic_constraints(connection_point_constraints, inter_constraints, constraint_rhs,
                               bids_and_indexes, indexes_for_inter, gen_info, index_offset, active_region_cons,
                               mnsp_con_data):
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
    inter_constraints_mnsp = pd.merge(inter_constraints_mnsp, mnsp_con_data, 'inner', 'INTERCONNECTORID')
    # Map in additional constraint data.
    inter_constraints_mnsp = pd.merge(inter_constraints_mnsp, constraint_rhs, 'inner', ['GENCONID'])
    # Shorten name for formatting.
    aic = inter_constraints_mnsp
    # Give interconnector constraint coefficients the correct sign based on their direction of flow.
    inter_constraints_mnsp['FACTOR'] = aic['FACTOR'] * aic['LHSFACTOR']
    # Refine data to just that needed for final generic constraint preparation.
    cols_to_keep = ('INDEX', 'FACTOR', 'CONSTRAINTTYPE', 'RHS', 'GENCONID', 'GENERICCONSTRAINTWEIGHT')
    inter_constraints_mnsp = inter_constraints_mnsp.loc[:, cols_to_keep]

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
    combined_constraints = pd.concat([duid_constraints, inter_constraints,
                                      region_cons, inter_constraints_mnsp], sort=False)  # type: pd.DataFrame

    # Prepare a dataframe summarising key constrain information
    type_and_rhs = combined_constraints.loc[:, ('CONSTRAINTTYPE', 'RHS', 'GENCONID', 'GENERICCONSTRAINTWEIGHT')]
    # Make just unique constraints, giving each constraint a specific row index in the constraint matrix.
    type_and_rhs = type_and_rhs.drop_duplicates(subset='GENCONID', keep='first')
    type_and_rhs = hf.save_index(type_and_rhs.reset_index(drop=True), 'ROWINDEX', index_offset + 1)

    # Make the row index into the combined constraints dataframe.
    combined_constraints = pd.merge(combined_constraints, type_and_rhs, 'inner', 'GENCONID')
    # Keep explicit duid constraints over those implied via regional constraints.
    just_duid_combined_cons = combined_constraints[combined_constraints['INDEX'].isin(bids_and_indexes['INDEX'])]
    just_duid_combined_cons = just_duid_combined_cons.sort_values('DUID')
    just_duid_combined_cons = just_duid_combined_cons.groupby(['INDEX', 'ROWINDEX'], as_index=False).first()
    just_non_duid_combined_cons = combined_constraints[~combined_constraints['INDEX'].isin(bids_and_indexes['INDEX'])]
    combined_constraints = pd.concat([just_duid_combined_cons, just_non_duid_combined_cons], sort=False)
    # Select just data needed for the constraint matrix.
    combined_constraints = combined_constraints.loc[:, ('INDEX', 'FACTOR', 'ROWINDEX')]
    # Rename columns to standard constrain matrix names.
    combined_constraints.columns = ['INDEX', 'LHSCOEFFICIENTS', 'ROWINDEX']

    # Rename to standard names.
    type_and_rhs.columns = ['CONSTRAINTTYPE', 'RHSCONSTANT', 'GENCONID', 'CONSTRAINTWEIGHT', 'ROWINDEX']

    return combined_constraints, type_and_rhs


def create_region_req_contribution_to_constraint_matrix(latest_region_req, start_index, ns):
    # Create a set of row indexes for each regions requirement constraints.
    region_req_by_row = index_region_constraints(latest_region_req, start_index, ns)
    # Change the naming of columns to match those use for generator bidding constraints. Allows the constraints to
    # be processed together later on.
    region_req_by_row = change_req_naming_to_bid_naming(region_req_by_row, ns)
    return region_req_by_row


def create_objective_coefficients(just_latest_price_bids, bids_all_data, duid_info, ns):
    # Map generator loss factor data to price bid data.
    bids_all_data = pd.merge(bids_all_data, duid_info, 'inner', on=ns.col_unit_name)
    # Combine generator transmission loss factors and distribution loss factors
    bids_all_data[ns.col_loss_factor] = bids_all_data[ns.col_loss_factor] * bids_all_data['DISTRIBUTIONLOSSFACTOR']
    # Reformat price bids so they can be merged with bid variable index data.
    objective_coefficients = create_objective_coefficient_by_duid_type_number(just_latest_price_bids, ns)
    # Add the capacity band type so price bids can be matched to bid variable indexes.
    objective_coefficients = add_capacity_band_type(objective_coefficients, ns)
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
    objective_coefficients[ns.col_bid_value] = np.where(objective_coefficients[ns.col_dispatch_type] == ns.type_energy,
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


def create_inter_seg_dispatch_order_constraints(inter_var_indexes, row_offset, var_offset):
    # Interconnector segments need to be dispatched in the correct order. Without constraints the segments with the
    # lowest losses will be dispatched first, but these will not always be the segments corrseponding to the lowest
    # levels of flow fo the interconnector.

    # These constraints will only apply to inter variables transferring energy between regions.
    energy_only_indexes = inter_var_indexes[inter_var_indexes['BIDTYPE'] == 'ENERGY']
    # A different method for constructing constraints is used for the negative flow variables and the positive flow
    # variables so these a split and processed separately.
    pos_flow_vars = energy_only_indexes[energy_only_indexes['MWBREAKPOINT'] >= 0]
    neg_flow_vars = energy_only_indexes[energy_only_indexes['MWBREAKPOINT'] < 0]
    pos_flow_constraints, max_row, max_var = create_pos_flow_cons(pos_flow_vars, row_offset, var_offset)
    neg_flow_constraints, max_row, max_var = create_neg_flow_cons(neg_flow_vars, max_row, max_var)
    one_directional_flow_constraints = create_one_directional_flow_constraints(pos_flow_vars, neg_flow_vars, max_row,
                                                                               max_var)
    flow_cons = pd.concat([pos_flow_constraints, neg_flow_constraints, one_directional_flow_constraints], sort=True
                          ).reset_index(
        drop=True)
    return flow_cons


def create_one_directional_flow_constraints(pos_flow_vars, neg_flow_vars, max_row, max_var):
    pos_flow_vars = pos_flow_vars[pos_flow_vars['UPPERBOUND'] > 0.0001]
    pos_flow_vars = pos_flow_vars.sort_values('MWBREAKPOINT')
    first_pos_flow_vars = pos_flow_vars.groupby('INTERCONNECTORID', as_index=False).first()
    neg_flow_vars = neg_flow_vars[neg_flow_vars['UPPERBOUND'] > 0.0001]
    neg_flow_vars = neg_flow_vars.sort_values('MWBREAKPOINT')
    first_neg_flow_vars = neg_flow_vars.groupby('INTERCONNECTORID', as_index=False).last()
    decision_variables = hf.save_index(first_neg_flow_vars.loc[:, ['INTERCONNECTORID']], 'INDEX', max_var + 1)
    constraint_rows_neg = hf.save_index(first_neg_flow_vars.loc[:, ['INTERCONNECTORID']], 'ROWINDEX', max_row + 1)
    max_row = constraint_rows_neg['ROWINDEX'].max()
    constraint_rows_pos = hf.save_index(first_pos_flow_vars.loc[:, ['INTERCONNECTORID']], 'ROWINDEX', max_row + 1)
    pos_flow_vars_coefficients = pd.merge(constraint_rows_pos, first_pos_flow_vars.loc[:,
                                                               ['INDEX', 'INTERCONNECTORID', 'UPPERBOUND',
                                                                'MWBREAKPOINT']], 'inner', 'INTERCONNECTORID')
    pos_flow_vars_coefficients['LHSCOEFFICIENTS'] = 1
    pos_flow_vars_coefficients['RHSCONSTANT'] = 0
    neg_flow_vars_coefficients = pd.merge(constraint_rows_neg, first_neg_flow_vars.loc[:,
                                                               ['INDEX', 'INTERCONNECTORID', 'UPPERBOUND',
                                                                'MWBREAKPOINT']], 'inner', 'INTERCONNECTORID')
    neg_flow_vars_coefficients['LHSCOEFFICIENTS'] = 1
    neg_flow_vars_coefficients['RHSCONSTANT'] = neg_flow_vars_coefficients['UPPERBOUND']
    vars_coefficients = pd.concat([pos_flow_vars_coefficients, neg_flow_vars_coefficients])
    decision_coefficients = vars_coefficients.loc[:, ['INTERCONNECTORID', 'ROWINDEX', 'UPPERBOUND', 'MWBREAKPOINT']]
    decision_coefficients = pd.merge(decision_coefficients, decision_variables, 'inner', 'INTERCONNECTORID')
    decision_coefficients['LHSCOEFFICIENTS'] = np.where(decision_coefficients['MWBREAKPOINT'] > 0,
                                                        -1 * decision_coefficients['UPPERBOUND'],
                                                        decision_coefficients['UPPERBOUND'])
    decision_coefficients['RHSCONSTANT'] = np.where(decision_coefficients['MWBREAKPOINT'] > 0, 0,
                                                    decision_coefficients['UPPERBOUND'])
    decision_coefficients['CAPACITYBAND'] = 'INTERTRIGGERVAR'
    all_coefficients = pd.concat([decision_coefficients, vars_coefficients], sort=True)
    return all_coefficients


def create_pos_flow_cons(pos_flow_vars, row_offset, var_offset):
    # Two constraints link the flow of adjacent segments such that a lower segment must reach full capacity before an
    # an upper segment can begin to flow. Therefore the lowest segment in the pos flow vars has just one constraint
    # applied to it, called max a trigger constraints. This constraint is the paired with a min trigger constraint
    # applied to the adjacent segment. When the lower segment is at full capacity this triggers a decision variable
    # to then relax a constraint on the upper segment. Hence two sets of constraints are constructed below, the max
    # trigger constraints apply to segments 0 to n-1, and the min trigger constraints apply to segments 1 to n.
    pos_flow_vars = pos_flow_vars.sort_values('LOSSSEGMENT')
    pos_flow_vars_not_first = pos_flow_vars.groupby('INTERCONNECTORID').apply(lambda group: group.iloc[1:, 1:])
    pos_flow_vars_not_last = pos_flow_vars.groupby('INTERCONNECTORID').apply(lambda group: group.iloc[:-1, 1:])
    max_trigger_cons, max_row, _max_var = create_pos_max_trigger_cons(pos_flow_vars_not_last, row_offset, var_offset)
    min_trigger_cons, max_row, max_var = create_pos_min_trigger_cons(pos_flow_vars_not_first, max_row, var_offset)
    return pd.concat([min_trigger_cons, max_trigger_cons]), max_row, max_var


def create_pos_min_trigger_cons(pos_flow_vars_not_first, row_offset, var_offset):
    # Create the trigger variables, in the linear problem these will be integer variables with either a value of zero
    # or one. In these constraints when the value is zero, the lower segment can be any flow value, but when the trigger
    # variable has a value of 1, then the lower segment must be at its upper bound. Putting this another way the trigger
    # value can only equal 1 when the lower segment is at full capacity. Then the trigger variable is re used in the
    # max trigger constraints to trigger the upper segment to flow once the lower is at full capacity.

    # Create the trigger variable coefficients with indexes.
    integer_var_coefficients = pos_flow_vars_not_first.copy()
    integer_var_coefficients = integer_var_coefficients.drop('INDEX', axis=1)
    integer_var_coefficients = integer_var_coefficients.reset_index(drop=True)
    integer_var_coefficients = hf.save_index(integer_var_coefficients, 'INDEX', var_offset + 1)
    integer_var_coefficients = hf.save_index(integer_var_coefficients, 'ROWINDEX', row_offset + 1)
    # The coefficient is set as the segments maximum capacity multiplied by minus 1. See constraint form below for
    # reasoning.
    integer_var_coefficients['LHSCOEFFICIENTS'] = -1 * integer_var_coefficients['UPPERBOUND']
    # The RHS is set to zero. See constraint form below for reasoning.
    integer_var_coefficients['RHSCONSTANT'] = 0
    # This flags the lp setup function to make this variable type integer.
    integer_var_coefficients['CAPACITYBAND'] = 'INTERTRIGGERVAR'
    # Select just the values we need.
    integer_var_coefficients = integer_var_coefficients.loc[:,
                               ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'RHSCONSTANT', 'CAPACITYBAND')]

    # Create the interconnector variable coefficients with indexes
    seg_var_coefficients = pos_flow_vars_not_first.copy()
    seg_var_coefficients = seg_var_coefficients.reset_index(drop=True)
    seg_var_coefficients = hf.save_index(seg_var_coefficients, 'ROWINDEX', row_offset + 1)
    # Set the coefficient values to 1. See constraint form below for reasoning.
    seg_var_coefficients['LHSCOEFFICIENTS'] = 1
    # The RHS value is 0, the same as above. See constraint form below for reasoning.
    seg_var_coefficients['RHSCONSTANT'] = 0
    # Just a label for the variable.
    seg_var_coefficients['CAPACITYBAND'] = 'INTERVAR'
    seg_var_coefficients = seg_var_coefficients.loc[:,
                           ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'RHSCONSTANT', 'CAPACITYBAND')]

    # The form of the constraints is then: 1 * Lower segment - Upper bound * trigger variable >= 0. Note when the
    # trigger variable is equal to 1, then the lower segement must equal the upper bound.

    # Combine all coefficients.
    min_trigger_cons = pd.concat([integer_var_coefficients, seg_var_coefficients])

    # Find max indexes used.
    max_row = seg_var_coefficients['ROWINDEX'].max()
    max_var = integer_var_coefficients['INDEX'].max()
    return min_trigger_cons, max_row, max_var


def create_pos_max_trigger_cons(pos_flow_vars_not_last, row_offset, var_offset):
    # Create the trigger variables, in the linear problem these will be integer variables with either a value of zero
    # or one. In these constraints when the value is zero, the upper segment must have a zero value, but when the
    # trigger variable has a value of 1, then the upper segment can have positive values. Then the trigger variable is
    # re used in the min trigger constraints such that it can only have a value of 1 when the lower segment is at full
    # capacity. Tis force the segments to be dispatched in the correct order.

    # Create the trigger variable coefficients with indexes.
    integer_var_coefficients = pos_flow_vars_not_last.copy()
    integer_var_coefficients = integer_var_coefficients.drop('INDEX', axis=1)
    integer_var_coefficients = integer_var_coefficients.reset_index(drop=True)
    integer_var_coefficients = hf.save_index(integer_var_coefficients, 'INDEX', var_offset + 1)
    integer_var_coefficients = hf.save_index(integer_var_coefficients, 'ROWINDEX', row_offset + 1)
    # The coefficient is set as the segments maximum capacity. See constraint form below for reasoning.
    integer_var_coefficients['LHSCOEFFICIENTS'] = integer_var_coefficients['UPPERBOUND']
    # The RHS is set to zero. See constraint form below for reasoning.
    integer_var_coefficients['RHSCONSTANT'] = 0
    # This flags the lp setup function to make this variable type integer.
    integer_var_coefficients['CAPACITYBAND'] = 'INTERTRIGGERVAR'
    # Select just the information needed.
    integer_var_coefficients = integer_var_coefficients.loc[:,
                               ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'RHSCONSTANT', 'CAPACITYBAND')]

    # Create the interconnector variable coefficients with indexes.
    seg_var_coefficients = pos_flow_vars_not_last.copy()
    seg_var_coefficients = seg_var_coefficients.reset_index(drop=True)
    seg_var_coefficients = hf.save_index(seg_var_coefficients, 'ROWINDEX', row_offset + 1)
    # Set the coefficient values to minus 1. See constraint form below for reasoning.
    seg_var_coefficients['LHSCOEFFICIENTS'] = - 1
    # The RHS value is 0, the same as above. See constraint form below for reasoning.
    seg_var_coefficients['RHSCONSTANT'] = 0
    # Just a label for the variable.
    seg_var_coefficients['CAPACITYBAND'] = 'INTERVAR'
    # Select just the columns we need.
    seg_var_coefficients = seg_var_coefficients.loc[:,
                           ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'RHSCONSTANT', 'CAPACITYBAND')]
    # Combine all coefficients.
    max_trigger_cons = pd.concat([integer_var_coefficients, seg_var_coefficients])
    # Find max indexes used.
    max_row = seg_var_coefficients['ROWINDEX'].max()
    max_var = integer_var_coefficients['INDEX'].max()
    return max_trigger_cons, max_row, max_var


def create_neg_flow_cons(neg_flow_vars, row_offset, var_offset):
    # Two constraints link the flow of adjacent segments such that a lower segment must reach full capacity before an
    # an upper segment can begin to flow. Therefore the lowest segment in the neg flow vars has just one constraint
    # applied to it, called max a trigger constraints. This constraint is the paired with a min trigger constraint
    # applied to the adjacent segment. When the lower segment is at full capacity this triggers a decision variable
    # to then relax a constraint on the upper segment. Hence two sets of constraints are constructed below, the max
    # trigger constraints apply to segments 0 to n-1, and the min trigger constraints apply to segments 1 to n.
    pos_flow_vars = neg_flow_vars.sort_values('LOSSSEGMENT', ascending=False)
    pos_flow_vars_not_first = pos_flow_vars.groupby('INTERCONNECTORID').apply(lambda group: group.iloc[1:, 1:])
    pos_flow_vars_not_last = pos_flow_vars.groupby('INTERCONNECTORID').apply(lambda group: group.iloc[:-1, 1:])
    max_trigger_cons, max_row, _max_var = create_pos_max_trigger_cons(pos_flow_vars_not_last, row_offset, var_offset)
    min_trigger_cons, max_row, max_var = create_pos_min_trigger_cons(pos_flow_vars_not_first, max_row, var_offset)
    return pd.concat([min_trigger_cons, max_trigger_cons]), max_row, max_var


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


def index_inter(inter_info: pd, inter_seg_definitions: pd, max_index: int, ns: object) -> pd:
    # Create a direction for each interconnector.
    cols_to_keep = [ns.col_inter_id]
    cols_to_stack = [ns.col_region_from, ns.col_region_to]
    type_name = ns.col_direction
    value_name = ns.col_region_id
    stacked_inter_directions = hf.stack_columns(inter_info, cols_to_keep, cols_to_stack, type_name, value_name)
    stacked_inter_directions['DUMMYJOIN'] = 1

    # Create a separate interconnector type for each FCAS type.
    fcas_services = pd.DataFrame()
    fcas_services[ns.col_bid_type] = ns.list_fcas_types
    fcas_services['LOSSSEGMENT'] = 1
    fcas_services['DUMMYJOIN'] = 1

    # Create an interconnector type for energy.
    energy_services = pd.DataFrame()
    energy_services[ns.col_bid_type] = [ns.type_energy]
    energy_services['DUMMYJOIN'] = 1

    # Select just the interconnector segment data we need.
    inter_segments = inter_seg_definitions.loc[:, (ns.col_inter_id, 'LOSSSEGMENT', 'MWBREAKPOINT')]
    # Beak segments for positive flow and negative flow into different groups.
    pos_inter_segments = inter_segments[inter_segments['MWBREAKPOINT'] >= 0]
    neg_inter_segments = inter_segments[inter_segments['MWBREAKPOINT'] < 0]

    # Merge interconnector directions with energy type to create energy specific interconnectors.
    inter_multiplied_by_energy_types = pd.merge(stacked_inter_directions, energy_services, 'inner', ['DUMMYJOIN'])
    # Merge interconnector from directions with positive flow segments to create the segments needed for the from
    # direction.
    pos_inter_multiplied_by_energy_types = pd.merge(
        inter_multiplied_by_energy_types[inter_multiplied_by_energy_types[ns.col_direction] == ns.col_region_from],
        pos_inter_segments, 'inner', [ns.col_inter_id])
    # Merge interconnector to directions with negative flow segments to create the segments needed for the to
    # direction.
    neg_inter_multiplied_by_energy_types = pd.merge(
        inter_multiplied_by_energy_types[inter_multiplied_by_energy_types[ns.col_direction] == ns.col_region_to],
        neg_inter_segments, 'inner', [ns.col_inter_id])

    # Combine pos and negative segments into one dataframe.
    inter_multiplied_by_energy_types = pd.concat([pos_inter_multiplied_by_energy_types,
                                                  neg_inter_multiplied_by_energy_types])

    # Sort values so indexing occurs in a logical sequence.
    inter_multiplied_by_types = inter_multiplied_by_energy_types.sort_values([ns.col_inter_id, ns.col_direction,
                                                                              ns.col_bid_type, 'LOSSSEGMENT'])
    # Create interconnector variable indexes.
    inter_multiplied_by_types = inter_multiplied_by_types.reset_index(drop=True)
    inter_multiplied_by_types = hf.save_index(inter_multiplied_by_types, ns.col_variable_index, max_index + 1)
    # Delete dummy column used for joining data.
    inter_multiplied_by_types = inter_multiplied_by_types.drop('DUMMYJOIN', 1)
    return inter_multiplied_by_types


def index_region_constraints(raw_constraints, max_constraint_row_index, ns):
    # Create constraint rows for regional based constraints.
    row_for_each_constraint = hf.stack_columns(raw_constraints, [ns.col_region_id], ['TOTALDEMAND'],
                                            ns.col_region_constraint_type, ns.col_region_constraint_value)
    row_for_each_constraint_indexed = hf.save_index(row_for_each_constraint, ns.col_constraint_row_index,
                                                 max_constraint_row_index + 1)
    return row_for_each_constraint_indexed


def change_req_naming_to_bid_naming(df_with_req_naming, ns):
    # Change the naming conventions of regional requirements to match the naming conventions of bidding, this allows
    # bidding variables to contribute to the correct regional requirements.
    df_with_bid_naming = df_with_req_naming.copy()
    # Apply the mapping of naming defined in the given function.
    df_with_bid_naming[ns.col_bid_type] = \
        df_with_bid_naming[ns.col_region_constraint_type].apply(map_req_naming_to_bid_naming, args=(ns,))
    df_with_bid_naming = df_with_bid_naming.drop(ns.col_region_constraint_type, axis=1)
    return df_with_bid_naming


def map_req_naming_to_bid_naming(req_name, ns):
    # Define the mapping of requirement naming to bidding naming.
    if req_name == ns.col_gen_req:
        bid_name = ns.type_energy
    else:
        characters_to_remove = len(ns.req_suffix)
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


def create_req_row_indexes_for_inter(inter_variable_indexes, req_row_indexes, ns):
    # Give each interconnector variable the correct requirement row indexes so it contributes to the correct regional
    # constraints.

    # Redefine directions such that each interconnector is defined as coming from a particular region.
    inter_variable_indexes[ns.col_direction] = np.where(
        inter_variable_indexes[ns.col_direction] == ns.col_region_to,
        ns.col_region_from, inter_variable_indexes[ns.col_direction])

    # Split up interconnectors depending on whether or not they represent negative or postive flow as each type needs
    # to be processed differently.
    first_of_pairs = inter_variable_indexes[inter_variable_indexes['MWBREAKPOINT'] >= 0]
    first_of_pairs = first_of_pairs.drop([ns.col_region_id, ns.col_direction], 1)
    second_of_pairs = inter_variable_indexes[inter_variable_indexes['MWBREAKPOINT'] < 0]
    second_of_pairs = second_of_pairs.drop([ns.col_region_id, ns.col_direction], 1)

    # Create copies of the interconnectors that have the direction name reversed. The copy is need as each
    # interconnector must make a contribution both to the region it flows out of and the region it flows into.
    opposite_directions = inter_variable_indexes.copy()
    opposite_directions[ns.col_direction] = np.where(opposite_directions[ns.col_direction] == ns.col_region_from,
                                                     ns.col_region_to, ns.col_region_from)

    # Break into negative and positive flow types so each can be processed separately. Also we only need to take one
    # unique set of inter id, direction, region id and bid type as the original direction data contains the segment
    # info.
    first_of_opposite_pairs = opposite_directions[opposite_directions['MWBREAKPOINT'] >= 0]. \
        groupby([ns.col_inter_id, ns.col_bid_type], as_index=False).first()
    first_of_opposite_pairs = first_of_opposite_pairs.loc[:, (ns.col_inter_id, ns.col_direction, ns.col_region_id,
                                                              ns.col_bid_type)]
    second_of_opposite_pairs = opposite_directions[opposite_directions['MWBREAKPOINT'] < 0]. \
        groupby([ns.col_inter_id, ns.col_bid_type], as_index=False).first()
    second_of_opposite_pairs = second_of_opposite_pairs.loc[:, (ns.col_inter_id, ns.col_direction, ns.col_region_id,
                                                                ns.col_bid_type)]

    # Merge opposite pairs with orginal paris to complete segment info.
    first_of_opposite_pairs = pd.merge(first_of_opposite_pairs, second_of_pairs, 'inner',
                                       [ns.col_inter_id, ns.col_bid_type])
    second_of_opposite_pairs = pd.merge(second_of_opposite_pairs, first_of_pairs, 'inner',
                                        [ns.col_inter_id, ns.col_bid_type])

    # Combine positive and negative flow segments back together.
    opposite_directions = pd.concat([first_of_opposite_pairs, second_of_opposite_pairs])

    # Combine opposite flow variables with normal variables.
    both_directions = pd.concat([inter_variable_indexes, opposite_directions])

    # Merge in requirement row info.
    inter_col_index_row_index = pd.merge(both_directions, req_row_indexes, 'left',
                                         [ns.col_bid_type, ns.col_region_id])

    return inter_col_index_row_index


def calculate_loss_factors_for_inter_segments(req_row_indexes, demand_by_region, inter_demand_coefficients,
                                              inter_constants, ns):
    # Calculate the average loss factor for each interconnector segment. This is based on the aemo dynamic loss
    # factor equations that take into account regional demand and determine different loss factors for discrete
    # interconnector segments.
    # Convert demand data into a dictionary to speed up value selection in the vectorized loss factor calculations.
    demand_by_region = demand_by_region.loc[:, ('REGIONID', 'TOTALDEMAND')]
    demand_by_region = demand_by_region.set_index('REGIONID')
    demand_by_region = demand_by_region['TOTALDEMAND'].to_dict()
    # Convert interconnector loss constant and loss coefficient data into a dictionary to speed up value selection in
    # the vectorized loss factor calculations.
    inter_constants = inter_constants.loc[:, ('INTERCONNECTORID', 'LOSSCONSTANT', 'LOSSFLOWCOEFFICIENT')]
    inter_constants = inter_constants.set_index('INTERCONNECTORID')
    loss_constants = inter_constants['LOSSCONSTANT'].to_dict()
    flow_coefficient = inter_constants['LOSSFLOWCOEFFICIENT'].to_dict()
    inter_demand_coefficients = inter_demand_coefficients.loc[:, ('REGIONID', 'INTERCONNECTORID', 'DEMANDCOEFFICIENT')]
    # Convert demand coefficient data into a dictionary to speed up value selection in the vectorized loss factor
    # calculations.
    inter_demand_coefficients = inter_demand_coefficients.set_index(['INTERCONNECTORID', 'REGIONID'])
    idc = inter_demand_coefficients
    inter_demand_coefficients = {level: idc.xs(level).to_dict('index') for level in idc.index.levels[0]}
    # Calculate the average loss factor for each interconnector segment.
    req_row_indexes[ns.col_lhs_coefficients] = \
        np.vectorize(calc_req_row_coefficients_for_inter,
                     excluded=['demand_by_region', 'ns', 'inter_demand_coefficients', 'loss_constants',
                               'flow_coefficient'])(
            req_row_indexes['MEANVALUE'], req_row_indexes[ns.col_inter_id], req_row_indexes[ns.col_direction],
            req_row_indexes[ns.col_bid_type], demand_by_region=demand_by_region, ns=ns,
            inter_demand_coefficients=inter_demand_coefficients, loss_constants=loss_constants, flow_coefficient=
            flow_coefficient)

    return req_row_indexes


def calc_req_row_coefficients_for_inter(flow, inter_id, direction, bid_type, demand_by_region, ns,
                                        inter_demand_coefficients, loss_constants, flow_coefficient):
    # This function implements the dynamic loss factor calculation.
    # Add the constant and the flow components to the loss factor.
    average_loss_factor = loss_constants[inter_id] + flow * flow_coefficient[inter_id]
    # Select the demand coefficients for the current interconnector.
    coefficients_subset = inter_demand_coefficients[inter_id]
    # Add the regional demand component for each region with a demand coefficient for this interconnector.
    for demand_region, demand_coefficient in coefficients_subset.items():
        demand = demand_by_region[demand_region]
        average_loss_factor += demand_coefficient['DEMANDCOEFFICIENT'] * demand

    # Translate the average loss factors to loss percentages.
    # TODO: Check this method.
    if direction == ns.col_region_from:
        average_loss_percent = average_loss_factor - 1
    if direction == ns.col_region_to:
        average_loss_percent = 1 - average_loss_factor

    return average_loss_percent


def convert_contribution_coefficients(req_row_indexes: pd, loss_proportions, ns) -> pd:
    # The loss from an interconnector are attributed to the two interconnected regions based on the input loss
    # proportions.
    # Select just the data needed.
    loss_proportions = loss_proportions.loc[:, ('INTERCONNECTORID', 'FROMREGIONLOSSSHARE')]
    # Map the loss proportions to the the loss percentages based on the interconnector.
    req_row_indexes = pd.merge(req_row_indexes, loss_proportions, 'inner', [ns.col_inter_id])
    # Modify the the loss percentages depending on whether it is a to or from loss percentage.
    req_row_indexes[ns.col_lhs_coefficients] = np.where(
        req_row_indexes[ns.col_direction] == ns.col_region_from,
        req_row_indexes[ns.col_lhs_coefficients] * req_row_indexes['FROMREGIONLOSSSHARE'],
        req_row_indexes[ns.col_lhs_coefficients] * (1 - req_row_indexes['FROMREGIONLOSSSHARE']))

    # Change the loss percentages to contribution coefficients i.e how the interconnectors contribute to meeting
    # regional demand requiremnets after accounting for loss and loss proportions.
    req_row_indexes[ns.col_lhs_coefficients] = np.where(req_row_indexes[ns.col_direction] == ns.col_region_from,
                                                        -1 * (req_row_indexes[ns.col_lhs_coefficients] + 1),
                                                        1 - (req_row_indexes[ns.col_lhs_coefficients]))
    return req_row_indexes


def add_capacity_band_type(df_with_price_bands, ns):
    # Map the names of the capacity bands to a dataframe that already has the names of the price bands.
    band_map = pd.DataFrame()
    band_map[ns.col_price_band_number] = ns.cols_bid_price_name_list
    band_map[ns.col_capacity_band_number] = ns.cols_bid_cap_name_list
    df_with_capacity_and_price_bands = pd.merge(df_with_price_bands, band_map, 'left', [ns.col_price_band_number])
    return df_with_capacity_and_price_bands


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


def create_inequality_types(bids: pd, region_req: pd, generic_type, inter_seg_dispatch_order_constraints,
                            joint_capacity_constraints, ns):
    # For each constraint row determine the inequality type and save in a list of ascending order according to row
    # index.

    # Process bidding constraints.
    bids = bids.drop_duplicates(subset=[ns.col_constraint_row_index]).copy()
    bids[ns.col_enquality_type] = 'equal_or_less'
    bids[ns.col_enquality_type] = np.where(bids[ns.col_enablement_type] == 'MINENERGY',
                                           'equal_or_greater', bids[ns.col_enquality_type])
    bids = bids.loc[:, (ns.col_constraint_row_index, ns.col_enquality_type, ns.col_rhs_constant)].copy()
    # Process region requirement constraints.
    region_req[ns.col_enquality_type] = 'equal'
    region_req = region_req.loc[:, (ns.col_constraint_row_index, ns.col_enquality_type, ns.col_rhs_constant)].copy()
    # Process generic constraints.
    generic_type[ns.col_enquality_type] = np.where(generic_type['CONSTRAINTTYPE'] == '>=', 'equal_or_greater', '')
    generic_type[ns.col_enquality_type] = np.where(generic_type['CONSTRAINTTYPE'] == '<=', 'equal_or_less',
                                                   generic_type[ns.col_enquality_type])
    generic_type[ns.col_enquality_type] = np.where(generic_type['CONSTRAINTTYPE'] == '=', 'equal',
                                                   generic_type[ns.col_enquality_type])

    # Process interconnector segments constraints.
    inter_seg_dispatch_order_constraints = inter_seg_dispatch_order_constraints.drop_duplicates(
        subset=[ns.col_constraint_row_index]).copy()
    inter_seg_dispatch_order_constraints[ns.col_enquality_type] = 'equal_or_less'
    inter_seg_dispatch_order_constraints = inter_seg_dispatch_order_constraints.loc[:,
                                           (ns.col_constraint_row_index, ns.col_enquality_type, ns.col_rhs_constant)].copy()
    # Process joint capacity constraints.
    joint_capacity_constraints = joint_capacity_constraints.drop_duplicates(
        subset=[ns.col_constraint_row_index]).copy()
    joint_capacity_constraints[ns.col_enquality_type] = \
        np.where(joint_capacity_constraints['CONSTRAINTTYPE'] == '>=', 'equal_or_greater', '')
    joint_capacity_constraints[ns.col_enquality_type] = \
        np.where(joint_capacity_constraints['CONSTRAINTTYPE'] == '<=', 'equal_or_less',
                 joint_capacity_constraints[ns.col_enquality_type])
    joint_capacity_constraints[ns.col_enquality_type] = \
        np.where(joint_capacity_constraints['CONSTRAINTTYPE'] == '=', 'equal',
                 joint_capacity_constraints[ns.col_enquality_type])

    # Combine type data, sort by row index and convert to type list.
    combined_type_data = pd.concat([bids, region_req, generic_type,
                                    # inter_seg_dispatch_order_constraints,
                                    joint_capacity_constraints], sort=False)
    #combined_type_data = combined_type_data.sort_values([ns.col_constraint_row_index], ascending=True)
    #combined_type_data = list(combined_type_data[ns.col_enquality_type])
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


def add_inter_bounds(inter_variable_indexes, inter_seg_definitions, ns):
    # Take the actual break point of each interconnector segment and store in series form.
    actual_break_points = inter_seg_definitions['MWBREAKPOINT']

    # For each segment find the break point of the segment number one higher and store in series form.
    inter_segs_higher = inter_seg_definitions.loc[:, ('INTERCONNECTORID', 'LOSSSEGMENT')]
    inter_segs_higher['LOSSSEGMENT'] = inter_segs_higher['LOSSSEGMENT'] + 1
    inter_segs_higher = pd.merge(inter_segs_higher, inter_seg_definitions, 'left', ['INTERCONNECTORID', 'LOSSSEGMENT'])
    high_break_points = inter_segs_higher['MWBREAKPOINT']

    # For each segment find the break point of the segment number one lower and store in series form.
    inter_segs_lower = inter_seg_definitions.loc[:, ('INTERCONNECTORID', 'LOSSSEGMENT')]
    inter_segs_lower['LOSSSEGMENT'] = inter_segs_lower['LOSSSEGMENT'] - 1
    inter_segs_lower = pd.merge(inter_segs_lower, inter_seg_definitions, 'left', ['INTERCONNECTORID', 'LOSSSEGMENT'])
    lower_break_points = inter_segs_lower['MWBREAKPOINT']

    # Find the absolute power flow limit of each segment and its mean absolute value.
    inter_seg_definitions[ns.col_upper_bound], inter_seg_definitions['MEANVALUE'] \
        = np.vectorize(calc_bound_for_segment)(actual_break_points, high_break_points, lower_break_points)

    # Map the results back to the interconnector variable indexes.
    seg_results = inter_seg_definitions.loc[:, ('INTERCONNECTORID', 'LOSSSEGMENT', 'MEANVALUE', ns.col_upper_bound)]
    inter_variable_indexes = pd.merge(inter_variable_indexes, seg_results, 'inner', ['INTERCONNECTORID', 'LOSSSEGMENT'])

    return inter_variable_indexes


def calc_bound_for_segment(actual_break_point, higher_break_point, lower_break_point):
    # If the segment is positive and not adjacent to negative segments it applies to the flow between its break point
    # and the break point of the adjacent lower segment.
    if actual_break_point > 0.1 and lower_break_point > 0.1:
        limit = abs(actual_break_point - lower_break_point)
        mean_value = (actual_break_point + lower_break_point) / 2
    # If the segment is positive but adjacent to a negative segment or a zero segment then it applies to the flow
    # between its break point and 0 MW.
    elif actual_break_point > 0.1 and lower_break_point <= 0.1:
        limit = actual_break_point
        mean_value = actual_break_point / 2
    # If the segment is the zero segment (may not exist) then it does not apply to any flow.
    elif 0.1 > actual_break_point > -0.1:
        limit = 0
        mean_value = 0
    # If the segment is negative but adjacent to a positive segment or the zero segment then it applies to the flow
    # between its break point and 0 MW.
    elif actual_break_point < - 0.1 and higher_break_point >= - 0.1:
        limit = abs(actual_break_point)
        mean_value = actual_break_point / 2
    # If the segment is negative and not adjacent to positive segments it applies to the flow between its break point
    # and the break point of the adjacent higher segment.
    elif actual_break_point < 0.1 and higher_break_point < 0.1:
        limit = abs(higher_break_point - actual_break_point)
        mean_value = (actual_break_point + higher_break_point) / 2
    # If a segment does not meet any of the above conditions raise an error.
    else:
        print(actual_break_point)
        raise ValueError('A segment could not be bounded')

    return limit, mean_value


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
    md['RELATIVEPRICE'] = np.where(md['BIDTYPE'] == 'ENERGY',
                                   (md['PRICE'] / (md['TRANSMISSIONLOSSFACTOR'] * md['DISTRIBUTIONLOSSFACTOR'])),
                                   md['PRICE'])
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


def create_joint_capacity_constraints(bids_and_indexes, capacity_bids, initial_conditions, max_con):
    t0 = perf_counter()
    # Pre calculate at table that allows for the efficient selection of generators according to which markets they are
    # bidding into
    ta = perf_counter()
    bid_type_check = bids_and_indexes.copy()
    bid_type_check = bid_type_check.loc[:, ('DUID', 'BIDTYPE')]
    bid_type_check = bid_type_check.drop_duplicates(['DUID', 'BIDTYPE'])
    bid_type_check['PRESENT'] = 1
    bid_type_check = bid_type_check.pivot('DUID', 'BIDTYPE', 'PRESENT')
    bid_type_check = bid_type_check.fillna(0)
    bid_type_check['DUID'] = bid_type_check.index
    print('Setup time in create_joint_capacity_constraints {}'.format(perf_counter() - ta))
    combined_joint_capacity_constraints = []
    ta = perf_counter()
    for fcas_service in ['RAISE6SEC', 'RAISE60SEC', 'RAISE5MIN', 'LOWER6SEC', 'LOWER60SEC', 'LOWER5MIN',
                         'LOWERREG', 'RAISEREG']:
        if fcas_service in ['RAISE6SEC', 'RAISE60SEC', 'RAISE5MIN']:
            joint_constraints = create_joint_capacity_constraints_raise(bids_and_indexes.copy(), capacity_bids.copy(),
                                                                        max_con, fcas_service, bid_type_check)
        if fcas_service in ['LOWER6SEC', 'LOWER60SEC', 'LOWER5MIN']:
            joint_constraints = create_joint_capacity_constraints_lower(bids_and_indexes.copy(), capacity_bids.copy(),
                                                                        max_con, fcas_service, bid_type_check)
        if fcas_service in ['LOWERREG', 'RAISEREG']:
            joint_constraints1 = create_joint_ramping_constraints(bids_and_indexes.copy(), initial_conditions.copy(),
                                                                  max_con, fcas_service, bid_type_check)
            max_con = hf.max_constraint_index(joint_constraints1[0])
            joint_constraints2 = joint_energy_and_reg_constraints(bids_and_indexes.copy(), capacity_bids.copy(),
                                                                  max_con, fcas_service, bid_type_check)
            joint_constraints = joint_constraints1 + joint_constraints2

        max_con = hf.max_constraint_index(joint_constraints[-1])
        combined_joint_capacity_constraints += joint_constraints
    print('Loop time in create_joint_capacity_constraints {}'.format(perf_counter() - ta))
    ta = perf_counter()
    combined_joint_capacity_constraints = pd.concat(combined_joint_capacity_constraints)
    print('Concat time in create_joint_capacity_constraints {}'.format(perf_counter() - ta))
    print('Total time in create_joint_capacity_constraints {}'.format(perf_counter()-t0))
    return combined_joint_capacity_constraints


def create_joint_capacity_constraints_raise(bids_and_indexes, capacity_bids, max_con, raise_contingency_service,
                                            bid_type_check):
    ta = perf_counter()
    t0 = perf_counter()
    units_with_reg_or_energy = bid_type_check[(bid_type_check['RAISEREG'] == 1) | (bid_type_check['ENERGY'] == 1)]
    units_with_raise_contingency = bids_and_indexes[(bids_and_indexes['BIDTYPE'] == raise_contingency_service)]
    print('0 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units = set(units_with_reg_or_energy['DUID']).intersection(units_with_raise_contingency['DUID'])
    units_to_constraint_raise = bids_and_indexes[bids_and_indexes['DUID'].isin(units)]
    print('1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    upper_slope_coefficients = capacity_bids.copy()
    upper_slope_coefficients = \
        upper_slope_coefficients[upper_slope_coefficients['BIDTYPE'] == raise_contingency_service]
    upper_slope_coefficients['UPPERSLOPE'] = ((upper_slope_coefficients['ENABLEMENTMAX'] -
                                               upper_slope_coefficients['HIGHBREAKPOINT']) /
                                              upper_slope_coefficients['MAXAVAIL'])
    upper_slope_coefficients = upper_slope_coefficients.loc[:, ('DUID', 'UPPERSLOPE', 'ENABLEMENTMAX')]
    print('2 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_raise = pd.merge(units_to_constraint_raise, upper_slope_coefficients, 'left', 'DUID')
    print('3 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_raise['LHSCOEFFICIENTS'] = np.where(units_to_constraint_raise['BIDTYPE'] == 'ENERGY', 1, 0)
    units_to_constraint_raise['LHSCOEFFICIENTS'] = np.where((units_to_constraint_raise['BIDTYPE'] == 'RAISEREG') &
                                                            (units_to_constraint_raise[
                                                                 'CAPACITYBAND'] != 'FCASINTEGER'),
                                                            1, units_to_constraint_raise['LHSCOEFFICIENTS'])
    print('4 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_raise['LHSCOEFFICIENTS'] = \
        np.where((units_to_constraint_raise['BIDTYPE'] == raise_contingency_service) &
                 (units_to_constraint_raise['CAPACITYBAND'] != 'FCASINTEGER'), units_to_constraint_raise['UPPERSLOPE'],
                 units_to_constraint_raise['LHSCOEFFICIENTS'])
    units_to_constraint_raise['RHSCONSTANT'] = units_to_constraint_raise['ENABLEMENTMAX']
    units_to_constraint_raise['CONSTRAINTTYPE'] = '<='
    #units_to_constraint_raise_rows = units_to_constraint_raise.groupby('DUID', as_index=False).first()
    #units_to_constraint_raise_rows = save_index(units_to_constraint_raise_rows, 'ROWINDEX', max_con + 1)
    #units_to_constraint_raise_rows = units_to_constraint_raise_rows.loc[:, ('DUID', 'ROWINDEX')]
    #units_to_constraint_raise = pd.merge(units_to_constraint_raise, units_to_constraint_raise_rows, 'left', 'DUID')
    print('4.1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    #unique_duids = units_to_constraint_raise['DUID'].unique()
    print('4.2 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    constraint_rows = dict(zip(units, np.arange(max_con + 1, max_con + 1 + len(units))))
    print('4.3 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_raise['ROWINDEX'] = units_to_constraint_raise['DUID'].map(constraint_rows)
    print('4.4 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_raise = \
        units_to_constraint_raise.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT')]
    print('5 {}'.format(perf_counter() - t0))
    print('create_joint_capacity_constraints_raise {}'.format(perf_counter() - ta))
    return [units_to_constraint_raise]


def create_joint_capacity_constraints_lower(bids_and_indexes, capacity_bids, max_con, raise_contingency_service,
                                            bid_type_check):
    ta = perf_counter()
    t0 = perf_counter()
    units_with_reg_or_energy = bid_type_check[(bid_type_check['LOWERREG'] == 1) | (bid_type_check['ENERGY'] == 1)]
    units_with_raise_contingency = bids_and_indexes[(bids_and_indexes['BIDTYPE'] == raise_contingency_service)]
    units_to_constraint_raise = bids_and_indexes[
        (bids_and_indexes['DUID'].isin(list(units_with_reg_or_energy['DUID']))) &
        (bids_and_indexes['DUID'].isin(list(units_with_raise_contingency['DUID']))) &
        ((bids_and_indexes['BIDTYPE'] == 'LOWERREG') |
         (bids_and_indexes['BIDTYPE'] == 'ENERGY') |
         (bids_and_indexes['BIDTYPE'] == raise_contingency_service))]
    print('1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    upper_slope_coefficients = capacity_bids.copy()
    upper_slope_coefficients = \
        upper_slope_coefficients[upper_slope_coefficients['BIDTYPE'] == raise_contingency_service]
    upper_slope_coefficients['LOWERSLOPE'] = ((upper_slope_coefficients['LOWBREAKPOINT'] -
                                               upper_slope_coefficients['ENABLEMENTMIN']) /
                                              upper_slope_coefficients['MAXAVAIL'])
    upper_slope_coefficients = upper_slope_coefficients.loc[:, ('DUID', 'LOWERSLOPE', 'ENABLEMENTMIN')]

    units_to_constraint_raise = pd.merge(units_to_constraint_raise, upper_slope_coefficients, 'left', 'DUID')
    units_to_constraint_raise['LHSCOEFFICIENTS'] = np.where(units_to_constraint_raise['BIDTYPE'] == 'ENERGY', 1, 0)
    units_to_constraint_raise['LHSCOEFFICIENTS'] = np.where((units_to_constraint_raise['BIDTYPE'] == 'LOWERREG') &
                                                            (units_to_constraint_raise[
                                                                 'CAPACITYBAND'] != 'FCASINTEGER'),
                                                            -1, units_to_constraint_raise['LHSCOEFFICIENTS'])
    units_to_constraint_raise['LHSCOEFFICIENTS'] = \
        np.where((units_to_constraint_raise['BIDTYPE'] == raise_contingency_service) &
                 (units_to_constraint_raise['CAPACITYBAND'] != 'FCASINTEGER'),
                 -1 * units_to_constraint_raise['LOWERSLOPE'],
                 units_to_constraint_raise['LHSCOEFFICIENTS'])
    print('2 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_raise['RHSCONSTANT'] = units_to_constraint_raise['ENABLEMENTMIN']
    units_to_constraint_raise['CONSTRAINTTYPE'] = '>='
    #units_to_constraint_raise_rows = units_to_constraint_raise.groupby('DUID', as_index=False).first()
    #units_to_constraint_raise_rows = save_index(units_to_constraint_raise_rows, 'ROWINDEX', max_con + 1)
    #units_to_constraint_raise_rows = units_to_constraint_raise_rows.loc[:, ('DUID', 'ROWINDEX')]
    #units_to_constraint_raise = pd.merge(units_to_constraint_raise, units_to_constraint_raise_rows, 'left', 'DUID')
    unique_duids = units_to_constraint_raise['DUID'].unique()
    constraint_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    units_to_constraint_raise['ROWINDEX'] = units_to_constraint_raise['DUID'].map(constraint_rows)
    units_to_constraint_raise = \
        units_to_constraint_raise.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT')]
    print('3 {}'.format(perf_counter() - t0))
    print('create_joint_capacity_constraints_lower {}'.format(perf_counter() - ta))
    return [units_to_constraint_raise]


def create_joint_ramping_constraints(bids_and_indexes, initial_conditions, max_con, regulation_service, bid_type_check):
    ta = perf_counter()
    t0 = perf_counter()
    units_with_reg_and_energy = \
        bid_type_check[(bid_type_check[regulation_service] == 1) | (bid_type_check['ENERGY'] == 1)]
    #units_with_reg_and_energy = units_with_reg_and_energy.reset_index(drop=True)
    unique_duids = units_with_reg_and_energy['DUID'].unique()
    constraint_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    #constraint_rows = save_index(units_with_reg_and_energy, 'ROWINDEX', max_con + 1)
    print('1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    applicable_bids = bids_and_indexes[(bids_and_indexes['BIDTYPE'] == 'ENERGY') |
                                       (bids_and_indexes['BIDTYPE'] == regulation_service)]
    constraint_variables = applicable_bids[applicable_bids['DUID'].isin(unique_duids)]
    initial_conditions = initial_conditions.loc[:, ('DUID', 'INITIALMW', 'RAMPDOWNRATE', 'RAMPUPRATE')]
    #constraint_rows = constraint_rows.loc[:, ('DUID', 'ROWINDEX')]
    #constraint_variables = pd.merge(constraint_variables, constraint_rows, 'left', 'DUID')
    constraint_variables['ROWINDEX'] = constraint_variables['DUID'].map(constraint_rows)
    constraint_variables = pd.merge(constraint_variables, initial_conditions, 'left', 'DUID')
    print('2 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    if regulation_service == 'RAISEREG':
        constraint_variables['CONSTRAINTTYPE'] = '<='
        constraint_variables['RHSCONSTANT'] = \
            constraint_variables['INITIALMW'] + (constraint_variables['RAMPUPRATE'] / 12)
        constraint_variables['LHSCOEFFICIENTS'] = 1
    elif regulation_service == 'LOWERREG':
        constraint_variables['CONSTRAINTTYPE'] = '>='
        constraint_variables['RHSCONSTANT'] = \
            constraint_variables['INITIALMW'] - (constraint_variables['RAMPDOWNRATE'] / 12)
        constraint_variables['LHSCOEFFICIENTS'] = np.where(constraint_variables['BIDTYPE'] == 'LOWERREG', -1, 1)
    constraint_variables = \
        constraint_variables.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT')]
    print('2 {}'.format(perf_counter() - t0))
    print('create_joint_ramping_constraints {}'.format(perf_counter() - ta))
    return [constraint_variables]


def joint_energy_and_reg_constraints(bids_and_indexes, capacity_bids, max_con, reg_service, bid_type_check):
    ta = perf_counter()
    t0 = perf_counter()
    units_with_reg_and_energy = bid_type_check[(bid_type_check[reg_service] == 1) & (bid_type_check['ENERGY'] == 1)]
    #units_with_reg_and_energy = units_with_reg_and_energy.reset_index(drop=True)

    #applicable_bids = bids_and_indexes[(bids_and_indexes['BIDTYPE'] == 'ENERGY') |
    #                                   (bids_and_indexes['BIDTYPE'] == reg_service)]
    print('Set up 1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units = list(units_with_reg_and_energy['DUID'])
    print('Set up 2 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    constraint_variables = bids_and_indexes[(bids_and_indexes['DUID'].isin(units) &
                                             ((bids_and_indexes['BIDTYPE'] == 'ENERGY') |
                                              (bids_and_indexes['BIDTYPE'] == reg_service)))].copy()
    #constraint_variables = pd.merge(bids_and_indexes, units_with_reg_and_energy, 'left', on='DUID')
    print('Set up 3 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    slope_coefficients = capacity_bids[(capacity_bids['BIDTYPE'] == reg_service) &
                                       (capacity_bids['DUID'].isin(units))].copy()
    print('Slope 1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    #slope_coefficients = slope_coefficients[slope_coefficients['BIDTYPE'] == reg_service]
    slope_coefficients['UPPERSLOPE'] = ((slope_coefficients['ENABLEMENTMAX'] - slope_coefficients['HIGHBREAKPOINT']) /
                                        slope_coefficients['MAXAVAIL'])
    slope_coefficients['LOWERSLOPE'] = ((slope_coefficients['LOWBREAKPOINT'] - slope_coefficients['ENABLEMENTMIN']) /
                                        slope_coefficients['MAXAVAIL'])
    print('Slope 2 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    slope_coefficients = \
         slope_coefficients.loc[:, ['DUID', 'UPPERSLOPE', 'LOWERSLOPE', 'ENABLEMENTMAX', 'ENABLEMENTMIN']]
    print('Slope 3 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    constraint_variables = pd.merge(constraint_variables, slope_coefficients, 'left', on='DUID')
    print('Upper 1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_upper = constraint_variables.copy()
    #units_to_constraint_upper = pd.merge(constraint_variables, slope_coefficients, 'left', 'DUID')
    #units_to_constraint_upper['LHSCOEFFICIENTS'] = np.where(units_to_constraint_upper['BIDTYPE'] == 'ENERGY', 1, 0)
    units_to_constraint_upper['LHSCOEFFICIENTS'] = np.where((units_to_constraint_upper['BIDTYPE'] == reg_service),
                                                             units_to_constraint_upper['UPPERSLOPE'], 1)
    units_to_constraint_upper['RHSCONSTANT'] = units_to_constraint_upper['ENABLEMENTMAX']
    units_to_constraint_upper['CONSTRAINTTYPE'] = '<='
    print('Upper 2 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    unique_duids = units_to_constraint_upper['DUID'].unique()
    units_to_constraint_upper_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    print('Group 1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_lower = constraint_variables.copy()
    #units_to_constraint_lower = pd.merge(constraint_variables, slope_coefficients, 'left', 'DUID')
    print('Copy 1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    #units_to_constraint_lower['LHSCOEFFICIENTS'] = np.where(units_to_constraint_lower['BIDTYPE'] == 'ENERGY', 1, 0)
    print('Replace 1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_lower['LHSCOEFFICIENTS'] = np.where((units_to_constraint_lower['BIDTYPE'] == reg_service),
                                                            -1 * units_to_constraint_lower['LOWERSLOPE'], 1)
    units_to_constraint_lower['RHSCONSTANT'] = units_to_constraint_lower['ENABLEMENTMIN']
    units_to_constraint_lower['CONSTRAINTTYPE'] = '>='
    #units_to_constraint_lower_rows = units_to_constraint_lower.groupby('DUID', as_index=False).first()
    print('Lower 1 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_upper['ROWINDEX'] = units_to_constraint_upper['DUID'].map(units_to_constraint_upper_rows)
    print('Upper 3 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    max_con = hf.max_constraint_index(units_to_constraint_upper)
    unique_duids = units_to_constraint_lower['DUID'].unique()
    units_to_constraint_lower_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    units_to_constraint_lower['ROWINDEX'] = units_to_constraint_lower['DUID'].map(units_to_constraint_lower_rows)
    print('Lower 2 {}'.format(perf_counter() - t0))
    t0 = perf_counter()
    units_to_constraint_lower = units_to_constraint_lower[['INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT']]
    units_to_constraint_upper = units_to_constraint_upper[['INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT']]
    #units_to_constraint = pd.concat([units_to_constraint_lower, units_to_constraint_upper])
    #units_to_constraint = units_to_constraint[['INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT']]
    print('Finnish {}'.format(perf_counter() - t0))
    print('Total {}'.format(perf_counter() - ta))
    return [units_to_constraint_upper, units_to_constraint_lower]



