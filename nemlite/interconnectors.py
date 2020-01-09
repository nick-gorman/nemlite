import pandas as pd
import numpy as np
from nemlite import helper_functions as hf


def index_inter(inter_info, inter_seg_definitions, max_index) -> pd:
    # Create a direction for each interconnector.
    cols_to_keep = ['INTERCONNECTORID']
    cols_to_stack = ['REGIONFROM', 'REGIONTO']
    type_name = 'DIRECTION'
    value_name = 'REGIONID'
    stacked_inter_directions = hf.stack_columns(inter_info, cols_to_keep, cols_to_stack, type_name, value_name)

    # Select just the interconnector segment data we need.
    inter_segments = inter_seg_definitions.loc[:, ("INTERCONNECTORID", 'LOSSSEGMENT', 'MWBREAKPOINT')]
    # Beak segments for positive flow and negative flow into different groups.
    pos_inter_segments = inter_segments[inter_segments['MWBREAKPOINT'] >= 0]
    neg_inter_segments = inter_segments[inter_segments['MWBREAKPOINT'] < 0]

    # Merge interconnector from directions with positive flow segments to create the segments needed for the from
    # direction.
    pos_inter = pd.merge(
        stacked_inter_directions[stacked_inter_directions['DIRECTION'] == 'REGIONFROM'],
        pos_inter_segments, 'inner', ['INTERCONNECTORID'])
    # Merge interconnector to directions with negative flow segments to create the segments needed for the to
    # direction.
    neg_inter = pd.merge(
        stacked_inter_directions[stacked_inter_directions["DIRECTION"] == 'REGIONTO'],
        neg_inter_segments, 'inner', ['INTERCONNECTORID'])

    # Combine pos and negative segments into one dataframe.
    inter_multiplied_by_energy_types = pd.concat([pos_inter, neg_inter])

    # Sort values so indexing occurs in a logical sequence.
    inter_multiplied_by_types = inter_multiplied_by_energy_types.sort_values(['INTERCONNECTORID', 'DIRECTION',
                                                                              'BIDTYPE', 'LOSSSEGMENT'])
    # Create interconnector variable indexes.
    inter_multiplied_by_types = inter_multiplied_by_types.reset_index(drop=True)
    inter_multiplied_by_types = hf.save_index(inter_multiplied_by_types, 'INDEX', max_index + 1)

    return inter_multiplied_by_types


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
    flow_cons['CONSTRAINTTYPE'] = '<='
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