import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, INTEGER, OptimizationStatus, LinExpr


def solve_lp(bid_bounds, inter_bounds, combined_constraints, objective_coefficients,
             rhs_and_inequality_types, region_req_by_row, regions_to_price):

    # Create the mip object.
    prob = Model("energymarket")

    # Create the set of variables that define generator energy and FCAS dispatch.
    variables = {}
    bid_bounds = bid_bounds.sort_values('INDEX')
    bid_bounds = bid_bounds.reset_index()
    bid_bounds['MIPINDEX'] = bid_bounds.index
    for upper_bound, index, band_type in zip(list(bid_bounds['BID']), list(bid_bounds['INDEX']),
                                             list(bid_bounds['CAPACITYBAND'])):
        if (band_type == 'FCASINTEGER') | (band_type == 'INTERTRIGGERVAR'):
            variables[index] = prob.add_var(lb=0, ub=upper_bound, var_type=INTEGER, name=str(index))
        else:
            variables[index] = prob.add_var(lb=0, ub=upper_bound, name=str(index))

    # Create set of variables that define interconnector flow.
    inter_bounds = inter_bounds.sort_values('INDEX')
    inter_bounds = inter_bounds.reset_index()
    inter_bounds['MIPINDEX'] = inter_bounds.index
    for index, upper_bound in zip(list(inter_bounds['INDEX']), list(inter_bounds['UPPERBOUND'])):
        variables[index] = prob.add_var(lb=0, ub=upper_bound, name=str(index))

    # Define objective function
    objective_coefficients = objective_coefficients.sort_values('INDEX')
    objective_coefficients = objective_coefficients.set_index('INDEX')
    prob.objective = minimize(xsum(objective_coefficients['BID'][i] * variables[i] for i in list(bid_bounds['INDEX'])))

    # Create a numpy array of the market variables and constraint matrix rows, this improves the efficiency of
    # adding constraints to the linear problem.
    combined_constraints = combined_constraints.sort_values('ROWINDEX').reset_index()
    constraint_matrix = combined_constraints.pivot('ROWINDEX', 'INDEX', 'LHSCOEFFICIENTS')
    constraint_matrix = constraint_matrix.sort_index(axis=1)
    row_indices = np.asarray(constraint_matrix.index)
    var_indices = np.asarray(constraint_matrix.columns)
    constraint_matrix = np.asarray(constraint_matrix)
    # constraint_dict = {g: s.tolist() for g, s in combined_constraints['LHSCOEFFICIENTSVARS'].groupby('ROWINDEX')}
    rhs = dict(zip(rhs_and_inequality_types['ROWINDEX'], rhs_and_inequality_types['RHSCONSTANT']))
    enq_type = dict(zip(rhs_and_inequality_types['ROWINDEX'], rhs_and_inequality_types['ENQUALITYTYPE']))
    var_list = np.asarray([v for k, v in variables.items()])
    for i in range(len(row_indices)):
        # Record the mapping between the index used to name a constraint internally to the pulp code and the row
        # index it is given in nemlite. This mapping allows constraints to be identified by the nemlite index and
        # modified later.
        new_constraint = make_constraint(var_list, constraint_matrix[i], rhs[row_indices[i]], enq_type[row_indices[i]],
                                         marginal_offset=0)
        prob.add_constr(new_constraint, name=str(i))

    # Dicts to store results on a run basis, a base run, and pricing run for each region.
    dispatches = {}
    inter_flows = {}

    bid_bounds['VARS'] = [variables[i] for i in bid_bounds['MIPINDEX']]
    bid_bounds['NAMECHECK'] = bid_bounds['VARS'].apply(lambda x: x.name)
    inter_bounds['VARS'] = [variables[i] for i in inter_bounds['MIPINDEX']]
    inter_bounds['NAMECHECK'] = inter_bounds['VARS'].apply(lambda x: x.name)

    # Copy initial problem so subsequent runs can use it.
    base_prob = prob
    # Solve for the base case.
    status = base_prob.optimize()
    # Check of a solution has been found.
    if status != OptimizationStatus.OPTIMAL:
        # Attempt find constraint causing infeasibility.
        con_index = find_problem_constraint(base_prob)
        print('Couldn\'t find an optimal solution, but removing con {} fixed INFEASIBLITY'.format(con_index))

    # Save base case results
    bid_bounds = outputs(bid_bounds)
    inter_bounds = outputs(inter_bounds)
    dispatches['BASERUN'] = strip_gen_outputs_to_minimal(bid_bounds)
    inter_flows['BASERUN'] = strip_inter_outputs_to_minimal(inter_bounds)
    # Perform pricing runs for each region.
    for region in regions_to_price:
        prob_marginal = prob
        row_index = get_region_load_constraint_index(region_req_by_row, region)
        mip_row_index = np.argwhere(row_indices == row_index)[0][0]
        old_constraint_index = get_con_by_name(prob_marginal.constrs, str(mip_row_index))
        old_constraint = prob_marginal.constrs[old_constraint_index]
        prob_marginal.remove(old_constraint)
        new_constraint = make_constraint(var_list, constraint_matrix[mip_row_index], rhs[row_index],
                                         enq_type[row_index], marginal_offset=1)
        prob_marginal.add_constr(new_constraint, name='blah')
        prob_marginal.optimize()
        bid_bounds = outputs(bid_bounds)
        inter_bounds = outputs(inter_bounds)
        dispatches[region] = strip_gen_outputs_to_minimal(bid_bounds)
        inter_flows[region] = strip_inter_outputs_to_minimal(inter_bounds)
        new_constraint_index = get_con_by_name(prob_marginal.constrs, 'blah')
        new_constraint = prob_marginal.constrs[new_constraint_index]
        prob_marginal.remove(new_constraint)
        old_constraint = make_constraint(var_list, constraint_matrix[mip_row_index], rhs[row_index],
                                         enq_type[row_index], marginal_offset=0)
        prob_marginal.add_constr(old_constraint, name=str(mip_row_index))
    return dispatches, inter_flows


def get_con_by_name(constraints, name):
    i = 0
    for con in constraints:
        if con.name == name:
            return i
        i += 1


def make_constraint(var_list, lhs, rhs, enq_type, marginal_offset=0):
    needed_varaiables_indices = np.argwhere(~np.isnan(lhs)).flatten()
    lhs_varaiables = var_list[needed_varaiables_indices]
    lhs = lhs[needed_varaiables_indices]
    exp = lhs_varaiables * lhs
    exp = exp.tolist()
    exp = xsum(exp)
    # Add based on inequality type.
    if enq_type == 'equal_or_less':
        con = exp <= rhs + marginal_offset
    elif enq_type == 'equal_or_greater':
        con = exp >= rhs + marginal_offset
    elif enq_type == 'equal':
        con = exp == rhs + marginal_offset
    else:
        print('missing types')
    return con


def get_region_load_constraint_index(region_req_by_row, region):
    row_index = region_req_by_row[(region_req_by_row['REGIONID'] == region) &
                                  (region_req_by_row['BIDTYPE'] == 'ENERGY')]['ROWINDEX'].values[0]
    return row_index


def find_problem_constraint(base_prob):
    for con in base_prob.constrs:
        con_index = con.name
        base_prob.remove(con)
        status = base_prob.optimize()
        if status == OptimizationStatus.OPTIMAL:
            break
    return con_index


def outputs(var_definitions):
    var_definitions['DISPATCHED'] = var_definitions['VARS'].apply(lambda x: x.x)
    return var_definitions


def strip_gen_outputs_to_minimal(gen_outputs):
    return gen_outputs.loc[:, ['DUID', 'BIDTYPE', 'CAPACITYBAND', 'INDEX', 'PRICE', 'DISPATCHED', 'BID']]


def strip_inter_outputs_to_minimal(gen_outputs):
    return gen_outputs.loc[:, ['INTERCONNECTORID', 'DIRECTION', 'REGIONID', 'BIDTYPE', 'LOSSSEGMENT',
                                                  'MWBREAKPOINT', 'INDEX', 'UPPERBOUND', 'DISPATCHED']]




