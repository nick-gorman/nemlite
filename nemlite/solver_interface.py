import pandas as pd
import numpy as np
from mip import Model, xsum, minimize, INTEGER, OptimizationStatus
from time import time


def solve_lp(bid_bounds, inter_bounds, combined_constraints, objective_coefficients,
             rhs_and_inequality_types, region_req_by_row, regions_to_price):

    # Create the mip object.
    prob = Model("energymarket")

    # Create the set of variables that define generator energy and FCAS dispatch.
    variables = {}
    for upper_bound, index, band_type in zip(list(bid_bounds['BID']), list(bid_bounds['INDEX']),
                                             list(bid_bounds['CAPACITYBAND'])):
        if (band_type == 'FCASINTEGER') | (band_type == 'INTERTRIGGERVAR'):
            variables[index] = prob.add_var(lb=0, ub=upper_bound, var_type=INTEGER, name=str(index))
        else:
            variables[index] = prob.add_var(lb=0, ub=upper_bound, name=str(index))

    # Create set of variables that define interconnector flow.
    for index, upper_bound in zip(list(inter_bounds['INDEX']), list(inter_bounds['UPPERBOUND'])):
        variables[index] = prob.add_var(lb=0, ub=upper_bound, name=str(index))

    # Define objective function
    tx = time()
    objective_coefficients = objective_coefficients.set_index('INDEX')
    print('time to set obj index {}'.format(time() - tx))
    prob.objective = minimize(xsum(objective_coefficients['BID'][i] * variables[i] for i in list(bid_bounds['INDEX'])))

    # Create a numpy array of the market variables and constraint matrix rows, this improves the efficiency of
    # adding constraints to the linear problem.
    #np_gen_vars = np.asarray(list(variables.values()))

    #vars in dict
    # Add the market constraints to the linear problem.
    tx = time()
    vars = pd.DataFrame(list(variables.items()), columns=['INDEX', 'VARS'])
    combined_constraints = pd.merge(combined_constraints, vars, "left", on='INDEX')
    combined_constraints['LHSCOEFFICIENTSVARS'] = combined_constraints['LHSCOEFFICIENTS'] * combined_constraints['VARS']
    #combined_constraints['LHSCOEFFICIENTSVARS'] = combined_constraints.groupby('ROWINDEX', as_index=False)['LHSCOEFFICIENTSVARS'].agg(xsum)
    print('time to set combined_constraints index {}'.format(time() - tx))
    tx = time()
    rhs = dict(zip(rhs_and_inequality_types['ROWINDEX'], rhs_and_inequality_types['RHSCONSTANT']))
    enq_type = dict(zip(rhs_and_inequality_types['ROWINDEX'], rhs_and_inequality_types['ENQUALITYTYPE']))
    print('time to set rhs_and_inequality_types index {}'.format(time() - tx))
    number = 0
    map = {}
    tx = time()
    tx1 = 0
    tx2 = 0
    combined_constraints = combined_constraints.set_index('ROWINDEX')
    row_groups = combined_constraints.loc[:, ['LHSCOEFFICIENTSVARS']].groupby('ROWINDEX')
    con = []
    name = []
    for i, row_group in row_groups:
        # Record the mapping between the index used to name a constraint internally to the pulp code and the row
        # index it is given in nemlite. This mapping allows constraints to be identified by the nemlite index and
        # modified later.
        b = time()
        exp = xsum(row_group['LHSCOEFFICIENTSVARS'])
        tx1 += time() - b
        b = time()
        new_constraint = make_constraint(exp, rhs[i], enq_type[i], marginal_offset=0)
        tx2 += time() - b
        prob.add_constr(new_constraint, name=str(i))
        map[i] = number
        number += 1
    print('time to xsum {}'.format(tx1))
    print('time to set add_constr make {}'.format(tx2))
    print('time to set add_constr all  {}'.format(time() - tx))
    # Dicts to store results on a run basis, a base run, and pricing run for each region.
    dispatches = {}
    inter_flows = {}

    # Copy initial problem so subsequent runs can use it.
    base_prob = prob.copy()

    # Solve for the base case.
    status = base_prob.optimize()
    # Check of a solution has been found.
    if status != OptimizationStatus.OPTIMAL:
        # Attempt find constraint causing infeasibility.
        con_index = find_problem_constraint(base_prob)
        print('Couldn\'t find an optimal solution, but removing con {} fixed INFEASIBLITY'.format(con_index))

    # Save base case results
    dispatches['BASERUN'] = gen_outputs(base_prob.vars, bid_bounds)
    inter_flows['BASERUN'] = gen_outputs(base_prob.vars, inter_bounds)

    # Perform pricing runs for each region.
    for region in regions_to_price:
        prob_marginal = prob.copy()
        row_index = get_region_load_constraint_index(region_req_by_row, region)
        constraint = prob_marginal.constrs[map[row_index]]
        prob_marginal.remove(constraint)
        row_group = combined_constraints[combined_constraints['ROWINDEX'] == row_index]
        new_constraint = make_constraint(row_group, rhs[row_index], enq_type[row_index], marginal_offset=1)
        prob_marginal.add_constr(new_constraint, name=str(row_index))
        prob_marginal.optimize()
        dispatches[region] = gen_outputs(prob_marginal.vars, bid_bounds)
        inter_flows[region] = gen_outputs(prob_marginal.vars, inter_bounds)
    return dispatches, inter_flows


def make_constraint(exp, rhs, enq_type, marginal_offset=0):
    # Multiply the variables and the coefficients to form the lhs.
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


def gen_outputs(solution, var_definitions):
    # Create a data frame that outlines the solution values of the market variables given to the function.
    summary = pd.DataFrame()
    index = []
    dispatched = []

    for variable in solution:
        index.append(int(variable.name))
        dispatched.append(variable.x)

    # Construct the data frame.
    summary['INDEX'] = index
    summary['DISPATCHED'] = dispatched
    dispatch = pd.merge(var_definitions, summary, 'inner', on=['INDEX'])
    return dispatch

