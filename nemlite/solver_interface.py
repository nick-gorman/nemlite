import pandas as pd
import numpy as np
from time import time
from cvxopt.modeling import op, dot, variable, matrix


def solve_lp(bid_bounds, inter_bounds, combined_constraints, objective_coefficients,
             rhs_and_inequality_types, region_req_by_row, regions_to_price):

    # Vars
    t0 = time()
    inter_bounds = inter_bounds.drop_duplicates('INDEX')
    bid_bounds = bid_bounds.drop_duplicates('INDEX')
    bid_bounds = bid_bounds.loc[:, ['INDEX', 'BID']]
    bid_bounds.columns = ['INDEX', 'UPPERBOUND']
    all_vars = pd.concat([bid_bounds, inter_bounds.loc[:, ['INDEX', 'UPPERBOUND']]])
    all_vars['LOWERBOUND'] = 0
    all_vars = all_vars.sort_values('INDEX')
    all_vars = save_index(all_vars, 'ROWINDEX', combined_constraints['ROWINDEX'].max() + 1)
    all_vars['RHSCONSTANT'] = all_vars["UPPERBOUND"]
    all_vars['RHSCONSTANT'] = all_vars["UPPERBOUND"]
    all_vars['RHSCONSTANT'] = all_vars["UPPERBOUND"]
    upper_bounds = all_vars.loc[:, ['INDEX', 'ROWINDEX', 'RHSCONSTANT', "LHSCOEFFICIENTS", 'ENQUALITYTYPE']]
    print('Vars {}'.format(time()-t0))
    x = variables(len(all_vars))

    # Objective
    t0 = time()
    objective_coefficients = objective_coefficients.loc[:, ['INDEX', 'BIDS']]
    objective_coefficients = pd.merge(all_vars, objective_coefficients, 'left', on='INDEX')
    objective_coefficients = objective_coefficients.fillna(0)
    objective_coefficients = objective_coefficients.sort_values('INDEX')
    c = matrix(objective_coefficients['BIDS'])
    print('Obj {}'.format(time() - t0))
    
    # RHS
    t0 = time()
    rhs_and_inequality_types = rhs_and_inequality_types.loc[:, ['ROWINDEX', 'ENQUALITYTYPE', 'RHSCONSTANT']]
    rhs_and_inequality_types = rhs_and_inequality_types.sort_values('ROWINDEX')
    rhs_and_inequality_types["RHSCONSTANT"] = np.where(rhs_and_inequality_types['ENQUALITYTYPE'] == 'equal_or_greater',
                                                           rhs_and_inequality_types["RHSCONSTANT"] * -1,
                                                           rhs_and_inequality_types["RHSCONSTANT"])
    rhs_and_inequality_types_en = rhs_and_inequality_types[rhs_and_inequality_types['ENQUALITYTYPE'] != 'equal']
    rhs_and_inequality_types_eq = rhs_and_inequality_types[rhs_and_inequality_types['ENQUALITYTYPE'] == 'equal']
    b = matrix(rhs_and_inequality_types['RHSCONSTANT'])
    print('RHS {}'.format(time() - t0))
    
    # Matrix
    t0 = time()
    combined_constraints = pd.merge(combined_constraints, rhs_and_inequality_types, 'left', on='ROWINDEX')
    combined_constraints["LHSCOEFFICIENTS"] = np.where(combined_constraints['ENQUALITYTYPE'] == 'equal_or_greater',
                                                       combined_constraints["LHSCOEFFICIENTS"] * -1,
                                                       combined_constraints["LHSCOEFFICIENTS"])
    print('Matrix 1/3 {}'.format(time() - t0))
    t0 =time()
    combined_data_matrix_format = combined_constraints.pivot('ROWINDEX', 'INDEX', "LHSCOEFFICIENTS")
    print('Matrix 2/3 {}'.format(time() - t0))
    t0 = time()
    #combined_data_matrix_format = combined_constraints.set_index(['ROWINDEX', 'INDEX'])
    #combined_data_matrix_format = combined_data_matrix_format.unstack('INDEX', fill_value=0)
    combined_data_matrix_format = combined_data_matrix_format.fillna(0)
    print('Matrix fillna {}'.format(time() - t0))
    t0 = time()
    #combined_data_matrix_format_en = combined_data_matrix_format.loc[rhs_and_inequality_types[rhs_and_inequality_types['ENQUALITYTYPE'] != 'equal']['ROWINDEX'], :]
    #combined_data_matrix_format_eq = combined_data_matrix_format.loc[rhs_and_inequality_types[rhs_and_inequality_types['ENQUALITYTYPE'] == 'equal']['ROWINDEX'], :]
    #combined_data_matrix_format_eq = np.asarray(combined_data_matrix_format_eq)
    #combined_data_matrix_format_en = np.asarray(combined_data_matrix_format_en)
    A = matrix(np.asarray(combined_data_matrix_format))
    print('Matrix 3/3 {}'.format(time() - t0))

    # Solve
    t0 = time()
    ineq = (A * x <= b)
    lp2 = op(dot(c, x), ineq)
    lp2.solve()
    #results = optimize.linprog(c=objective_coefficients['BIDS'],
    #                           A_ub=combined_data_matrix_format_en,
    #                           b_ub=rhs_and_inequality_types_en['RHSCONSTANT'],
    #                           A_eq=combined_data_matrix_format_eq ,
    #                           b_eq=rhs_and_inequality_types_eq["RHSCONSTANT"],
    #                           bounds=all_vars.loc[:,['LOWERBOUND', 'UPPERBOUND']].values,
    #                           options={'presolve': True, 'lstsq': True})
    print('Solve {}'.format(time() - t0))

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


def save_index(dataframe, new_col_name, offset=0):
    # Save the indexes of the data frame as an np array.
    index_list = np.array(dataframe.index.values)
    # Add an offset to each element of the array.
    offset_index_list = index_list + offset
    # Add the list of indexes as a column to the data frame.
    dataframe[new_col_name] = offset_index_list
    return dataframe
