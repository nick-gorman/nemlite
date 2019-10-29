import pulp
import pandas as pd
from shutil import copyfile
from joblib import delayed
import os
import subprocess
import numpy as np
import re
from mip import Model, xsum, minimize, INTEGER


class EnergyMarketLp:
    def __init__(self, number_of_variables, number_of_constraints, bid_bounds, inter_bounds,
                 constraint_matrix, objective_coefficients, row_rhs_values, names, inequality_types,
                 inter_penalty_factors, indices, region_req_by_row, mnsp_link_indexes, market_cap_and_floor):

        self.number_of_variables = number_of_variables
        self.number_of_constraints = number_of_constraints
        self.bid_bounds = bid_bounds
        self.inter_bounds = inter_bounds
        self.constraint_matrix = constraint_matrix
        self.objective_coefficients = objective_coefficients
        self.row_rhs_values = row_rhs_values
        self.names = names
        self.inequality_types = inequality_types
        self.penalty_factors = inter_penalty_factors
        self.indices = indices
        self.region_req_by_row = region_req_by_row
        self.prob = None
        self.name_index_to_row_index = {}
        self.saved_region_cons = {}
        self.mnsp_link_indexes = mnsp_link_indexes
        self.mcp = market_cap_and_floor['VOLL'][0]

    def pulp_setup(self):
        # --- Start Linear Program Definitions
        # prob = pulp.LpProblem("energymarket", pulp.LpMinimize)
        prob = Model("energymarket")
        # Create the set of market variables.
        #variables = pulp.LpVariable.dicts("Variables", list(range(self.number_of_variables)))
        #self.variables = dict((v.name, k) for k, v in variables.items())
        # Set the properties of the market variables associated with the generators.
        variables = {}
        for upper_bound, index, band_type in zip(list(self.bid_bounds[self.names.col_bid_value]),
                                                 list(self.bid_bounds[self.names.col_variable_index]),
                                                 list(self.bid_bounds[self.names.col_capacity_band_number])):
            # Set the upper bound of the variables, note the lower bound is always zero.
            #variables[int(index)].bounds(0, upper_bound)
            # The variables used to model the FCAS no FCAS decision are set to type integer. Note technically they
            # are binary decision variables, but this is achieved through a type of integer and an upper bound of 1.
            if (band_type == self.names.col_fcas_integer_variable) | (band_type == 'INTERTRIGGERVAR'):
                #variables[int(index)].cat = 'Integer'
                variables[index] = prob.add_var(lb=0, ub=upper_bound, var_type=INTEGER)
            else:
                variables[index] = prob.add_var(lb=0, ub=upper_bound)

        # Set the properties of the market variables associated with the interconnectors.
        for index, upper_bound in zip(list(self.inter_bounds[self.names.col_variable_index]),
                                      list(self.inter_bounds[self.names.col_upper_bound])):
            #variables[int(index)].bounds(0, upper_bound)
            variables[index] =  prob.add_var(lb=0, ub=upper_bound)

        # Add the variables to the linear problem.
        #prob.addVariables(list(variables.values()))
        # Define objective function
        prob.objective = minimize(xsum(self.objective_coefficients[i] * variables[i] for i in range(self.number_of_variables)))
        # Create a list of the indexes of constraints to which penalty factors apply.
        penalty_factor_indexes = self.penalty_factors['ROWINDEX'].tolist()
        var_indexes = range(self.number_of_variables)
        constraint_matrix = self.constraint_matrix
        row_rhs_values = self.row_rhs_values
        inequality_types = self.inequality_types
        # Create a numpy array of the market variables and constraint matrix rows, this improves the efficiency of
        # adding constraints to the linear problem.
        np_gen_vars = np.asarray(list(variables.values()))
        np_constraint_matrix = np.asarray(constraint_matrix)
        gen_var_values = list(variables.values())

        # Add the market constraints to the linear problem.
        for i in range(self.number_of_constraints):
            # Record the mapping between the index used to name a constraint internally to the pulp code and the row
            # index it is given in nemlite. This mapping allows constraints to be identified by the nemlite index and
            # modified later.
            self.name_index_to_row_index[self.indices[i]] = i + 1
            # If a constraint uses a penalty factor it needs to be added to the problem in a specific way.
            #if self.indices[i] in penalty_factor_indexes:
                # Select the indexes of the variables used in the constraint.
                #indx = np.nonzero(np_constraint_matrix[i])
                # Select the names of the variables used in the constraint.
                #gen_var_values = np_gen_vars[indx].tolist()
                # Select the the coefficients of the variables used in the constraint.
                #cm = np_constraint_matrix[i][indx].tolist()
                # Create an object representing the left hand side of the constraint.
                #lhs = pulp.LpAffineExpression(zip(gen_var_values, cm))
                # Add the constraint based on its inequality type.
                #if inequality_types[i] == 'equal_or_less':
                #     constraint = pulp.LpConstraint(lhs, pulp.LpConstraintLE, rhs=row_rhs_values[i], name='{}'.format(i))
                #elif inequality_types[i] == 'equal_or_greater':
                 #   constraint = pulp.LpConstraint(lhs, pulp.LpConstraintGE, rhs=row_rhs_values[i], name='{}'.format(i))
                #elif inequality_types[i] == 'equal':
                #    constraint = pulp.LpConstraint(lhs, pulp.LpConstraintEQ, rhs=row_rhs_values[i], name='{}'.format(i))
               # else:
               #     print('missing types')
                # Calculate the penalty associated with the constraint.
                #penalty = self.penalty_factors[self.penalty_factors['ROWINDEX'] == self.indices[i]][
                #              'CONSTRAINTWEIGHT'].reset_index(drop=True).loc[0] * self.mcp
                # Convert the constraint to elastic so it can be broken at the cost of the penalty.
                #constraint = constraint.makeElasticSubProblem(penalty=penalty, proportionFreeBound=0)
                #extend_3(prob, constraint)

            #else:
                # If no penalty factors are associated with the constraint add the constraint with optimised procedure
                # implemented below.
                # Select the indexes of the variables used in the constraint.
            indx = np.nonzero(np_constraint_matrix[i])
            # Multiply the variables and the coefficients to form the lhs. Use numpy arrays for efficiency.
            v = np_constraint_matrix[i][indx] * np_gen_vars[indx]
            # Convert back to list for adding to problem.
            v = v.tolist()
            # Add based on inequality type.
            if inequality_types[i] == 'equal_or_less':
                prob += xsum(v) <= row_rhs_values[i]
            elif inequality_types[i] == 'equal_or_greater':
                prob += xsum(v) >= row_rhs_values[i]
            elif inequality_types[i] == 'equal':
                prob += xsum(v) == row_rhs_values[i]
            else:
                print('missing types')

        # Assign the pulp problem to the nemlite level object.
        self.prob = prob

    def add_marginal_load(self, region, bidtype):
        # Add the marginal load of the region and bid type given. This is done modifying the constraint that forces
        # a regions load to be met.
        # Find the nemlite level index of the region and bid type.
        row_index = self.region_req_by_row[(self.region_req_by_row['REGIONID'] == region) &
                                           (self.region_req_by_row['BIDTYPE'] == bidtype)]['ROWINDEX'].values[0]
        # Map the nemlite level index to the index used in the pulp object.
        name_i = self.name_index_to_row_index[row_index]
        # Modify the constraint to its original value minus 1. Minus 1 is used at the load constraint is defined as a
        # negative value.
        load = self.prob.constraints['_C' + str(name_i)].constant
        self.prob.constraints['_C' + str(name_i)].constant = self.prob.constraints['_C' + str(name_i)].constant - 1
        return '_C' + str(name_i), load

    def remove_marginal_load(self, region, bidtype):
        # Remove the marginal load of the region and bid type given. This is done modifying the constraint that forces
        # a regions load to be met.
        # Find the nemlite level index of the region and bid type.
        row_index = self.region_req_by_row[(self.region_req_by_row['REGIONID'] == region) &
                                           (self.region_req_by_row['BIDTYPE'] == bidtype)]['ROWINDEX'].values[0]
        name_i = self.name_index_to_row_index[row_index]
        # Return the constraint to its original value before the marginal load was added.
        self.prob.constraints['_C' + str(name_i)].constant = self.prob.constraints['_C' + str(name_i)].constant + 1
        return


def extend_3(self, other, use_objective=True):
    """
    extends an LpProblem by adding constraints either from a dictionary
    a tuple or another LpProblem object.

    @param use_objective: determines whether the objective is imported from
    the other problem

    For dictionaries the constraints will be named with the keys
    For tuples an unique name will be generated
    For LpProblems the name of the problem will be added to the constraints
    name
    """
    if isinstance(other, dict):
        for name in other:
            self.constraints[name] = other[name]
    elif isinstance(other, pulp.LpProblem):
        for v in other.variables()[-3:]:
            v.name = other.name + v.name
        for name, c in other.constraints.items():
            c.name = other.name + name
            self.addConstraint(c)
        if use_objective:
            self.objective += other.objective
    else:
        for c in other:
            if isinstance(c, tuple):
                name = c[0]
                c = c[1]
            else:
                name = None
            if not name: name = c.name
            if not name: name = self.unusedConstraintName()
            self.constraints[name] = c


def run_solves(base_prob, var_definitions, inter_definitions, ns, regions_to_price, pool,
               region_req_by_row, name_index_to_row_index):
    # Run the solves required to price given regions. This always includes a base solves and then one additional
    # solve for each region to be priced.
    base_prob.optimize()

    return None


def regional_load_constraint_name(region_req_by_row, region, bidtype, name_index_to_row_index, constraints):
    row_index = region_req_by_row[(region_req_by_row['REGIONID'] == region) &
                                  (region_req_by_row['BIDTYPE'] == bidtype)]['ROWINDEX'].values[0]
    # Map the nemlite level index to the index used in the pulp object.
    name_i = name_index_to_row_index[row_index]
    name = '_C' + str(name_i)
    load = constraints['_C' + str(name_i)].constant
    return name, load


def run_par(region):
    cmd = 'cbc.exe {}-pulp.mps branch printingOptions all solution {}-pulp.sol '.format(region, region)
    pipe = open(os.devnull, 'w')
    cbc = subprocess.Popen((cmd).split(), stdout=pipe, stderr=pipe)
    if cbc.wait() != 0:
        raise ("Pulp: Error while trying to execute ")
    return


def gen_outputs(solution, var_definitions, ns):
    # Create a data frame that outlines the solution values of the market variables given to the function.
    summary = pd.DataFrame()
    index = []
    dispatched = []
    name = []

    for variable in solution.variables():
        # Skip dummy variables that do not correspond to and actual market variable.
        if (variable.name == "__dummy") | ('elastic' in variable.name):
            continue
        # Process the variables name to retrive its index value.
        index.append(int(re.findall('\d+', variable.name)[-1]))
        # Set the variables solution value as the dispatch value.
        dispatched.append(variable.value())
        # Record the variables name.
        name.append(variable.name)

    # Construct the data frame.
    summary[ns.col_variable_index] = index
    summary['DISPATCHED'] = dispatched
    summary['NAME'] = name
    dispatch = pd.merge(var_definitions, summary, 'inner', on=[ns.col_variable_index])
    return dispatch


def gen_outputs_new(values, var_definitions, ns, variables_name_2_index):
    # Create a data frame that outlines the solution values of the market variables given to the function.
    summary = pd.DataFrame()
    indexes = []
    dispatched = []
    name = []

    for name, index in variables_name_2_index.items():
        # Process the variables name to retrive its index value.
        indexes.append(index)
        # Set the variables solution value as the dispatch value.
        dispatched.append(values[name])
        # Record the variables name.
        # name.append(variable.name)

    # Construct the data frame.
    summary[ns.col_variable_index] = indexes
    summary['DISPATCHED'] = dispatched
    # summary['NAME'] = name
    dispatch = pd.merge(var_definitions, summary, 'inner', on=[ns.col_variable_index])
    return dispatch