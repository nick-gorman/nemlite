import pandas as pd
from nemlite import helper_functions as hf
import numpy as np


def create_joint_capacity_constraints(bids_and_indexes, capacity_bids, initial_conditions, max_con):
    # Pre calculate a table that allows for the efficient selection of generators according to which markets they are
    # bidding into
    bid_type_check = bids_and_indexes.copy()
    bid_type_check = bid_type_check.loc[:, ('DUID', 'BIDTYPE')]
    bid_type_check = bid_type_check.drop_duplicates(['DUID', 'BIDTYPE'])
    bid_type_check['PRESENT'] = 1
    bid_type_check = bid_type_check.pivot('DUID', 'BIDTYPE', 'PRESENT')
    bid_type_check = bid_type_check.fillna(0)
    bid_type_check['DUID'] = bid_type_check.index
    combined_joint_capacity_constraints = []

    # Create constraint sets
    for fcas_service in ['RAISE6SEC', 'RAISE60SEC', 'RAISE5MIN', 'LOWER6SEC', 'LOWER60SEC', 'LOWER5MIN',
                         'LOWERREG', 'RAISEREG']:
        if fcas_service in ['RAISE6SEC', 'RAISE60SEC', 'RAISE5MIN', 'LOWER6SEC', 'LOWER60SEC', 'LOWER5MIN']:
            joint_constraints1 = create_joint_capacity_constraints_upper_slope(bids_and_indexes.copy(),
                                                                               capacity_bids.copy(), max_con,
                                                                               fcas_service, bid_type_check)
            max_con = hf.max_constraint_index(joint_constraints1[0])
            joint_constraints2 = create_joint_capacity_constraints_lower_slope(bids_and_indexes.copy(),
                                                                               capacity_bids.copy(), max_con,
                                                                               fcas_service, bid_type_check)
            joint_constraints = joint_constraints1 + joint_constraints2
        if fcas_service in ['LOWERREG', 'RAISEREG']:
            joint_constraints1 = create_joint_ramping_constraints(bids_and_indexes.copy(), initial_conditions.copy(),
                                                                  max_con, fcas_service, bid_type_check)
            max_con = hf.max_constraint_index(joint_constraints1[0])
            joint_constraints2 = joint_energy_and_reg_constraints(bids_and_indexes.copy(), capacity_bids.copy(),
                                                                  max_con, fcas_service, bid_type_check)
            joint_constraints = joint_constraints1 + joint_constraints2

        max_con = hf.max_constraint_index(joint_constraints[-1])
        combined_joint_capacity_constraints += joint_constraints

    # Combine constraint sets into single dataframe
    combined_joint_capacity_constraints = pd.concat(combined_joint_capacity_constraints)
    return combined_joint_capacity_constraints


def create_joint_capacity_constraints_upper_slope(bids_and_indexes, capacity_bids, max_con, contingency_service,
                                                  bid_type_check):
    units_to_constraint, duids = get_units_to_constrain_joint_capacity(bid_type_check, bids_and_indexes,
                                                                        contingency_service, 'RAISEREG')
    upper_slope_coefficients = calc_slope_joint_capacity_upper(capacity_bids.copy(), contingency_service)
    units_to_constraint = define_joint_capacity_constraint_values_upper_slope(units_to_constraint,
                                                                              upper_slope_coefficients)
    units_to_constraint = set_row_index(units_to_constraint, duids, max_con)
    return [units_to_constraint]


def create_joint_capacity_constraints_lower_slope(bids_and_indexes, capacity_bids, max_con, contingency_service,
                                                  bid_type_check):
    units_to_constraint, duids = get_units_to_constrain_joint_capacity(bid_type_check, bids_and_indexes,
                                                                        contingency_service, 'LOWERREG')
    lower_slope_coefficients = calc_slope_joint_capacity_lower(capacity_bids.copy(), contingency_service)
    units_to_constraint = define_joint_capacity_constraint_values_lower_slope(units_to_constraint,
                                                                              lower_slope_coefficients)
    units_to_constraint = set_row_index(units_to_constraint, duids, max_con)
    return [units_to_constraint]


def get_units_to_constrain_joint_capacity(bid_type_check, bids_and_indexes, contingency_service, reg_pair):
    units_with_con_and_energy = bid_type_check[(bid_type_check[contingency_service] == 1) &
                                               (bid_type_check['ENERGY'] == 1)]['DUID'].unique()
    units_to_constraint_raise = bids_and_indexes[
        (bids_and_indexes['DUID'].isin(list(units_with_con_and_energy))) &
        ((bids_and_indexes['BIDTYPE'] == reg_pair) |
         (bids_and_indexes['BIDTYPE'] == 'ENERGY') |
         (bids_and_indexes['BIDTYPE'] == contingency_service))]
    return units_to_constraint_raise, units_with_con_and_energy


def set_row_index(units_to_constraint, duids, max_con):
    constraint_rows = dict(zip(duids, np.arange(max_con + 1, max_con + 1 + len(duids))))
    units_to_constraint['ROWINDEX'] = units_to_constraint['DUID'].map(constraint_rows)
    units_to_constraint = units_to_constraint.loc[:,
                          ['INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT']]
    return units_to_constraint


def define_joint_capacity_constraint_values_upper_slope(units_to_constraint, upper_slope_coefficients):
    units_to_constraint = pd.merge(units_to_constraint, upper_slope_coefficients, 'left', 'DUID')
    units_to_constraint['LHSCOEFFICIENTS'] = 1
    units_to_constraint['LHSCOEFFICIENTS'] = np.where(~units_to_constraint['BIDTYPE'].isin(['ENERGY', 'RAISEREG']),
                                                      units_to_constraint['UPPERSLOPE'],
                                                      units_to_constraint['LHSCOEFFICIENTS'])
    units_to_constraint['RHSCONSTANT'] = units_to_constraint['ENABLEMENTMAX']
    units_to_constraint['CONSTRAINTTYPE'] = '<='
    return units_to_constraint


def define_joint_capacity_constraint_values_lower_slope(units_to_constraint, lower_slope_coefficients):
    units_to_constraint = pd.merge(units_to_constraint, lower_slope_coefficients, 'left', 'DUID')
    units_to_constraint['LHSCOEFFICIENTS'] = 1
    units_to_constraint['LHSCOEFFICIENTS'] = np.where((units_to_constraint['BIDTYPE'] == 'LOWERREG'),
                                                      -1, units_to_constraint['LHSCOEFFICIENTS'])
    units_to_constraint['LHSCOEFFICIENTS'] = np.where(
        ~units_to_constraint['BIDTYPE'].isin(['ENERGY', 'LOWERREG']), -1 * units_to_constraint['LOWERSLOPE'],
        units_to_constraint['LHSCOEFFICIENTS'])
    units_to_constraint['RHSCONSTANT'] = units_to_constraint['ENABLEMENTMIN']
    units_to_constraint['CONSTRAINTTYPE'] = '>='
    return units_to_constraint


def calc_slope_joint_capacity_lower(capacity_bids, contingency_service):
    lower_slope_coefficients = capacity_bids[capacity_bids['BIDTYPE'] == contingency_service].copy()
    lower_slope_coefficients['LOWERSLOPE'] = ((lower_slope_coefficients['LOWBREAKPOINT'] -
                                               lower_slope_coefficients['ENABLEMENTMIN']) /
                                              lower_slope_coefficients['MAXAVAIL'])
    lower_slope_coefficients = lower_slope_coefficients.loc[:, ('DUID', 'LOWERSLOPE', 'ENABLEMENTMIN')]
    return lower_slope_coefficients


def calc_slope_joint_capacity_upper(capacity_bids, contingency_service):
    upper_slope_coefficients = capacity_bids[capacity_bids['BIDTYPE'] == contingency_service].copy()
    upper_slope_coefficients['UPPERSLOPE'] = ((upper_slope_coefficients['ENABLEMENTMAX'] -
                                               upper_slope_coefficients['HIGHBREAKPOINT']) /
                                              upper_slope_coefficients['MAXAVAIL'])
    upper_slope_coefficients = upper_slope_coefficients.loc[:, ('DUID', 'UPPERSLOPE', 'ENABLEMENTMAX')]
    return upper_slope_coefficients


def create_joint_ramping_constraints(bids_and_indexes, initial_conditions, max_con, regulation_service, bid_type_check):
    unique_duids = get_duids_that_joint_ramping_constraints_apply_to(bid_type_check, initial_conditions,
                                                                     regulation_service)
    constraint_variables = setup_data_to_calc_joint_ramping_constraints(unique_duids, bids_and_indexes,
                                                                        initial_conditions, regulation_service, max_con)
    constraint_variables = calc_joint_ramping_constraint_values(constraint_variables, regulation_service)
    constraint_variables = \
        constraint_variables.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT')]
    return [constraint_variables]


def get_duids_that_joint_ramping_constraints_apply_to(bid_type_check, initial_conditions, regulation_service):
    units_with_reg_and_energy = \
        bid_type_check[(bid_type_check[regulation_service] == 1) & (bid_type_check['ENERGY'] == 1)]
    unique_duids = units_with_reg_and_energy['DUID'].unique()
    if regulation_service == 'RAISEREG':
        units_with_ramp = initial_conditions[initial_conditions['RAMPUPRATE'] > 0.0]['DUID'].unique()
    else:
        units_with_ramp = initial_conditions[initial_conditions['RAMPDOWNRATE'] > 0.0]['DUID'].unique()
    unique_duids = list(unique_duids)
    units_with_ramp = list(units_with_ramp)
    return [duid for duid in unique_duids if duid in units_with_ramp]


def setup_data_to_calc_joint_ramping_constraints(unique_duids, bids_and_indexes, initial_conditions,
                                                 regulation_service, max_con):
    constraint_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    applicable_bids = bids_and_indexes[(bids_and_indexes['BIDTYPE'] == 'ENERGY') |
                                       (bids_and_indexes['BIDTYPE'] == regulation_service)]
    constraint_variables = applicable_bids[applicable_bids['DUID'].isin(unique_duids)]
    initial_conditions = initial_conditions.loc[:, ['DUID', 'INITIALMW', 'RAMPUPRATE', 'RAMPDOWNRATE']]
    constraint_variables = pd.merge(constraint_variables, initial_conditions, on='DUID')
    constraint_variables['ROWINDEX'] = constraint_variables['DUID'].map(constraint_rows)
    return constraint_variables


def calc_joint_ramping_constraint_values(constraint_variables, regulation_service):
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
    return constraint_variables


def joint_energy_and_reg_constraints(bids_and_indexes, capacity_bids, max_con, reg_service, bid_type_check):
    constraint_variables, units = \
        joint_energy_and_reg_get_units_to_constrain(bids_and_indexes, reg_service, bid_type_check)
    slope_coefficients = joint_energy_and_reg_slope_coefficients(capacity_bids, reg_service, units)
    constraint_variables = pd.merge(constraint_variables, slope_coefficients, 'left', on='DUID')
    units_to_constraint_upper = \
        joint_energy_and_reg_upper_slope_constraints(constraint_variables, reg_service, max_con)
    max_con = hf.max_constraint_index(units_to_constraint_upper)
    units_to_constraint_lower = \
        joint_energy_and_reg_lower_slope_constraints(constraint_variables, reg_service, max_con)
    return [units_to_constraint_upper, units_to_constraint_lower]


def joint_energy_and_reg_get_units_to_constrain(bids_and_indexes, reg_service, bid_type_check):
    units_with_reg_and_energy = bid_type_check[(bid_type_check[reg_service] == 1) & (bid_type_check['ENERGY'] == 1)]
    units = list(units_with_reg_and_energy['DUID'])
    constraint_variables = bids_and_indexes[(bids_and_indexes['DUID'].isin(units) &
                                             ((bids_and_indexes['BIDTYPE'] == 'ENERGY') |
                                              (bids_and_indexes['BIDTYPE'] == reg_service)))].copy()
    return constraint_variables, units


def joint_energy_and_reg_slope_coefficients(capacity_bids, reg_service, units):
    slope_coefficients = capacity_bids[(capacity_bids['BIDTYPE'] == reg_service) &
                                       (capacity_bids['DUID'].isin(units))].copy()
    slope_coefficients['UPPERSLOPE'] = ((slope_coefficients['ENABLEMENTMAX'] - slope_coefficients['HIGHBREAKPOINT']) /
                                        slope_coefficients['MAXAVAIL'])
    slope_coefficients['LOWERSLOPE'] = ((slope_coefficients['LOWBREAKPOINT'] - slope_coefficients['ENABLEMENTMIN']) /
                                        slope_coefficients['MAXAVAIL'])
    slope_coefficients = \
        slope_coefficients.loc[:, ['DUID', 'UPPERSLOPE', 'LOWERSLOPE', 'ENABLEMENTMAX', 'ENABLEMENTMIN']]
    return slope_coefficients


def joint_energy_and_reg_upper_slope_constraints(constraint_variables, reg_service, max_con):
    units_to_constraint_upper = constraint_variables.copy()
    units_to_constraint_upper['LHSCOEFFICIENTS'] = np.where((units_to_constraint_upper['BIDTYPE'] == reg_service),
                                                            units_to_constraint_upper['UPPERSLOPE'], 1)
    units_to_constraint_upper['RHSCONSTANT'] = units_to_constraint_upper['ENABLEMENTMAX']
    units_to_constraint_upper['CONSTRAINTTYPE'] = '<='
    unique_duids = units_to_constraint_upper['DUID'].unique()
    units_to_constraint_upper_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    units_to_constraint_upper['ROWINDEX'] = units_to_constraint_upper['DUID'].map(units_to_constraint_upper_rows)
    units_to_constraint_upper = \
        units_to_constraint_upper[['INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT']]
    return units_to_constraint_upper


def joint_energy_and_reg_lower_slope_constraints(constraint_variables, reg_service, max_con):
    units_to_constraint_lower = constraint_variables.copy()
    units_to_constraint_lower['LHSCOEFFICIENTS'] = np.where((units_to_constraint_lower['BIDTYPE'] == reg_service),
                                                            -1 * units_to_constraint_lower['LOWERSLOPE'], 1)
    units_to_constraint_lower['RHSCONSTANT'] = units_to_constraint_lower['ENABLEMENTMIN']
    units_to_constraint_lower['CONSTRAINTTYPE'] = '>='

    unique_duids = units_to_constraint_lower['DUID'].unique()
    units_to_constraint_lower_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    units_to_constraint_lower['ROWINDEX'] = units_to_constraint_lower['DUID'].map(units_to_constraint_lower_rows)
    units_to_constraint_lower = \
        units_to_constraint_lower[['INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT']]
    return units_to_constraint_lower
