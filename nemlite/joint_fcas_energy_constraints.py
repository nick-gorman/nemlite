import pandas as pd
from nemlite import helper_functions as hf
import numpy as np


def create_joint_capacity_constraints(bids_and_indexes, capacity_bids, initial_conditions, max_con):
    # Pre calculate at table that allows for the efficient selection of generators according to which markets they are
    # bidding into
    bid_type_check = bids_and_indexes.copy()
    bid_type_check = bid_type_check.loc[:, ('DUID', 'BIDTYPE')]
    bid_type_check = bid_type_check.drop_duplicates(['DUID', 'BIDTYPE'])
    bid_type_check['PRESENT'] = 1
    bid_type_check = bid_type_check.pivot('DUID', 'BIDTYPE', 'PRESENT')
    bid_type_check = bid_type_check.fillna(0)
    bid_type_check['DUID'] = bid_type_check.index
    combined_joint_capacity_constraints = []
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
    combined_joint_capacity_constraints = pd.concat(combined_joint_capacity_constraints)
    return combined_joint_capacity_constraints


def create_joint_capacity_constraints_raise(bids_and_indexes, capacity_bids, max_con, raise_contingency_service,
                                            bid_type_check):
    units_with_reg_or_energy = bid_type_check[(bid_type_check['RAISEREG'] == 1) | (bid_type_check['ENERGY'] == 1)]
    units_with_raise_contingency = bids_and_indexes[(bids_and_indexes['BIDTYPE'] == raise_contingency_service)]
    units = set(units_with_reg_or_energy['DUID']).intersection(units_with_raise_contingency['DUID'])
    units_to_constraint_raise = bids_and_indexes[bids_and_indexes['DUID'].isin(units)]
    upper_slope_coefficients = capacity_bids.copy()
    upper_slope_coefficients = \
        upper_slope_coefficients[upper_slope_coefficients['BIDTYPE'] == raise_contingency_service]
    upper_slope_coefficients['UPPERSLOPE'] = ((upper_slope_coefficients['ENABLEMENTMAX'] -
                                               upper_slope_coefficients['HIGHBREAKPOINT']) /
                                              upper_slope_coefficients['MAXAVAIL'])
    upper_slope_coefficients = upper_slope_coefficients.loc[:, ('DUID', 'UPPERSLOPE', 'ENABLEMENTMAX')]
    units_to_constraint_raise = pd.merge(units_to_constraint_raise, upper_slope_coefficients, 'left', 'DUID')
    units_to_constraint_raise['LHSCOEFFICIENTS'] = np.where(units_to_constraint_raise['BIDTYPE'] == 'ENERGY', 1, 0)
    units_to_constraint_raise['LHSCOEFFICIENTS'] = np.where((units_to_constraint_raise['BIDTYPE'] == 'RAISEREG') &
                                                            (units_to_constraint_raise[
                                                                 'CAPACITYBAND'] != 'FCASINTEGER'),
                                                            1, units_to_constraint_raise['LHSCOEFFICIENTS'])
    units_to_constraint_raise['LHSCOEFFICIENTS'] = \
        np.where((units_to_constraint_raise['BIDTYPE'] == raise_contingency_service) &
                 (units_to_constraint_raise['CAPACITYBAND'] != 'FCASINTEGER'), units_to_constraint_raise['UPPERSLOPE'],
                 units_to_constraint_raise['LHSCOEFFICIENTS'])
    units_to_constraint_raise['RHSCONSTANT'] = units_to_constraint_raise['ENABLEMENTMAX']
    units_to_constraint_raise['CONSTRAINTTYPE'] = '<='
    constraint_rows = dict(zip(units, np.arange(max_con + 1, max_con + 1 + len(units))))
    units_to_constraint_raise['ROWINDEX'] = units_to_constraint_raise['DUID'].map(constraint_rows)
    units_to_constraint_raise = \
        units_to_constraint_raise.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT')]
    return [units_to_constraint_raise]


def create_joint_capacity_constraints_lower(bids_and_indexes, capacity_bids, max_con, raise_contingency_service,
                                            bid_type_check):
    units_with_reg_or_energy = bid_type_check[(bid_type_check['LOWERREG'] == 1) | (bid_type_check['ENERGY'] == 1)]
    units_with_raise_contingency = bids_and_indexes[(bids_and_indexes['BIDTYPE'] == raise_contingency_service)]
    units_to_constraint_raise = bids_and_indexes[
        (bids_and_indexes['DUID'].isin(list(units_with_reg_or_energy['DUID']))) &
        (bids_and_indexes['DUID'].isin(list(units_with_raise_contingency['DUID']))) &
        ((bids_and_indexes['BIDTYPE'] == 'LOWERREG') |
         (bids_and_indexes['BIDTYPE'] == 'ENERGY') |
         (bids_and_indexes['BIDTYPE'] == raise_contingency_service))]
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
    units_to_constraint_raise['RHSCONSTANT'] = units_to_constraint_raise['ENABLEMENTMIN']
    units_to_constraint_raise['CONSTRAINTTYPE'] = '>='
    unique_duids = units_to_constraint_raise['DUID'].unique()
    constraint_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    units_to_constraint_raise['ROWINDEX'] = units_to_constraint_raise['DUID'].map(constraint_rows)
    units_to_constraint_raise = \
        units_to_constraint_raise.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT')]
    return [units_to_constraint_raise]


def create_joint_ramping_constraints(bids_and_indexes, initial_conditions, max_con, regulation_service, bid_type_check):
    unique_duids = get_duids_that_joint_ramping_constraints_apply_to(bid_type_check, initial_conditions)
    constraint_variables = setup_data_to_calc_joint_ramping_constraints(bids_and_indexes, initial_conditions,
                                                                        unique_duids, regulation_service, max_con)
    constraint_variables = calc_constraint_values(constraint_variables, regulation_service)
    constraint_variables = \
        constraint_variables.loc[:, ('INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT')]
    return [constraint_variables]


def get_duids_that_joint_ramping_constraints_apply_to(bid_type_check, initial_conditions, regulation_service):
    units_with_reg_and_energy = \
        bid_type_check[(bid_type_check[regulation_service] == 1) & (bid_type_check['ENERGY'] == 1)]
    unique_duids = units_with_reg_and_energy['DUID'].unique()
    return unique_duids


def setup_data_to_calc_joint_ramping_constraints(unique_duids, bids_and_indexes, initial_conditions,
                                                 regulation_service, max_con):
    constraint_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    applicable_bids = bids_and_indexes[(bids_and_indexes['BIDTYPE'] == 'ENERGY') |
                                       (bids_and_indexes['BIDTYPE'] == regulation_service)]
    constraint_variables = applicable_bids[applicable_bids['DUID'].isin(unique_duids)]
    initial_conditions = initial_conditions.loc[:, ('DUID', 'INITIALMW', 'RAMPDOWNRATE', 'RAMPUPRATE')]
    constraint_variables['ROWINDEX'] = constraint_variables['DUID'].map(constraint_rows)
    constraint_variables = pd.merge(constraint_variables, initial_conditions, 'left', 'DUID')
    return constraint_variables


def calc_constraint_values(constraint_variables, regulation_service):
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
    return calc_constraint_values


def joint_energy_and_reg_constraints(bids_and_indexes, capacity_bids, max_con, reg_service, bid_type_check):
    units_with_reg_and_energy = bid_type_check[(bid_type_check[reg_service] == 1) & (bid_type_check['ENERGY'] == 1)]
    units = list(units_with_reg_and_energy['DUID'])
    constraint_variables = bids_and_indexes[(bids_and_indexes['DUID'].isin(units) &
                                             ((bids_and_indexes['BIDTYPE'] == 'ENERGY') |
                                              (bids_and_indexes['BIDTYPE'] == reg_service)))].copy()
    slope_coefficients = capacity_bids[(capacity_bids['BIDTYPE'] == reg_service) &
                                       (capacity_bids['DUID'].isin(units))].copy()
    slope_coefficients['UPPERSLOPE'] = ((slope_coefficients['ENABLEMENTMAX'] - slope_coefficients['HIGHBREAKPOINT']) /
                                        slope_coefficients['MAXAVAIL'])
    slope_coefficients['LOWERSLOPE'] = ((slope_coefficients['LOWBREAKPOINT'] - slope_coefficients['ENABLEMENTMIN']) /
                                        slope_coefficients['MAXAVAIL'])
    slope_coefficients = \
         slope_coefficients.loc[:, ['DUID', 'UPPERSLOPE', 'LOWERSLOPE', 'ENABLEMENTMAX', 'ENABLEMENTMIN']]
    constraint_variables = pd.merge(constraint_variables, slope_coefficients, 'left', on='DUID')
    units_to_constraint_upper = constraint_variables.copy()
    units_to_constraint_upper['LHSCOEFFICIENTS'] = np.where((units_to_constraint_upper['BIDTYPE'] == reg_service),
                                                             units_to_constraint_upper['UPPERSLOPE'], 1)
    units_to_constraint_upper['RHSCONSTANT'] = units_to_constraint_upper['ENABLEMENTMAX']
    units_to_constraint_upper['CONSTRAINTTYPE'] = '<='
    unique_duids = units_to_constraint_upper['DUID'].unique()
    units_to_constraint_upper_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    units_to_constraint_lower = constraint_variables.copy()
    units_to_constraint_lower['LHSCOEFFICIENTS'] = np.where((units_to_constraint_lower['BIDTYPE'] == reg_service),
                                                            -1 * units_to_constraint_lower['LOWERSLOPE'], 1)
    units_to_constraint_lower['RHSCONSTANT'] = units_to_constraint_lower['ENABLEMENTMIN']
    units_to_constraint_lower['CONSTRAINTTYPE'] = '>='
    units_to_constraint_upper['ROWINDEX'] = units_to_constraint_upper['DUID'].map(units_to_constraint_upper_rows)
    max_con = hf.max_constraint_index(units_to_constraint_upper)
    unique_duids = units_to_constraint_lower['DUID'].unique()
    units_to_constraint_lower_rows = dict(zip(unique_duids, np.arange(max_con + 1, max_con + 1 + len(unique_duids))))
    units_to_constraint_lower['ROWINDEX'] = units_to_constraint_lower['DUID'].map(units_to_constraint_lower_rows)
    units_to_constraint_lower = \
        units_to_constraint_lower[['INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT']]
    units_to_constraint_upper = \
        units_to_constraint_upper[['INDEX', 'ROWINDEX', 'LHSCOEFFICIENTS', 'CONSTRAINTTYPE', 'RHSCONSTANT']]
    return [units_to_constraint_upper, units_to_constraint_lower]