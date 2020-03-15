import numpy as np
import pandas as pd


def filter_and_scale(capacity_bids, price_bids, unit_solution):
    # Fast start plants must conform to their dispatch profiles, therefore their maximum available energy are overridden
    # to match the dispatch profiles.
    # capacity_bids = over_ride_max_energy_for_fast_start_plants(capacity_bids.copy(), unit_solution, fast_start_gens)

    # Add an additional column that defines the maximum output of a plant, considering its, max avail bid, its stated
    # availability in the previous intervals unit solution and its a ramp rates.
    capacity_bids = add_max_unit_energy(capacity_bids, unit_solution)
    capacity_bids = add_min_unit_energy(capacity_bids, unit_solution)
    capacity_bids = over_ride_max_and_min_energy_for_fast_start_units(capacity_bids, unit_solution,
                                                                      price_bids)

    # If the maximum energy of a plant is below its minimum energy then reset the minimum energy to equal that maximum.
    # Note this means the plants is being constrained up by is minimum energy level.
    capacity_bids = rationalise_max_energy_constraint(capacity_bids)

    # Any units where their maximum available energy is zero have all their bids removed.
    bids_with_zero_avail_removed = remove_energy_bids_with_max_energy_zero(capacity_bids.copy())
    bids_with_zero_avail_removed = remove_fcas_bids_with_max_avail_zero(bids_with_zero_avail_removed)

    bids_with_zero_avail_removed = pd.merge(bids_with_zero_avail_removed,
                                            unit_solution.loc[:,
                                            ('DUID', 'RAISEREGENABLEMENTMAX', 'RAISEREGENABLEMENTMIN',
                                             'LOWERREGENABLEMENTMAX', 'LOWERREGENABLEMENTMIN',
                                             'RAMPDOWNRATE', 'RAMPUPRATE')],
                                            'left', ['DUID'])

    bids_with_scaled_fcas = fcas_trapezium_scaling_on_telemetered_raise_reg_enablement(bids_with_zero_avail_removed)
    bids_with_scaled_fcas = fcas_trapezium_scaling_on_telemetered_lower_reg_enablement(bids_with_scaled_fcas)
    bids_with_scaled_fcas = fcas_trapezium_scaling_on_ramp_up_rate(bids_with_scaled_fcas)
    bids_with_scaled_fcas = fcas_trapezium_scaling_on_ramp_down_rate(bids_with_scaled_fcas)
    bids_with_filtered_fcas = apply_fcas_enablement_criteria(bids_with_scaled_fcas.copy(), unit_solution.copy())
    bids_with_filtered_fcas = bids_with_filtered_fcas.drop(
        ['RAISEREGENABLEMENTMAX', 'RAISEREGENABLEMENTMIN', 'LOWERREGENABLEMENTMAX', 'LOWERREGENABLEMENTMIN',
         'OFFERMAXENERGY'], axis=1)
    return bids_with_filtered_fcas


def add_max_unit_energy(capacity_bids, unit_solution):
    # Select just the energy bids.
    just_energy = capacity_bids[capacity_bids['BIDTYPE'] == 'ENERGY'].copy()

    # Select just the information we need from the energy bids, which is unit and bid number for matching, index and max
    # energy out put of plant.
    just_energy = just_energy.loc[:, ['DUID', 'MAXAVAIL']]
    just_energy.columns = ['DUID', 'OFFERMAXENERGY']

    # The unit solution file provides availability data, ramp rate data and initial condition output data. These are
    # combined to give each unit a new availability number.
    # Combine the initial dispatch of the unit and its ramp rate to give the max dispatch due to ramping constraints.
    # TODO: Remove card code of 5 min dispatch.
    unit_solution['MAXRAMPMW'] = unit_solution['INITIALMW'] + unit_solution['RAMPUPRATE'] / 12
    # Choose the smallest limit between availability and ramp rate constraint as the actual availability.
    unit_solution['MAXENERGY'] = np.where(unit_solution['AVAILABILITY'] > unit_solution['MAXRAMPMW'],
                                          unit_solution['MAXRAMPMW'], unit_solution['AVAILABILITY'])
    # Select just the data needed to map the new availability back to the bidding data.
    just_energy_unit_solution = unit_solution.loc[:, ['DUID', 'MAXENERGY', 'MAXRAMPMW']]

    # Check that availability is not lower than unit bid max energy, if it is then set availability as max energy.
    max_energy = pd.merge(just_energy, just_energy_unit_solution, 'inner', on=['DUID'])
    max_energy['MAXENERGY'] = np.where((max_energy['OFFERMAXENERGY'] < max_energy['MAXENERGY']) &
                                       (max_energy['OFFERMAXENERGY'] > 0.0),
                                       max_energy['OFFERMAXENERGY'], max_energy['MAXENERGY'])
    max_energy = max_energy.loc[:, ['DUID', 'MAXENERGY', 'OFFERMAXENERGY']]

    # Map the max energy availability given by each generator to all the bids given by that generator. This information
    # is needed by all bids for constraint formulation.
    bids_plus_energy_data = pd.merge(capacity_bids, max_energy, how='left', on=['DUID'], sort=False)

    return bids_plus_energy_data


def add_min_unit_energy(capacity_bids, unit_solution):
    # The unit solution file provides availability data, ramp rate data and initial condition output data. These are
    # combined to give each unit a new availability number.
    # Combine the initial dispatch of the unit and its ramp rate to give the max dispatch due to ramping constraints.
    # TODO: Remove card code of 5 min dispatch.
    unit_solution['MINENERGY'] = unit_solution['INITIALMW'] - unit_solution['RAMPDOWNRATE'] / 12
    # Ramp down constraints don't apply to plants in dispatch mode 1.
    # unit_solution['MINENERGY'] = np.where(unit_solution['DISPATCHMODE']==1, 0, unit_solution['MINENERGY'])
    min_energy = unit_solution.loc[:, ('DUID', 'MINENERGY')]

    # Map the max energy availability given by each generator to all the bids given by that generator. This information
    # is needed by all bids for constraint formulation.
    bids_plus_energy_data = pd.merge(capacity_bids, min_energy, how='left', on=['DUID'], sort=False)

    return bids_plus_energy_data


def rationalise_max_energy_constraint(capacity_bids):
    # Reset the max energy where it is lower than the min energy.
    capacity_bids['MAXENERGY'] = np.where(capacity_bids['MAXENERGY'] < capacity_bids['MINENERGY'],
                                          capacity_bids['MINENERGY'], capacity_bids['MAXENERGY'])

    return capacity_bids


def remove_energy_bids_with_max_energy_zero(gen_bidding_data):
    # Keep the bids if they have max energy greater than zero, or if they are FCAS bids.
    gen_bidding_data = gen_bidding_data[(gen_bidding_data['MAXENERGY'] > 0.01) |
                                        (gen_bidding_data['BIDTYPE'] != 'ENERGY')].copy()
    return gen_bidding_data


def remove_fcas_bids_with_max_avail_zero(gen_bidding_data):
    # Keep the bids if they have max energy greater than zero, or if they are FCAS bids.
    gen_bidding_data = gen_bidding_data[(gen_bidding_data['MAXAVAIL'] > 0.01) |
                                        (gen_bidding_data['BIDTYPE'] == 'ENERGY')].copy()
    return gen_bidding_data


def fcas_trapezium_scaling_on_telemetered_raise_reg_enablement(bids_and_data):
    bids_and_data['ENABLEMENTMAXUNSCALED'] = bids_and_data['ENABLEMENTMAX']
    bids_and_data['ENABLEMENTMINUNSCALED'] = bids_and_data['ENABLEMENTMIN']
    bids_and_data['ENABLEMENTMAX'], bids_and_data['HIGHBREAKPOINT'] = \
        scale_upper_slope_based_on_telemetered_data(bids_and_data['ENABLEMENTMAXUNSCALED'],
                                                    bids_and_data['RAISEREGENABLEMENTMAX'],
                                                    bids_and_data['HIGHBREAKPOINT'],
                                                    bids_and_data['BIDTYPE'],
                                                    'RAISEREG')

    bids_and_data['ENABLEMENTMIN'], bids_and_data['LOWBREAKPOINT'] = \
        scale_lower_slope_based_on_telemetered_data(bids_and_data['ENABLEMENTMINUNSCALED'],
                                                    bids_and_data['RAISEREGENABLEMENTMIN'],
                                                    bids_and_data['LOWBREAKPOINT'],
                                                    bids_and_data['BIDTYPE'],
                                                    'RAISEREG')
    bids_and_data = bids_and_data.drop(['ENABLEMENTMAXUNSCALED', 'ENABLEMENTMINUNSCALED'], axis=1)
    return bids_and_data


def fcas_trapezium_scaling_on_telemetered_lower_reg_enablement(bids_and_data):
    bids_and_data['ENABLEMENTMAXUNSCALED'] = bids_and_data['ENABLEMENTMAX']
    bids_and_data['ENABLEMENTMINUNSCALED'] = bids_and_data['ENABLEMENTMIN']
    bids_and_data['ENABLEMENTMAX'], bids_and_data['HIGHBREAKPOINT'] = \
        scale_upper_slope_based_on_telemetered_data(bids_and_data['ENABLEMENTMAXUNSCALED'],
                                                    bids_and_data['LOWERREGENABLEMENTMAX'],
                                                    bids_and_data['HIGHBREAKPOINT'],
                                                    bids_and_data['BIDTYPE'],
                                                    'LOWERREG')

    bids_and_data['ENABLEMENTMIN'], bids_and_data['LOWBREAKPOINT'] = \
        scale_lower_slope_based_on_telemetered_data(bids_and_data['ENABLEMENTMINUNSCALED'],
                                                    bids_and_data['LOWERREGENABLEMENTMIN'],
                                                    bids_and_data['LOWBREAKPOINT'],
                                                    bids_and_data['BIDTYPE'],
                                                    'LOWERREG')
    bids_and_data = bids_and_data.drop(['ENABLEMENTMAXUNSCALED', 'ENABLEMENTMINUNSCALED'], axis=1)
    return bids_and_data


def fcas_trapezium_scaling_on_ramp_up_rate(bids_and_data):
    bids_and_data['MAXAVAILUNSCALED'] = bids_and_data['MAXAVAIL']
    bids_and_data['MAXAVAIL'] = scale_max_available_based_on_ramp_rate(bids_and_data['MAXAVAILUNSCALED'],
                                                                       bids_and_data['RAMPUPRATE'] / 12,
                                                                       bids_and_data['BIDTYPE'],
                                                                       'RAISEREG')

    bids_and_data['LOWBREAKPOINT'] = scale_low_break_point(bids_and_data['RAMPUPRATE'] / 12,
                                                           bids_and_data['MAXAVAILUNSCALED'],
                                                           bids_and_data['ENABLEMENTMIN'],
                                                           bids_and_data['LOWBREAKPOINT'],
                                                           bids_and_data['BIDTYPE'],
                                                           'RAISEREG')

    bids_and_data['HIGHBREAKPOINT'] = scale_high_break_point(bids_and_data['RAMPUPRATE'] / 12,
                                                             bids_and_data['MAXAVAILUNSCALED'],
                                                             bids_and_data['ENABLEMENTMAX'],
                                                             bids_and_data['HIGHBREAKPOINT'],
                                                             bids_and_data['BIDTYPE'],
                                                             'RAISEREG')
    bids_and_data = bids_and_data.drop(['MAXAVAILUNSCALED'], axis=1)
    return bids_and_data


def fcas_trapezium_scaling_on_ramp_down_rate(bids_and_data):
    bids_and_data['MAXAVAILUNSCALED'] = bids_and_data['MAXAVAIL']
    bids_and_data['MAXAVAIL'] = scale_max_available_based_on_ramp_rate(bids_and_data['MAXAVAILUNSCALED'],
                                                                       bids_and_data['RAMPDOWNRATE'] / 12,
                                                                       bids_and_data['BIDTYPE'],
                                                                       'LOWERREG')

    bids_and_data['LOWBREAKPOINT'] = scale_low_break_point(bids_and_data['RAMPDOWNRATE'] / 12,
                                                           bids_and_data['MAXAVAILUNSCALED'],
                                                           bids_and_data['ENABLEMENTMIN'],
                                                           bids_and_data['LOWBREAKPOINT'],
                                                           bids_and_data['BIDTYPE'],
                                                           'LOWERREG')

    bids_and_data['HIGHBREAKPOINT'] = scale_high_break_point(bids_and_data['RAMPDOWNRATE'] / 12,
                                                             bids_and_data['MAXAVAILUNSCALED'],
                                                             bids_and_data['ENABLEMENTMAX'],
                                                             bids_and_data['HIGHBREAKPOINT'],
                                                             bids_and_data['BIDTYPE'],
                                                             'LOWERREG')
    bids_and_data = bids_and_data.drop(['MAXAVAILUNSCALED'], axis=1)
    return bids_and_data


def scale_max_available_based_on_ramp_rate_s(max_available, ramp_rate, bid_type, bid_type_to_scale):
    if (max_available > ramp_rate) & (bid_type == bid_type_to_scale):
        new_max = ramp_rate
    else:
        new_max = max_available
    return new_max


scale_max_available_based_on_ramp_rate = np.vectorize(scale_max_available_based_on_ramp_rate_s)


def scale_upper_slope_based_on_telemetered_data_s(enablement_max_as_bid, enablement_max_as_telemetered,
                                                  high_break_point,
                                                  bid_type, bid_type_to_scale):
    if ((bid_type == bid_type_to_scale) and (enablement_max_as_telemetered < enablement_max_as_bid) and
            (enablement_max_as_telemetered > 0.0)):
        enablement_max = enablement_max_as_telemetered
        high_break_point = high_break_point - (enablement_max_as_bid - enablement_max_as_telemetered)
    else:
        enablement_max = enablement_max_as_bid
    return enablement_max, high_break_point


scale_upper_slope_based_on_telemetered_data = np.vectorize(scale_upper_slope_based_on_telemetered_data_s)


def scale_lower_slope_based_on_telemetered_data_s(enablement_min_as_bid, enablement_min_as_telemetered, low_break_point,
                                                  bid_type, bid_type_to_scale):
    if ((bid_type == bid_type_to_scale) and (enablement_min_as_telemetered > enablement_min_as_bid) and
            (enablement_min_as_telemetered > 0.0)):
        enablement_min = enablement_min_as_telemetered
        low_break_point = low_break_point + (enablement_min_as_telemetered - enablement_min_as_bid)
    else:
        enablement_min = enablement_min_as_bid
    return enablement_min, low_break_point


scale_lower_slope_based_on_telemetered_data = np.vectorize(scale_lower_slope_based_on_telemetered_data_s)


def scale_low_break_point_s(ramp_rate, max_avail, enable_min, low_break, reg_type, type_to_scale):
    if (max_avail > ramp_rate) & (reg_type == type_to_scale) & (enable_min < low_break):
        new_break = low_break - (((low_break - enable_min) * (max_avail - ramp_rate)) / max_avail)
    else:
        new_break = low_break
    return new_break


scale_low_break_point = np.vectorize(scale_low_break_point_s)


def scale_high_break_point_s(ramp_rate, max_avail, enable_max, high_break, reg_type, type_to_scale):
    if (max_avail > ramp_rate) & (reg_type == type_to_scale) & (enable_max > high_break):
        new_break = high_break + (((enable_max - high_break) * (ramp_rate - max_avail)) / (-1 * max_avail))
    else:
        new_break = high_break
    return new_break


scale_high_break_point = np.vectorize(scale_high_break_point_s)


def apply_fcas_enablement_criteria(capacity_bids, initial_conditions):
    initial_mw = initial_conditions.loc[:, ('DUID', 'INITIALMW', 'AGCSTATUS')]

    bids_and_initial = pd.merge(capacity_bids, initial_mw)

    available_for_fcas = bids_and_initial[
        ((bids_and_initial['ENABLEMENTMIN'] <= bids_and_initial['INITIALMW']) &
         (bids_and_initial['INITIALMW'] <= bids_and_initial['ENABLEMENTMAX']))
        | (bids_and_initial['BIDTYPE'] == 'ENERGY')]

    available_for_fcas = available_for_fcas[
        (available_for_fcas['ENABLEMENTMIN'] <= available_for_fcas['MAXENERGY']) |
        (available_for_fcas['BIDTYPE'] == 'ENERGY') | available_for_fcas['MAXENERGY'].isnull()]

    available_for_fcas = available_for_fcas[(available_for_fcas['AGCSTATUS'] == 1)
                                            | ((available_for_fcas['BIDTYPE'] != 'LOWERREG')
                                               & (available_for_fcas['BIDTYPE'] != 'RAISEREG'))]

    available_for_fcas = available_for_fcas.drop(['INITIALMW', 'AGCSTATUS'], axis=1)

    return available_for_fcas


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


def over_ride_max_and_min_energy_for_fast_start_units(current_max_and_min,
                                                      dispatch_inflexibility_profiles):
    """
    Calculates the constraints on unit dispatch due to fast start inflexibility profiles.

    :param current_max_and_min: DataFrame
        DUID: str
            The dispatch unit the row pertains to.
        BIDTYPE: str
            The service the row pertains to.
        MAXENERGY: float
            The maximum level at which the unit can be dispatched.
        MINENERGY: float
            The minimum level at which the unit can be dispatched.
        DISPATCHMODE: int
            As determined by AEMO in historical dispatch.
        TOTALCLEARED: int
            As determined by AEMO in historical dispatch.
        Other columns may be present but are not used or changed by this function.
    :param dispatch_inflexibility_profiles: DataFrame
        DUID: str
            The dispatch unit the row pertains to.
        MINIMUMLOAD: float
            The minimum loading allowed for the unit during the T3 period of the dispatch inflexibility profile.
        Other columns may be present but are not used or changed by this function.
    :return: DataFrame
        DUID: str
            The dispatch unit the row pertains to.
        BIDTYPE: str
            The service the row pertains to.
        MAXENERGY: float
            The maximum level at which the unit can be dispatched.
        MINENERGY: float
            The minimum level at which the unit can be dispatched.
        Other columns may be present but are not used or changed by this function.
    """

    # Combine data frames together.
    current_max_and_min = pd.merge(current_max_and_min,
                                   dispatch_inflexibility_profiles.loc[:,
                                   ['DUID', 'MINIMUMLOAD']], 'left', 'DUID')
    # current_max_and_min = pd.merge(current_max_and_min,
    #                                initial_cons.loc[:,
    #                                ['DUID', 'TOTALCLEARED', 'INITIALMW', 'LASTTOTALCLEARED']], 'left', 'DUID')

    # Override the MINENERGY con for unit not in dispatch mode 0
    current_max_and_min['MINENERGY'] = np.where(current_max_and_min['DISPATCHMODE'] == 1,
                                                current_max_and_min['TOTALCLEARED'],
                                                current_max_and_min['MINENERGY'])
    current_max_and_min['MINENERGY'] = np.where(current_max_and_min['DISPATCHMODE'] == 2,
                                                current_max_and_min['TOTALCLEARED'],
                                                current_max_and_min['MINENERGY'])
    current_max_and_min['MINENERGY'] = np.where(current_max_and_min['DISPATCHMODE'] == 3,
                                                current_max_and_min['MINIMUMLOAD'], current_max_and_min['MINENERGY'])

    # Override the MAXENERGY con for units in dispatch mode 1 and 2
    current_max_and_min['MAXENERGY'] = np.where(current_max_and_min['DISPATCHMODE'] == 1,
                                                current_max_and_min['TOTALCLEARED'],
                                                current_max_and_min['MAXENERGY'])
    current_max_and_min['MAXENERGY'] = np.where(current_max_and_min['DISPATCHMODE'] == 2,
                                                current_max_and_min['TOTALCLEARED'],
                                                current_max_and_min['MAXENERGY'])

    # Change starting max limit if unit below minimum load
    current_max_and_min['MAXENERGY'] = np.where((current_max_and_min['TOTALCLEARED'] > current_max_and_min['MAXENERGY'])
                                                & (current_max_and_min['DISPATCHMODE'].isin([3, 4]))
                                                & (current_max_and_min['LASTTOTALCLEARED'] >
                                                current_max_and_min['INITIALMW']),
                                                current_max_and_min['LASTTOTALCLEARED'] +
                                                current_max_and_min['RAMPUPRATE'] / 12,
                                                current_max_and_min['MAXENERGY'])

    current_max_and_min['MAXENERGY'] = np.where((current_max_and_min['TOTALCLEARED'] > current_max_and_min['MAXENERGY'])
                                                & (current_max_and_min['DISPATCHMODE'].isin([3, 4])),
                                                current_max_and_min['TOTALCLEARED'],
                                                current_max_and_min['MAXENERGY'])

    current_max_and_min['MINENERGY'] = np.where((current_max_and_min['MINENERGY'] > current_max_and_min['MINIMUMLOAD'])
                                                & (current_max_and_min['DISPATCHMODE'].isin([3, 4])),
                                                current_max_and_min['MINIMUMLOAD'],
                                                current_max_and_min['MINENERGY'])

    # Remove extra columns from max and min data frame.
    cols_discard = ['MINIMUMLOAD']
    cols_to_keep = [col for col in current_max_and_min.columns if col not in cols_discard]
    current_max_and_min = current_max_and_min.loc[:, cols_to_keep]
    return current_max_and_min
