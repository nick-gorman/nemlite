import os
import pandas as pd
import numpy as np
from nemosis import data_fetch_methods
from nemosis import defaults
from nemosis import query_wrapers
from datetime import datetime, timedelta
from nemlite import nemlite_defaults
from nemlite import engine


def actual_inputs_replicator(start_time, end_time, raw_aemo_data_folder, filtered_data_folder, run_pre_filter=True):
    # Create a datetime generator for each process that iterates over each 5 minutes interval in the
    # start to end time window.
    delta = timedelta(minutes=5)
    start_time_obj = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
    end_time_obj = datetime.strptime(end_time, '%Y/%m/%d %H:%M:%S')
    date_times = datetime_dispatch_sequence(start_time_obj, end_time_obj, delta)
    if run_pre_filter:
        pre_filter_all_tables(date_times, raw_aemo_data_folder, filtered_data_folder)
    return actual_inputs_generator(date_times, filtered_data_folder)


def pre_filter_all_tables(date_times, raw_aemo_data_folder, filtered_data_folder):
    # Pre filter dispatch constraint first as some tables are filtered based of its filtered files.
    pre_filter('DISPATCHCONSTRAINT', date_times, raw_aemo_data_folder, filtered_data_folder)
    # Pre filter the rest of the tables.
    for table in nemlite_defaults.parent_tables:
        if table != 'DISPATCHCONSTRAINT':
            pre_filter(table, date_times, raw_aemo_data_folder, filtered_data_folder)


def actual_inputs_generator(date_time_iterator, filtered_data_folder):
    for date_time in date_time_iterator:
        yield load_from_datetime(date_time, filtered_data_folder)


def load_from_datetime(date_time, filtered_data_folder):
    datetime_name = date_time.replace('/', '')
    datetime_name = datetime_name.replace(" ", "_")
    datetime_name = datetime_name.replace(":", "")
    input_tables = load_and_merge(datetime_name, filtered_data_folder)

    return input_tables['generator_information'], input_tables['capacity_bids'], input_tables['initial_conditions'], \
           input_tables['interconnectors'], input_tables['demand'], input_tables['price_bids'], \
           input_tables['interconnector_segments'], input_tables['connection_point_constraints'], \
           input_tables['interconnector_constraints'], input_tables['constraint_data'], \
           input_tables['region_constraints'], date_time, input_tables['interconnector_dynamic_loss_coefficients'], \
           input_tables['market_interconnectors'], input_tables['market_interconnector_price_bids'], \
           input_tables['market_interconnector_capacity_bids'], input_tables['price_cap_and_floor']


def run_datetime(date_time, filtered_data_folder):

    gen_info_raw, capacity_bids_raw, initial_conditions, inter_direct_raw, region_req_raw, price_bids_raw, \
    inter_seg_definitions, con_point_constraints, inter_gen_constraints, gen_con_data, region_constraints, \
    timestamp, inter_demand_coefficients, mnsp_inter, mnsp_price_bids, mnsp_capacity_bids, \
    market_cap_and_floor = load_from_datetime(date_time, filtered_data_folder)

    #try:
    nemlite_results, dispatches, inter_flows = engine.run(gen_info_raw, capacity_bids_raw,
                                                          initial_conditions,
                                                          inter_direct_raw, region_req_raw, price_bids_raw,
                                                          inter_seg_definitions, con_point_constraints,
                                                          inter_gen_constraints, gen_con_data,
                                                          region_constraints, inter_demand_coefficients,
                                                          mnsp_inter, mnsp_price_bids, mnsp_capacity_bids,
                                                          market_cap_and_floor)
    nemlite_results['DateTime'] = timestamp
    dispatches['BASERUN']['DateTime'] = timestamp
    inter_flows['BASERUN']['DateTime'] = timestamp
    #except:
        #print('Dispatch failed for {}'.format(timestamp))
        #nemlite_results = pd.DataFrame()
        #dispatches = {'BASERUN': pd.DataFrame()}
        #inter_flows = {'BASERUN': pd.DataFrame()}

    return nemlite_results, dispatches, inter_flows


def load_and_merge(date_time_name, filtered_data):
    save_location_formated = filtered_data + '/{}_{}.csv'
    child_tables = {}
    for child_table_name, parent_tables_names in nemlite_defaults.child_parent_map.items():
        parent_tables = []
        for name in parent_tables_names:
            save_name_and_location = save_location_formated.format(name, date_time_name)
            parent_tables.append(pd.read_csv(save_name_and_location))
        if len(parent_tables) == 2:
            child_tables[child_table_name] = pd.merge(parent_tables[0], parent_tables[1], 'inner',
                                                      nemlite_defaults.parent_merge_cols[child_table_name])
        elif len(parent_tables) == 1:
            child_tables[child_table_name] = parent_tables[0]
        else:
            print('Parent table left unmerged')

    return child_tables


def pre_filter(table, date_time_sequence, raw_data_location, filtered_data):
    if 'INTERVENTION' in defaults.table_columns[table]:
        filter_cols = ['INTERVENTION']
        filter_values = [['0']]
    else:
        filter_cols = None
        filter_values = None

    sorted_date_times = sorted(date_time_sequence)
    start_time = sorted_date_times[0]
    start_time_obj = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S') - timedelta(minutes=5)
    start_time = start_time_obj.isoformat().replace('T', ' ').replace('-', '/')
    end_time = sorted_date_times[-1]
    all_data = data_fetch_methods.method_map[table](start_time, end_time, table, raw_data_location,
                                                    filter_cols=filter_cols, filter_values=filter_values)

    save_location_formated = filtered_data + '/{}_{}.csv'
    for date_time in sorted_date_times:
        datetime_name = date_time.replace('/', '')
        datetime_name = datetime_name.replace(" ", "_")
        datetime_name = datetime_name.replace(":", "")
        date_time_specific_data = filter_map[table](data=all_data, save_location_formated=save_location_formated,
                                                    date_time=date_time, datetime_name=datetime_name, table_name=table)
        if nemlite_defaults.required_cols[table] is not None:
            date_time_specific_data = date_time_specific_data.loc[:, nemlite_defaults.required_cols[table]]
        date_time_specific_data.to_csv(save_location_formated.format(table, datetime_name), sep=',', index=False,
                                       date_format='%Y/%m/%d %H:%M:%S')


def constraint_filter(**kwargs):
    data, date_time, table_name, save_location_formated, datetime_name = \
        kwargs['data'], kwargs['date_time'], kwargs['table_name'], kwargs['save_location_formated'], \
        kwargs['datetime_name']
    dispatch_cons_filename = save_location_formated.format('DISPATCHCONSTRAINT', datetime_name)
    dispatched_constraints = pd.read_csv(dispatch_cons_filename, dtype=str)
    merge_cols = ('GENCONID', 'EFFECTIVEDATE', 'VERSIONNO')
    dispatched_constraints = dispatched_constraints.loc[:, merge_cols]
    dispatched_constraints['EFFECTIVEDATE'] = pd.to_datetime(dispatched_constraints['EFFECTIVEDATE'])
    filtered_constraints = pd.merge(dispatched_constraints, data, 'inner',
                                    ['GENCONID', 'EFFECTIVEDATE', 'VERSIONNO'])
    filtered_constraints = filtered_constraints.drop_duplicates(defaults.table_primary_keys[table_name])
    return filtered_constraints


def settlement_date_filter(**kwargs):
    data, date_time, table_name = kwargs['data'], kwargs['date_time'], kwargs['table_name']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['SETTLEMENTDATE'] == date_time)]
    if table_name == 'DISPATCHCONSTRAINT':
        data = data.loc[:, ('CONSTRAINTID', 'RHS', 'GENCONID_VERSIONNO', 'GENCONID_EFFECTIVEDATE')]
        data.columns = ['GENCONID', 'RHS', 'VERSIONNO', 'EFFECTIVEDATE']
    return data


def interval_datetime_filter(**kwargs):
    data, date_time, table_name = kwargs['data'], kwargs['date_time'], kwargs['table_name']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['INTERVAL_DATETIME'] = pd.to_datetime(data['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['INTERVAL_DATETIME'] == date_time)]
    if table_name == 'DISPATCHCONSTRAINT':
        data = data.loc[:, nemlite_defaults.required_cols['DISPATCHCONSTRAINT']]
        data.columns = ['GENCONID', 'RHS', 'VERSIONNO', 'EFFECTIVEDATE']

    return data


def half_hour_periods(**kwargs):
    data, date_time, table_name = kwargs['data'], kwargs['date_time'], kwargs['table_name']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['SETTLEMENTDATE'] > date_time) & (data['SETTLEMENTDATE'] - timedelta(minutes=30) <= date_time)]
    group_cols = defaults.effective_date_group_col[table_name]
    data = most_recent_version(data, table_name, group_cols)
    return data


def settlement_just_date_filter(**kwargs):
    data, date_time = kwargs['data'], kwargs['date_time']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    date_time = date_time - timedelta(hours=4, seconds=1)
    date = date_time.replace(hour=0, minute=0, second=0)
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['SETTLEMENTDATE'] == date)]
    return data


def settlement_just_date_and_version_filter(**kwargs):
    data, date_time, table_name = kwargs['data'], kwargs['date_time'], kwargs['table_name']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    date_time = date_time - timedelta(hours=4, seconds=1)
    date = date_time.replace(hour=0, minute=0, second=0)
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['SETTLEMENTDATE'] == date)]
    group_cols = defaults.effective_date_group_col[table_name]
    data = most_recent_version(data, table_name, group_cols)
    return data


def effective_date_filter(**kwargs):
    data, date_time, table_name = kwargs['data'], kwargs['date_time'], kwargs['table_name']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['EFFECTIVEDATE'] = pd.to_datetime(data['EFFECTIVEDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[data['EFFECTIVEDATE'] <= date_time]
    group_cols = defaults.effective_date_group_col[table_name]
    data = query_wrapers.most_recent_records_before_start_time(data, date_time, table_name)
    return data


def effective_date_and_version_filter(**kwargs):
    data, date_time, table_name = kwargs['data'], kwargs['date_time'], kwargs['table_name']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['EFFECTIVEDATE'] = pd.to_datetime(data['EFFECTIVEDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[data['EFFECTIVEDATE'] <= date_time]
    group_cols = defaults.effective_date_group_col[table_name]
    data = query_wrapers.most_recent_records_before_start_time(data, date_time, table_name)
    data = most_recent_version(data, table_name, group_cols)
    return data


def effective_date_and_version_filter_for_inter_seg(**kwargs):
    data, date_time, table_name = kwargs['data'], kwargs['date_time'], kwargs['table_name']
    data_orginal = data.copy()
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['EFFECTIVEDATE'] = pd.to_datetime(data['EFFECTIVEDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[data['EFFECTIVEDATE'] <= date_time]
    group_cols = defaults.effective_date_group_col[table_name]
    data = query_wrapers.most_recent_records_before_start_time(data, date_time, table_name)
    data = most_recent_version(data, table_name, group_cols)
    data = data.drop_duplicates(['EFFECTIVEDATE', 'INTERCONNECTORID', 'VERSIONNO', 'LOSSSEGMENT'])
    # data = data.loc[:, ('EFFECTIVEDATE', 'INTERCONNECTORID', 'VERSIONNO')]
    # data = pd.merge(data, data_orginal, 'left', ['EFFECTIVEDATE', 'INTERCONNECTORID', 'VERSIONNO'])
    return data


def start_date_end_date_filter(**kwargs):
    data, date_time = kwargs['data'], kwargs['date_time']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['START_DATE'] = pd.to_datetime(data['START_DATE'], format='%Y/%m/%d %H:%M:%S')
    data['END_DATE'] = pd.to_datetime(data['END_DATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['START_DATE'] <= date_time) & (data['END_DATE'] > date_time)]
    data = data.sort_values('END_DATE').groupby(['DUID', 'START_DATE'], as_index=False).first()
    data = data.sort_values('START_DATE').groupby(['DUID'], as_index=False).first()
    # data = query_wrapers.most_recent_records_before_start_time(data, date_time, table_name)
    return data


def no_filter(**kwargs):
    return kwargs['data']


def filter_dispatch_load(**kwargs):
    """
    Calculates the time since a fast start unit received instructions to commit. Filters to single dispatch interval.

    :keyword data: DataFrame
        SETTLEMENTDATE: pandas.Timestamp
            The dispatch interval the row pertains to.
        DUID: str
            The dispatch unit the row pertains to.
        DISPATCHMODE: str
            The stage of dispatch inflexibility profile the unit is in, can be 0, 1, 2, 3, 4.
        Other columns will be present but are not used by this function.
    :keyword date_time:
    :return: data: DataFrame
        SETTLEMENTDATE: pandas.Timestamp
            The dispatch interval the row pertains to.
        DUID: str
            The dispatch unit the row pertains to.
        DISPATCHMODE: str
            The stage of dispatch inflexibility profile the unit is in, can be 0, 1, 2, 3, 4.
        TIMESINCECOMMITMENT: float
            The time in minutes since the start of the dispatch interval in which the unit was committed. If the unit is
            in dispatch mode 0 then this number is 0, but has not meaning.
        Other columns will be present but are not used by this function.
    """
    data, date_time = kwargs['data'], kwargs['date_time']
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')

    # Split data based on processing needed.
    units_in_dispatch_mode_0 = data[(data['DISPATCHMODE'] == '0') & (data['SETTLEMENTDATE'] == date_time)]['DUID']
    units_in_dispatch_mode_0 = data[data['DUID'].isin(units_in_dispatch_mode_0)]
    units_not_in_dispatch_mode_0 = data[(data['DISPATCHMODE'] != '0') & (data['SETTLEMENTDATE'] == date_time)]['DUID']
    units_not_in_dispatch_mode_0 = data[data['DUID'].isin(units_not_in_dispatch_mode_0)]

    # For units in dispatch mode 0 just set TIMESINCECOMMITMMENT to 0.
    units_in_dispatch_mode_0['TIMESINCECOMMITMENT'] = 0
    units_in_dispatch_mode_0 = units_in_dispatch_mode_0[units_in_dispatch_mode_0['SETTLEMENTDATE'] == date_time]

    # For units not in dispatch mode 0, calculate the time since they last were in mode 0. Limiting search to a maximum
    # of 60 minutes.
    commitment_time_by_unit = units_not_in_dispatch_mode_0[units_not_in_dispatch_mode_0['DISPATCHMODE'] == '0']
    commitment_time_by_unit = commitment_time_by_unit[
        commitment_time_by_unit['SETTLEMENTDATE'] >= date_time - timedelta(minutes=60)]
    commitment_time_by_unit = commitment_time_by_unit.groupby('DUID', as_index=False).\
        aggregate({'SETTLEMENTDATE': 'max'})
    commitment_time_by_unit.columns = ['DUID', 'COMMITMENTTIME']
    units_not_in_dispatch_mode_0 = units_not_in_dispatch_mode_0[
        units_not_in_dispatch_mode_0['SETTLEMENTDATE'] == date_time]
    units_not_in_dispatch_mode_0 = pd.merge(units_not_in_dispatch_mode_0, commitment_time_by_unit, 'left', 'DUID')
    units_not_in_dispatch_mode_0['TIMESINCECOMMITMENT'] = units_not_in_dispatch_mode_0['SETTLEMENTDATE'] - \
        units_not_in_dispatch_mode_0['COMMITMENTTIME']
    units_not_in_dispatch_mode_0['TIMESINCECOMMITMENT'] = np.where(
        units_not_in_dispatch_mode_0['TIMESINCECOMMITMENT'].isnull(), timedelta(minutes=60),
        units_not_in_dispatch_mode_0['TIMESINCECOMMITMENT'])
    units_not_in_dispatch_mode_0['TIMESINCECOMMITMENT'] = \
        pd.to_timedelta(units_not_in_dispatch_mode_0['TIMESINCECOMMITMENT']).dt.total_seconds().div(60).astype(int)
    cols_to_keep = [col for col in units_not_in_dispatch_mode_0 if col != 'COMMITMENTTIME']
    units_not_in_dispatch_mode_0 = units_not_in_dispatch_mode_0.loc[:, cols_to_keep]

    # Combined data
    data = pd.concat([units_in_dispatch_mode_0, units_not_in_dispatch_mode_0])
    return data


def most_recent_version(data, table_name, group_cols):
    data = data.sort_values('VERSIONNO')
    if len(group_cols) > 0:
        data_most_recent_v = data.groupby(group_cols, as_index=False).last()
    else:
        data_most_recent_v = data.tail(1)
    group_cols = group_cols + ['VERSIONNO']
    data_most_recent_v = pd.merge(data_most_recent_v.loc[:, group_cols], data, 'inner', group_cols)
    return data_most_recent_v


def datetime_dispatch_sequence(start_time, end_time, delta):
    # Generator to produce time stamps at set intervals. Requires a datetime object as an input, but outputs
    # the date time as string formatted as 'YYYY/MM/DD HH:MM:SS'.
    date_times = []
    curr = start_time + delta
    while curr <= end_time:
        # Change the datetime object to a timestamp and modify it format by replacing characters.
        date_times.append(curr.isoformat().replace('T', ' ').replace('-', '/'))
        curr += delta
    return date_times


def derive_group_cols(table_name, date_col, also_exclude=None):
    exclude_from_group_cols = [date_col, 'VERSIONNO']
    if also_exclude is not None:
        exclude_from_group_cols.append(also_exclude)
    group_cols = [column for column in defaults.table_primary_keys[table_name] if column not in exclude_from_group_cols]
    return group_cols


filter_map = {'SPDCONNECTIONPOINTCONSTRAINT': constraint_filter,
              'GENCONDATA': constraint_filter,
              'SPDINTERCONNECTORCONSTRAINT': constraint_filter,
              'BIDPEROFFER_D': interval_datetime_filter,
              'DISPATCHINTERCONNECTORRES': settlement_date_filter,
              'INTERCONNECTOR': no_filter,
              'INTERCONNECTORCONSTRAINT': effective_date_and_version_filter,
              'MNSP_INTERCONNECTOR': effective_date_and_version_filter,
              'DISPATCHPRICE': settlement_date_filter,
              'DUDETAILSUMMARY': start_date_end_date_filter,
              'DISPATCHCONSTRAINT': settlement_date_filter,
              'SPDREGIONCONSTRAINT': constraint_filter,
              'BIDDAYOFFER_D': settlement_just_date_filter,
              'MNSP_DAYOFFER': settlement_just_date_and_version_filter,
              'MNSP_PEROFFER': half_hour_periods,
              'DISPATCHLOAD': settlement_date_filter,
              'LOSSMODEL': effective_date_and_version_filter_for_inter_seg,
              'LOSSFACTORMODEL': effective_date_and_version_filter,
              'DISPATCHREGIONSUM': settlement_date_filter,
              'MARKET_PRICE_THRESHOLDS': effective_date_and_version_filter}
