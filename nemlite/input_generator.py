import pandas as pd
from nemosis import data_fetch_methods
from nemosis import defaults
from nemosis import query_wrapers
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from nemlite import nemlite_defaults


def actual_inputs_replicator(start_time, end_time, raw_aemo_data_folder, filtered_data_folder, run_pre_filter=True):

    # Create a datetime generator for generator for each process that iterates over each 5 minutes interval in the
    # start to end time window.
    delta = timedelta(minutes=5)
    start_time_obj = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
    end_time_obj = datetime.strptime(end_time, '%Y/%m/%d %H:%M:%S')

    date_times_generator_2 = datetime_dispatch_sequence(start_time_obj, end_time_obj, delta)
    if run_pre_filter:
        with Parallel(n_jobs=1) as pool:
            # Pre filter dispatch constraint first as some tables are filtered based of its filtered files.
            run_pf('DISPATCHCONSTRAINT', start_time, end_time, raw_aemo_data_folder, filtered_data_folder)
            # Pre filter the rest of the tables.
            pool(delayed(run_pf)(table, start_time, end_time, raw_aemo_data_folder, filtered_data_folder)
                 for table in nemlite_defaults.parent_tables if table != 'DISPATCHCONSTRAINT')

    for date_time in date_times_generator_2:
        datetime_name = date_time.replace('/', '')
        datetime_name = datetime_name.replace(" ", "_")
        datetime_name = datetime_name.replace(":", "")
        input_tables = load_and_merge(datetime_name, filtered_data_folder)

        yield input_tables['generator_information'], input_tables['capacity_bids'], input_tables['initial_conditions'],\
            input_tables['interconnectors'], input_tables['demand'], input_tables['price_bids'], \
            input_tables['interconnector_segments'], input_tables['connection_point_constraints'], \
            input_tables['interconnector_constraints'], input_tables['constraint_data'], \
            input_tables['region_constraints'], date_time, input_tables['interconnector_dynamic_loss_coefficients'],\
            input_tables['market_interconnectors'], input_tables['market_interconnector_price_bids'], \
            input_tables['market_interconnector_capacity_bids'], input_tables['price_cap_and_floor']


def run_pf(table, start_time, end_time, raw_aemo_data_folder, filtered_data_folder):
    start_time_obj = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
    end_time_obj = datetime.strptime(end_time, '%Y/%m/%d %H:%M:%S')
    delta = timedelta(minutes=5)
    date_times_generator_1 = datetime_dispatch_sequence(start_time_obj, end_time_obj, delta)
    pre_filter(start_time, end_time, table, date_times_generator_1, raw_aemo_data_folder, filtered_data_folder)


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

    child_tables['interconnector_segments'] = \
        child_tables['interconnector_segments'][~child_tables['interconnector_segments']['INTERCONNECTORID'].isin(child_tables['interconnectors'][child_tables['interconnectors']['ICTYPE']
                                                                                                                                                  == 'MNSP']['INTERCONNECTORID'])]

    return child_tables


def pre_filter(start_time, end_time, table, date_time_sequence, raw_data_location, filtered_data):

    if 'INTERVENTION' in defaults.table_columns[table]:
        filter_cols = ['INTERVENTION']
        filter_values = [['0']]
    else:
        filter_cols = None
        filter_values = None

    all_data = data_fetch_methods.method_map[table](start_time, end_time, table, raw_data_location,
                                                    filter_cols=filter_cols, filter_values=filter_values)

    save_location_formated = filtered_data + '/{}_{}.csv'
    for date_time in date_time_sequence:
        datetime_name = date_time.replace('/', '')
        datetime_name = datetime_name.replace(" ", "_")
        datetime_name = datetime_name.replace(":", "")
        date_time_specific_data = filter_map[table](all_data, save_location_formated, date_time, datetime_name, table)
        if nemlite_defaults.required_cols[table] is not None:
            date_time_specific_data = date_time_specific_data.loc[:, nemlite_defaults.required_cols[table]]
        date_time_specific_data.to_csv(save_location_formated.format(table, datetime_name), sep=',', index=False,
                                       date_format='%Y/%m/%d %H:%M:%S')


def constraint_filter(constraint_data, save_location_formated, date_time, datetime_name, table_name):
    dispatch_cons_filename = save_location_formated.format('DISPATCHCONSTRAINT', datetime_name)
    dispatched_constraints = pd.read_csv(dispatch_cons_filename, dtype=str)
    merge_cols = ('GENCONID', 'EFFECTIVEDATE', 'VERSIONNO')
    dispatched_constraints = dispatched_constraints.loc[:, merge_cols]
    dispatched_constraints['EFFECTIVEDATE'] = pd.to_datetime(dispatched_constraints['EFFECTIVEDATE'])
    filtered_constraints = pd.merge(dispatched_constraints, constraint_data, 'inner',
                                    ['GENCONID', 'EFFECTIVEDATE', 'VERSIONNO'])
    filtered_constraints = filtered_constraints.drop_duplicates(defaults.table_primary_keys[table_name])

    return filtered_constraints


def settlement_date_filter(data, save_location_formated, date_time, datetime_name, table_name):
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['SETTLEMENTDATE'] == date_time)]
    if table_name == 'DISPATCHCONSTRAINT':
        data = data.loc[:, ('CONSTRAINTID', 'RHS', 'GENCONID_VERSIONNO', 'GENCONID_EFFECTIVEDATE')]
        data.columns = ['GENCONID', 'RHS', 'VERSIONNO', 'EFFECTIVEDATE']

    return data


def interval_datetime_filter(data, save_location_formated, date_time, datetime_name, table_name):
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['INTERVAL_DATETIME'] = pd.to_datetime(data['INTERVAL_DATETIME'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['INTERVAL_DATETIME'] == date_time)]
    if table_name == 'DISPATCHCONSTRAINT':
        data = data.loc[:, nemlite_defaults.required_cols['DISPATCHCONSTRAINT']]
        data.columns = ['GENCONID', 'RHS', 'VERSIONNO', 'EFFECTIVEDATE']

    return data


def half_hour_peroids(data, save_location_formated, date_time, datetime_name, table_name):
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['SETTLEMENTDATE'] > date_time) & (data['SETTLEMENTDATE'] - timedelta(minutes=30) <= date_time)]
    group_cols = defaults.effective_date_group_col[table_name]
    data = most_recent_version(data, table_name, group_cols)
    return data


def settlement_just_date_filter(data, save_location_formated, date_time, datetime_name, table_name):
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    date_time = date_time - timedelta(hours=4, seconds=1)
    date = date_time.replace(hour=0, minute=0, second=0)
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['SETTLEMENTDATE'] == date)]
    return data


def settlement_just_date_and_version_filter(data, save_location_formated, date_time, datetime_name, table_name):
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    date_time = date_time - timedelta(hours=4, seconds=1)
    date = date_time.replace(hour=0, minute=0, second=0)
    data['SETTLEMENTDATE'] = pd.to_datetime(data['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['SETTLEMENTDATE'] == date)]
    group_cols = defaults.effective_date_group_col[table_name]
    data = most_recent_version(data, table_name, group_cols)
    return data


def effective_date_filter(data, save_location_formated, date_time, datetime_name, table_name):
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['EFFECTIVEDATE'] = pd.to_datetime(data['EFFECTIVEDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[data['EFFECTIVEDATE'] <= date_time]
    data = query_wrapers.most_recent_records_before_start_time(data, date_time, table_name)
    return data


def effective_date_and_version_filter(data, save_location_formated, date_time, datetime_name, table_name):
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['EFFECTIVEDATE'] = pd.to_datetime(data['EFFECTIVEDATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[data['EFFECTIVEDATE'] <= date_time]
    group_cols = defaults.effective_date_group_col[table_name]
    data = query_wrapers.most_recent_records_before_start_time(data, date_time, table_name)
    data = most_recent_version(data, table_name, group_cols)
    return data


def effective_date_and_version_filter_for_inter_seg(data, save_location_formated, date_time, datetime_name, table_name):
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


def start_date_end_date_filter(data, save_location_formated, date_time, datetime_name, table_name):
    date_time = datetime.strptime(date_time, '%Y/%m/%d %H:%M:%S')
    data['START_DATE'] = pd.to_datetime(data['START_DATE'], format='%Y/%m/%d %H:%M:%S')
    data['END_DATE'] = pd.to_datetime(data['END_DATE'], format='%Y/%m/%d %H:%M:%S')
    data = data[(data['START_DATE'] <= date_time) & (data['END_DATE'] > date_time)]
    data = data.sort_values('END_DATE').groupby(['DUID', 'START_DATE'], as_index=False).first()
    data = data.sort_values('START_DATE').groupby(['DUID'], as_index=False).first()
    # data = query_wrapers.most_recent_records_before_start_time(data, date_time, table_name)
    return data


def no_filter(data, save_location_formated, date_time, datetime_name, table_name):
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
    curr = start_time + delta
    while curr <= end_time:
        # Change the datetime object to a timestamp and modify it format by replacing characters.
        yield curr.isoformat().replace('T', ' ').replace('-', '/')
        curr += delta


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
              'MNSP_PEROFFER': half_hour_peroids,
              'DISPATCHLOAD': settlement_date_filter,
              'LOSSMODEL': effective_date_and_version_filter_for_inter_seg,
              'LOSSFACTORMODEL': effective_date_and_version_filter,
              'DISPATCHREGIONSUM': settlement_date_filter,
              'MARKET_PRICE_THRESHOLDS': effective_date_and_version_filter}
