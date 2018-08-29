import pandas as pd
import query_wrapers as qw
import dashboards as db
import data_fetch_methods

year = 2017
ram_disk_path = 'E:/anvil_data'
regions_to_price = ['SA1', 'NSW1', 'QLD1', 'VIC1', 'TAS1']
raw_data = 'E:/anvil_data/raw'
filtered_data = 'E:/anvil_data/filtered'
results = 'E:/anvil_data/results_2'
for month in range(9, 10):
    start_time = '2017/{}/01 00:00:00'.format(str(month).zfill(2))  # inclusive
    if month != 12:
        end_time = '2017/{}/03 00:00:00'.format(str(month + 1 - 1).zfill(2))  # exclusive
    else:
        end_time = '2018/01/01 00:00:00'
    nemlite_results = pd.read_csv('E:/anvil_data/test_one_dir_flow_cons{}_{}.csv'.format(year, str(month).zfill(2)))
    nemlite_results = nemlite_results[nemlite_results['DateTime'] < end_time]
    actual_prices = data_fetch_methods.method_map['DISPATCHPRICE'](start_time, end_time, 'DISPATCHPRICE', raw_data, filter_cols=('INTERVENTION',),
                                filter_values=(['0'],))
    actual_prices['SETTLEMENTDATE'] = actual_prices['SETTLEMENTDATE'].apply(lambda dt: dt.strftime('%Y/%m/%d %H:%M:%S'))
    actual_prices['RRP'] = pd.to_numeric(actual_prices['RRP'])
    db.construct_pdf(nemlite_results, actual_prices,'E:/anvil_data/test_one_dir_flow_cons_2_{}_{}.pdf'.format(year, str(month).zfill(2)))

    # Check this is not in master