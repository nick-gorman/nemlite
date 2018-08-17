import input_generator
import engine
import dashboards as db
import pandas as pd
import data_fetch_methods

# The time window variables, these define the times for which the program will run the dispatch algo.
start_time = '2017/04/01 12:05:00'  # inclusive
end_time = '2017/04/01 13:10:00'  # exclusive
ram_disk_path = 'C:/Users/user/PycharmProjects/anvil/venv/Lib/site-packages/pulp/solverdir/cbc/win/64'
regions_to_price = ['SA1', 'NSW1', 'QLD1', 'VIC1', 'TAS1']
raw_data = 'E:/anvil_data/raw'
filtered_data = 'E:/anvil_data/filtered_small_2'
inputs = input_generator.actual_inputs_replicator(start_time, end_time, raw_data, filtered_data, True)
nemlite_results, objective_data_frame = engine.run(inputs, start_time, end_time, cbc_path=ram_disk_path,
                                                   regions_to_price=regions_to_price,
                                                   save_to='E:/anvil_data/results_2')
nemlite_results.to_csv('E:/anvil_data/test.csv')
#nemlite_results = pd.read_csv('C:/Users/user/anvil_data/test_new_FSO_min_energy.csv')
actual_prices = data_fetch_methods.method_map['DISPATCHPRICE']\
    (start_time, end_time, 'DISPATCHPRICE', raw_data, filter_cols=('INTERVENTION',),filter_values=(['0'],))
actual_prices['SETTLEMENTDATE'] = actual_prices['SETTLEMENTDATE'].apply(lambda dt: dt.strftime('%Y/%m/%d %H:%M:%S'))
actual_prices['RRP'] = pd.to_numeric(actual_prices['RRP'])
db.construct_pdf(nemlite_results, actual_prices, save_as='test.pdf')

