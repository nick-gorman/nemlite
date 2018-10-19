import input_generator
import engine

year = 2017
cbc_path = 'C:/Users/user/PycharmProjects/anvil/venv/Lib/site-packages/pulp/solverdir/cbc/win/64'
regions_to_price = ['SA1', 'NSW1', 'QLD1', 'VIC1', 'TAS1']
raw_data = 'E:/anvil_data/raw'
filtered_data = 'E:/anvil_data/filtered'
results = 'E:/nemlite_journal_results/base_line'

for month in range(1, 2):
    start_time = '2017/{}/01 00:00:00'.format(str(month).zfill(2))  # inclusive
    if month != 12:
        end_time = '2017/{}/01 00:00:00'.format(str(month + 1).zfill(2))  # exclusive
    else:
        end_time = '2018/01/01 00:00:00'
    inputs = input_generator.actual_inputs_replicator(start_time, end_time, raw_data, filtered_data, True)
    nemlite_results, objective_data_frame = engine.run(inputs, cbc_path, regions_to_price=regions_to_price,
                                                       save_to=results, feed_back=False)
    nemlite_results.to_csv('E:/nemlite_journal_results/base_line_{}_{}.csv'.format(year, str(month).zfill(2)))