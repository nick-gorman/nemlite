import input_generator
import engine

child_parent_map = {'connection_point_constraints': ['SPDCONNECTIONPOINTCONSTRAINT'],
                    'constraint_data': ['GENCONDATA', 'DISPATCHCONSTRAINT'],
                    'interconnector_constraints': ['SPDINTERCONNECTORCONSTRAINT'],
                    'capacity_bids': ['BIDPEROFFER'], # Dispatch load should get merged here.
                    'interconnectors': ['INTERCONNECTOR', 'INTERCONNECTORCONSTRAINT'],
                    'market_interconnectors': ['MNSP_INTERCONNECTOR'],
                    'generator_information': ['DUDETAILSUMMARY'],
                    'region_constraints': ['SPDREGIONCONSTRAINT'],
                    'price_bids': ['BIDDAYOFFER'],
                    'market_interconnector_price_bids': ['MNSP_DAYOFFER'],
                    'market_interconnector_capacity_bids': ['MNSP_PEROFFER'],
                    'initial_conditions': ['DISPATCHLOAD'],
                    'interconnector_segments': ['LOSSMODEL', 'DISPATCHINTERCONNECTORRES'],
                    'interconnector_dynamic_loss_coefficients': ['LOSSFACTORMODEL'],
                    'demand': ['DISPATCHREGIONSUM']}

year = 2017
cbc_path = 'C:/Users/user/PycharmProjects/anvil/venv/Lib/site-packages/pulp/solverdir/cbc/win/64'
regions_to_price = ['SA1', 'NSW1', 'QLD1', 'VIC1', 'TAS1']
raw_data = 'E:/anvil_data/raw'
filtered_data = 'E:/anvil_data/filtered'
results = 'E:/anvil_data/results_2'

run_names = ['dir_flow_cons', 'dir_flow_cons_feed_back', 'dir_flow_cons_feed_back_no_fs']
feed_back_settings = [False, True, True]
fast_start_settings = [True, True, False]
pre_filter_settings = [True, False, False]

for name, feed_back, fast_start, pre_filter in zip(run_names, feed_back_settings, fast_start_settings, pre_filter_settings):

    for month in range(9, 10):
        start_time = '2017/{}/01 00:00:00'.format(str(month).zfill(2))  # inclusive
        if month != 12:
            end_time = '2017/{}/01 00:35:00'.format(str(month + 1 - 1).zfill(2))  # exclusive
        else:
            end_time = '2018/01/01 00:00:00'
        inputs = input_generator.actual_inputs_replicator(start_time, end_time, raw_data, filtered_data, pre_filter,
                                                          alternate_table_map=None)
        nemlite_results, objective_data_frame = engine.run(inputs, cbc_path, regions_to_price=regions_to_price,
                                                           save_to=results, feed_back=feed_back, fast_start=fast_start)
        nemlite_results.to_csv('E:/anvil_data/{}_{}_{}.csv'.format(name, year, str(month).zfill(2)))