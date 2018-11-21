# Nemlite
See wiki for more information.

# Example usage
Using the inbuilt historical input replicator 

import nemlite
import pandas as pd
```
# Note running this file as is will consume large amounts of disk space and take several weeks, probably try running it
# for a much smaller time window first.

# Specify some locations to save data. Create and specify your own!
raw_data = 'your folder path to cache raw data from AEMO'
filtered_data = 'your folder path to save filtered AEMO data for running nemlite'

# Specify the backcast period. Choose a short period, it will probably not work for some time back in history when the
# AEMO data was structured differently.
start_times = '2017/01/01 00:00:00'
end_times = '2017/01/01 00:30:00'


# Create an generator of actual historical NEMDE inputs.
inputs = nemlite.actual_inputs_replicator(start_time, end_time, raw_data, filtered_data, True)

# Create a data frame to save the results
nemlite_results_cumulative = pd.DataFrame()

# Iterate other the inputs to
for [dispatch_unit_information, dispatch_unit_capacity_bids, initial_conditions, interconnectors,
     regional_demand, dispatch_unit_price_bids, regulated_interconnectors_loss_model, connection_point_constraints,
     interconnector_constraints, constraint_data, region_constraints, timestamp,
     regulated_interconnector_loss_factor_model,
     market_interconnectors, market_interconnector_price_bids, market_interconnector_capacity_bids,
     market_cap_and_floor] in inputs:

     nemlite_results, dispatches, inter_flows = nemlite.run(dispatch_unit_information, dispatch_unit_capacity_bids,
                                                            initial_conditions, interconnectors,
                                                            regional_demand, dispatch_unit_price_bids,
                                                            regulated_interconnectors_loss_model,
                                                            connection_point_constraints,
                                                            interconnector_constraints, constraint_data,
                                                            region_constraints,
                                                            regulated_interconnector_loss_factor_model,
                                                            market_interconnectors, market_interconnector_price_bids,
                                                            market_interconnector_capacity_bids,
                                                            market_cap_and_floor)

      nemlite_results['DateTime'] = timestamp
      nemlite_results_cumulative = pd.concat([nemlite_results_cumulative, nemlite_results])
      print(timestamp)

nemlite_results_cumulative.to_csv('your_path/price_results_{}_{}.csv'.format(start_time[:4], start_time[5:7]))
    ```


