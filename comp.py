import pandas as pd

date_time = '2017/09/01 04:25:00'
datetime_name = date_time.replace('/', '')
datetime_name = datetime_name.replace(" ", "_")
datetime_name = datetime_name.replace(":", "")
filtered_data = 'E:/anvil_data/filtered'
save_location_formated = filtered_data + '/{}_{}.csv'
save_name_and_location = save_location_formated.format('DISPATCHLOAD', datetime_name)
save_name_and_location_info = save_location_formated.format('DUDETAILSUMMARY', datetime_name)
save_name_and_location_inter = save_location_formated.format('DISPATCHINTERCONNECTORRES', datetime_name)
actual_inter_flow = pd.read_csv(save_name_and_location_inter)
gen_info = pd.read_csv(save_name_and_location_info)
gen_info = gen_info.loc[:, ('DUID', 'DISPATCHTYPE', 'REGIONID')]
actual_distpatch = pd.read_csv(save_name_and_location)
nemlite_dispatch = pd.read_csv('E:/anvil_data/results_2/dispatch_{}_BASERUN.CSV'.format(datetime_name))
nemlite_interflow = pd.read_csv('E:/anvil_data/results_2/inter_flow_{}_BASERUN.CSV'.format(datetime_name))
nemlite_dispatch_e = nemlite_dispatch[nemlite_dispatch['BIDTYPE']=='ENERGY']
nemlite_dispatch_eg = nemlite_dispatch_e.groupby('DUID', as_index=False).sum()
comp = pd.merge(actual_distpatch, nemlite_dispatch_eg, 'inner', 'DUID')
comp = pd.merge(comp, gen_info, 'inner', 'DUID')
comp['ERROR'] = comp['TOTALCLEARED'] - comp['DISPATCHED']
view = comp.loc[:, ('DUID', 'DISPATCHTYPE', "REGIONID", 'INTERVENTION', 'DISPATCHMODE', 'AVAILABILITY', 'INITIALMW', 'TOTALCLEARED', 'RAMPUPRATE', 'RAMPDOWNRATE', 'DISPATCHED', 'ERROR')]
print(comp['ERROR'].sum())
x=1