import unittest
from nemlite import input_generator
import pandas as pd
from pandas.util.testing import assert_frame_equal


class TestFilterDispatchLoad(unittest.TestCase):
    def test_all_units_in_mode_0(self):
        dispatch_data = pd.DataFrame({
            'DUID': ['A', 'A', 'B', 'B'],
            'DISPATCHMODE': ['0', '0', '0', '1'],
            'SETTLEMENTDATE': ['2018/01/01 00:05:00', '2018/01/01 00:00:00',
                               '2018/01/01 00:05:00', '2018/01/01 00:00:00']
        })
        date_time = '2018/01/01 00:05:00'
        output = input_generator.filter_dispatch_load(data=dispatch_data, date_time=date_time)
        expected_output = pd.DataFrame({
            'DUID': ['A', 'B'],
            'DISPATCHMODE': ['0', '0'],
            'SETTLEMENTDATE': ['2018/01/01 00:05:00', '2018/01/01 00:05:00'],
            'TIMESINCECOMMITMENT': [0, 0]
        })
        expected_output['SETTLEMENTDATE'] = pd.to_datetime(expected_output['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
        assert_frame_equal(output.reset_index(drop=True), expected_output.reset_index(drop=True))

    def test_all_units_not_in_mode_0_but_not_mode_0_in_inputs(self):
        dispatch_data = pd.DataFrame({
            'DUID': ['A', 'A', 'B', 'B'],
            'DISPATCHMODE': ['1', '2', '4', '1'],
            'SETTLEMENTDATE': ['2018/01/01 00:05:00', '2018/01/01 00:00:00',
                               '2018/01/01 00:05:00', '2018/01/01 00:00:00']
        })
        date_time = '2018/01/01 00:05:00'
        output = input_generator.filter_dispatch_load(data=dispatch_data, date_time=date_time)
        expected_output = pd.DataFrame({
            'DUID': ['A', 'B'],
            'DISPATCHMODE': ['1', '4'],
            'SETTLEMENTDATE': ['2018/01/01 00:05:00', '2018/01/01 00:05:00'],
            'TIMESINCECOMMITMENT': [60, 60]
        })
        expected_output['SETTLEMENTDATE'] = pd.to_datetime(expected_output['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
        assert_frame_equal(output.reset_index(drop=True), expected_output.reset_index(drop=True))

    def test_all_units_not_in_mode_0_history_goes_back_to_mode_0(self):
        dispatch_data = pd.DataFrame({
            'DUID': ['A', 'A', 'B', 'B'],
            'DISPATCHMODE': ['1', '0', '4', '0'],
            'SETTLEMENTDATE': ['2018/01/01 00:05:00', '2018/01/01 00:00:00',
                               '2018/01/01 00:05:00', '2017/12/31 23:50:00']
        })
        date_time = '2018/01/01 00:05:00'
        output = input_generator.filter_dispatch_load(data=dispatch_data, date_time=date_time)
        expected_output = pd.DataFrame({
            'DUID': ['A', 'B'],
            'DISPATCHMODE': ['1', '4'],
            'SETTLEMENTDATE': ['2018/01/01 00:05:00', '2018/01/01 00:05:00'],
            'TIMESINCECOMMITMENT': [5, 15]
        })
        expected_output['SETTLEMENTDATE'] = pd.to_datetime(expected_output['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
        assert_frame_equal(output.reset_index(drop=True), expected_output.reset_index(drop=True))

    def test_units_in_different_buckets(self):
        dispatch_data = pd.DataFrame({
            'DUID': ['A', 'A', 'B', 'B', 'C', 'C'],
            'DISPATCHMODE': ['0', '0', '4', '0', '2', '1'],
            'SETTLEMENTDATE': ['2018/01/01 00:05:00', '2018/01/01 00:00:00',
                               '2018/01/01 00:05:00', '2017/12/31 23:50:00',
                               '2018/01/01 00:05:00', '2017/12/31 23:50:00']
        })
        date_time = '2018/01/01 00:05:00'
        output = input_generator.filter_dispatch_load(data=dispatch_data, date_time=date_time)
        expected_output = pd.DataFrame({
            'DUID': ['A', 'B', 'C'],
            'DISPATCHMODE': ['0', '4', '2'],
            'SETTLEMENTDATE': ['2018/01/01 00:05:00', '2018/01/01 00:05:00', '2018/01/01 00:05:00'],
            'TIMESINCECOMMITMENT': [0, 15, 60]
        })
        expected_output['SETTLEMENTDATE'] = pd.to_datetime(expected_output['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
        assert_frame_equal(output.reset_index(drop=True), expected_output.reset_index(drop=True))

