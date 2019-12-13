import unittest
from nemlite import pre_process_bids
import pandas as pd
from pandas.util.testing import assert_frame_equal


class TestAddMaxUnitEnergy(unittest.TestCase):
    def test_simple_case(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        maxavail = [100, 50, 26]
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype,
                                           'MAXAVAIL': maxavail})

        # Test unit solution data
        duid = ["B1", "B2"]
        avail = [99, 25]
        ramp_up = [36, 12]
        initial = [99, 25]
        initial_cons = pd.DataFrame.from_dict({'DUID': duid,
                                               'AVAILABILITY': avail,
                                               'INITIALMW': initial,
                                               'RAMPUPRATE': ramp_up})

        # Expected output
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_energy = [99.0, 99.0, 25.0]
        offer_max_energy = [100, 100, 26]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MAXAVAIL': maxavail,
                                         'MAXENERGY': max_energy,
                                         'OFFERMAXENERGY': offer_max_energy})

        calculated_answer = pre_process_bids.add_max_unit_energy(cap_bids, initial_cons)
        assert_frame_equal(calculated_answer, result)

    def test_ramp_rates_bind_on_one(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        maxavail = [100, 50, 26]
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype,
                                           'MAXAVAIL': maxavail})

        # Test unit solution data
        duid = ["B1", "B2"]
        avail = [99, 26]
        ramp_up = [36, 12]
        initial = [90, 25]
        initial_cons = pd.DataFrame.from_dict({'DUID': duid,
                                               'AVAILABILITY': avail,
                                               'INITIALMW': initial,
                                               'RAMPUPRATE': ramp_up})

        # Expected output
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_energy = [93.0, 93.0, 26.0]
        offer_max_energy = [100, 100, 26]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MAXAVAIL': maxavail,
                                         'MAXENERGY': max_energy,
                                         'OFFERMAXENERGY': offer_max_energy})

        calculated_answer = pre_process_bids.add_max_unit_energy(cap_bids, initial_cons)
        assert_frame_equal(calculated_answer, result)


class TestAddMinUnitEnergy(unittest.TestCase):
    def test_simple_case(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype})

        # Test unit solution data
        duid = ["B1", "B2"]
        ramp_down = [36, 12]
        initial = [99, 25]
        initial_cons = pd.DataFrame.from_dict({'DUID': duid,
                                               'INITIALMW': initial,
                                               'RAMPDOWNRATE': ramp_down})

        # Expected output
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        min_energy = [96.0, 96.0, 24.0]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MINENERGY': min_energy})

        calculated_answer = pre_process_bids.add_min_unit_energy(cap_bids, initial_cons)
        assert_frame_equal(calculated_answer, result)


class TestRationaliseMaxEnergyConstraint(unittest.TestCase):
    def test_simple_case(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        min_energy = [96.0, 96.0, 24.0]
        max_energy = [97.0, 97.0, 23.0]
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype,
                                           'MINENERGY': min_energy,
                                           'MAXENERGY': max_energy})

        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        min_energy = [96.0, 96.0, 24.0]
        max_energy = [97.0, 97.0, 24.0]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MINENERGY': min_energy,
                                         'MAXENERGY': max_energy})

        calculated_answer = pre_process_bids.rationalise_max_energy_constraint(cap_bids)
        assert_frame_equal(calculated_answer, result)


class TestRemoveEnergyBidsWithMaxEnergyZero(unittest.TestCase):
    def test_remove_none(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_energy = [97.0, 97.0, 24.0]
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype,
                                           'MAXENERGY': max_energy})

        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_energy = [97.0, 97.0, 24.0]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MAXENERGY': max_energy})

        calculated_answer = pre_process_bids.remove_energy_bids_with_max_energy_zero(cap_bids)
        assert_frame_equal(calculated_answer, result)

    def test_remove_one(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_energy = [97.0, 97.0, 0.0]
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype,
                                           'MAXENERGY': max_energy})

        duid = ["B1", "B1"]
        bidtype = ['ENERGY', 'OTHER']
        max_energy = [97.0, 97.0]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MAXENERGY': max_energy})

        calculated_answer = pre_process_bids.remove_energy_bids_with_max_energy_zero(cap_bids)
        assert_frame_equal(calculated_answer, result)

    def test_remove_energy_keep_fcas(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_energy = [0.0, 0.0, 24.0]
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype,
                                           'MAXENERGY': max_energy})

        duid = ["B1", "B2"]
        bidtype = ['OTHER', 'ENERGY']
        max_energy = [0.0, 24.0]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MAXENERGY': max_energy})

        calculated_answer = pre_process_bids.remove_energy_bids_with_max_energy_zero(cap_bids)
        assert_frame_equal(calculated_answer.reset_index(drop=True), result)


class TestRemoveFCASBidsWithMaxAvailZero(unittest.TestCase):
    def test_remove_none(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_avail = [97.0, 97.0, 24.0]
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype,
                                           'MAXAVAIL': max_avail})

        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_avail = [97.0, 97.0, 24.0]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MAXAVAIL': max_avail})

        calculated_answer = pre_process_bids.remove_fcas_bids_with_max_avail_zero(cap_bids)
        assert_frame_equal(calculated_answer, result)

    def test_remove_one(self):
        # Test capacity bid data
        duid = ["B1", "B1", "B2"]
        bidtype = ['ENERGY', 'OTHER', 'ENERGY']
        max_avail = [97.0, 0.0, 0.0]
        cap_bids = pd.DataFrame.from_dict({'DUID': duid,
                                           'BIDTYPE': bidtype,
                                           'MAXAVAIL': max_avail})

        duid = ["B1", "B2"]
        bidtype = ['ENERGY', 'ENERGY']
        max_avail = [97.0, 0.0]
        result = pd.DataFrame.from_dict({'DUID': duid,
                                         'BIDTYPE': bidtype,
                                         'MAXAVAIL': max_avail})

        calculated_answer = pre_process_bids.remove_fcas_bids_with_max_avail_zero(cap_bids)
        assert_frame_equal(calculated_answer.reset_index(drop=True), result)


class TestScaleMaxAvailableBasedOnRampRateS(unittest.TestCase):
    def test_ignore_wrong_bid_type(self):
        new_max = pre_process_bids.scale_max_available_based_on_ramp_rate_s(10, 5, 'A', 'B')
        self.assertEqual(new_max, 10)

    def test_scale_right_bid_type(self):
        new_max = pre_process_bids.scale_max_available_based_on_ramp_rate_s(10, 5, 'A', 'A')
        self.assertEqual(new_max, 5)

    def test_scale_ramp_rate_large_than_max(self):
        new_max = pre_process_bids.scale_max_available_based_on_ramp_rate_s(10, 11, 'A', 'A')
        self.assertEqual(new_max, 10)


class TestScaleUpperSlopeBasedOnTelemeteredDataS(unittest.TestCase):
    def test_ignore_wrong_bid_type(self):
        new_enablement, new_high_break_point = \
            pre_process_bids.scale_upper_slope_based_on_telemetered_data_s(10, 5, 4, 'A', 'B')
        self.assertEqual(new_enablement, 10)
        self.assertEqual(new_high_break_point, 4)

    def test_scale_right_bid_type(self):
        new_enablement, new_high_break_point = \
            pre_process_bids.scale_upper_slope_based_on_telemetered_data_s(10, 5, 4, 'A', 'A')
        self.assertEqual(new_enablement, 5)
        self.assertEqual(new_high_break_point, -1)

    def test_scale_ramp_rate_large_than_max(self):
        new_enablement, new_high_break_point = \
            pre_process_bids.scale_upper_slope_based_on_telemetered_data_s(10, 11, 4, 'A', 'A')
        self.assertEqual(new_enablement, 10)
        self.assertEqual(new_high_break_point, 4)


class TestScaleLowerSlopeBasedOnTelemeteredDataS(unittest.TestCase):
    def test_ignore_wrong_bid_type(self):
        new_enablement, new_low_break_point = \
            pre_process_bids.scale_lower_slope_based_on_telemetered_data_s(10, 5, 6, 'A', 'B')
        self.assertEqual(new_enablement, 10)
        self.assertEqual(new_low_break_point, 6)

    def test_scale_right_bid_type(self):
        new_enablement, new_low_break_point = \
            pre_process_bids.scale_lower_slope_based_on_telemetered_data_s(10, 5, 4, 'A', 'A')
        self.assertEqual(new_enablement, 5)
        self.assertEqual(new_low_break_point, 9)

    def test_scale_ramp_rate_large_than_max(self):
        new_enablement, new_low_break_point = \
            pre_process_bids.scale_lower_slope_based_on_telemetered_data_s(10, 11, 6, 'A', 'A')
        self.assertEqual(new_enablement, 10)
        self.assertEqual(new_low_break_point, 6)


class TestScaleLowBreakPointS(unittest.TestCase):
    def test_ignore_wrong_bid_type(self):
        new_low_break_point = pre_process_bids.scale_low_break_point_s(ramp_rate=5, max_avail=15, enable_min=0,
                                                                       low_break=15, reg_type='A', type_to_scale='B')
        self.assertEqual(new_low_break_point, 15)

    def test_scale_right_bid_type(self):
        new_low_break_point = pre_process_bids.scale_low_break_point_s(ramp_rate=5, max_avail=15, enable_min=0,
                                                                       low_break=15, reg_type='A', type_to_scale='A')
        self.assertEqual(new_low_break_point, 5.0)

    def test_scale_ramp_rate_large_than_max(self):
        new_low_break_point = pre_process_bids.scale_low_break_point_s(ramp_rate=20, max_avail=15, enable_min=0,
                                                                       low_break=15, reg_type='A', type_to_scale='A')
        self.assertEqual(new_low_break_point, 15)

    def test_enable_and_break_are_equal(self):
        new_low_break_point = pre_process_bids.scale_low_break_point_s(ramp_rate=5, max_avail=15, enable_min=15.0,
                                                                       low_break=15.0, reg_type='A', type_to_scale='A')
        self.assertEqual(new_low_break_point, 15.0)


class TestScaleHighBreakPointS(unittest.TestCase):
    def test_ignore_wrong_bid_type(self):
        new_high_break_point = pre_process_bids.scale_high_break_point_s(ramp_rate=5, max_avail=15, enable_max=15,
                                                                         high_break=0, reg_type='A', type_to_scale='B')
        self.assertEqual(new_high_break_point, 0)

    def test_scale_right_bid_type(self):
        new_high_break_point = pre_process_bids.scale_high_break_point_s(ramp_rate=5, max_avail=15, enable_max=15,
                                                                         high_break=0, reg_type='A', type_to_scale='A')
        self.assertEqual(new_high_break_point, 10.0)

    def test_scale_ramp_rate_large_than_max(self):
        new_high_break_point = pre_process_bids.scale_high_break_point_s(ramp_rate=20, max_avail=15, enable_max=15,
                                                                         high_break=0, reg_type='A', type_to_scale='A')
        self.assertEqual(new_high_break_point, 0)

    def test_enable_and_break_are_equal(self):
        new_high_break_point = pre_process_bids.scale_high_break_point_s(ramp_rate=5, max_avail=15, enable_max=15.0,
                                                                         high_break=15.0, reg_type='A',
                                                                         type_to_scale='A')
        self.assertEqual(new_high_break_point, 15.0)
