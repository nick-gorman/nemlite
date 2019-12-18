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
            pre_process_bids.scale_lower_slope_based_on_telemetered_data_s(enablement_min_as_bid=10,
                                                                           enablement_min_as_telemetered=15,
                                                                           low_break_point=4,
                                                                           bid_type='A', bid_type_to_scale='A')
        self.assertEqual(new_enablement, 15)
        self.assertEqual(new_low_break_point, 9)

    def test_dont_scale_because_less_restrictive(self):
        new_enablement, new_low_break_point = \
            pre_process_bids.scale_lower_slope_based_on_telemetered_data_s(10, 9, 6, 'A', 'A')
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


class TestFcasTrapeziumScalingOnTelemeteredRaiseRegEnablement(unittest.TestCase):
    def test_nothing_to_scale_because_telemetered_less_restrictive(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [11, 16]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [0, 0]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'RAISEREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'RAISEREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_raise_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_just_upper_slope_of_first_row(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [9, 16]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [0, 0]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'RAISEREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        enablement_max = [9, 15]
        high_break_point = [4, 10]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'RAISEREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_raise_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_just_upper_slope_of_both_rows(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [9, 12]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [0, 0]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'RAISEREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        enablement_max = [9, 12]
        high_break_point = [4, 7]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'RAISEREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_raise_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_just_lower_slope_of_first_row(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [11, 16]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [1, 0]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'RAISEREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        low_break_point = [6, 10]
        enablement_min = [1, 0]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'RAISEREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_raise_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_all_slopes(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [7, 11]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [1, 2]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'RAISEREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        enablement_max = [7, 11]
        high_break_point = [2, 6]
        low_break_point = [6, 12]
        enablement_min = [1, 2]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'RAISEREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_raise_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_nothing_because_of_bid_type(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [7, 11]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [1, 2]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'RAISEREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'RAISEREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_raise_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)


class TestFcasTrapeziumScalingOnTelemeteredLowerRegEnablement(unittest.TestCase):
    def test_nothing_to_scale_because_telemetered_less_restrictive(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [11, 16]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [0, 0]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'LOWERREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'LOWERREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_lower_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_just_upper_slope_of_first_row(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [9, 16]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [0, 0]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'LOWERREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        enablement_max = [9, 15]
        high_break_point = [4, 10]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'LOWERREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_lower_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_just_upper_slope_of_both_rows(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [9, 12]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [0, 0]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'LOWERREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        enablement_max = [9, 12]
        high_break_point = [4, 7]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'LOWERREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_lower_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_just_lower_slope_of_first_row(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [11, 16]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [1, 0]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'LOWERREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        low_break_point = [6, 10]
        enablement_min = [1, 0]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'LOWERREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_lower_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_all_slopes(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [7, 11]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [1, 2]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'LOWERREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        enablement_max = [7, 11]
        high_break_point = [2, 6]
        low_break_point = [6, 12]
        enablement_min = [1, 2]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'LOWERREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_lower_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale_nothing_because_of_bid_type(self):
        enablement_max = [10, 15]
        high_break_point = [5, 10]
        raise_reg_enablement_max = [7, 11]
        low_break_point = [5, 10]
        enablement_min = [0, 0]
        raise_reg_enablement_min = [1, 2]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'LOWERREGENABLEMENTMAX': raise_reg_enablement_max, 'ENABLEMENTMIN': enablement_min,
                                   'LOWBREAKPOINT': low_break_point, 'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                   'BIDTYPE': bid_type})
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'LOWERREGENABLEMENTMAX': raise_reg_enablement_max,
                                             'ENABLEMENTMIN': enablement_min, 'LOWBREAKPOINT': low_break_point,
                                             'LOWERREGENABLEMENTMIN': raise_reg_enablement_min,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_telemetered_lower_reg_enablement(input_data)
        assert_frame_equal(output_data, expected_output_data)


class TestFcasTrapeziumScalingOnRampUpRate(unittest.TestCase):
    def test_scale_nothing_as_ramp_rate_not_restrictive(self):
        enablement_max = [10, 15]
        high_break_point = [5.0, 10.0]
        low_break_point = [5.0, 10.0]
        enablement_min = [0, 0]
        maxavail = [10, 20]
        ramp_rate = [10 * 12, 20 * 12]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min, 'RAMPUPRATE': ramp_rate,
                                   'LOWBREAKPOINT': low_break_point, 'BIDTYPE': bid_type})
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min,
                                             'RAMPUPRATE': ramp_rate, 'LOWBREAKPOINT': low_break_point,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_ramp_up_rate(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale(self):
        enablement_max = [10, 15]
        high_break_point = [5.0, 10.0]
        low_break_point = [5.0, 10.0]
        enablement_min = [0, 0]
        maxavail = [10, 20]
        ramp_rate = [5 * 12, 10 * 12]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min, 'RAMPUPRATE': ramp_rate,
                                   'LOWBREAKPOINT': low_break_point, 'BIDTYPE': bid_type})
        high_break_point = [7.5, 12.5]
        low_break_point = [2.5, 5.0]
        maxavail = [5.0, 10.0]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min,
                                             'RAMPUPRATE': ramp_rate, 'LOWBREAKPOINT': low_break_point,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_ramp_up_rate(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_dont_scale_because_wrong_bid_type(self):
        enablement_max = [10, 15]
        high_break_point = [5.0, 10.0]
        low_break_point = [5.0, 10.0]
        enablement_min = [0, 0]
        maxavail = [10, 20]
        ramp_rate = [5 * 12, 10 * 12]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min, 'RAMPUPRATE': ramp_rate,
                                   'LOWBREAKPOINT': low_break_point, 'BIDTYPE': bid_type})
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min,
                                             'RAMPUPRATE': ramp_rate, 'LOWBREAKPOINT': low_break_point,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_ramp_up_rate(input_data)
        assert_frame_equal(output_data, expected_output_data)


class TestFcasTrapeziumScalingOnRampDOWNRate(unittest.TestCase):
    def test_scale_nothing_as_ramp_rate_not_restrictive(self):
        enablement_max = [10, 15]
        high_break_point = [5.0, 10.0]
        low_break_point = [5.0, 10.0]
        enablement_min = [0, 0]
        maxavail = [10, 20]
        ramp_rate = [10 * 12, 20 * 12]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min, 'RAMPDOWNRATE': ramp_rate,
                                   'LOWBREAKPOINT': low_break_point, 'BIDTYPE': bid_type})
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min,
                                             'RAMPDOWNRATE': ramp_rate, 'LOWBREAKPOINT': low_break_point,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_ramp_down_rate(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_scale(self):
        enablement_max = [10, 15]
        high_break_point = [5.0, 10.0]
        low_break_point = [5.0, 10.0]
        enablement_min = [0, 0]
        maxavail = [10, 20]
        ramp_rate = [5 * 12, 10 * 12]
        bid_type = 'LOWERREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min, 'RAMPDOWNRATE': ramp_rate,
                                   'LOWBREAKPOINT': low_break_point, 'BIDTYPE': bid_type})
        high_break_point = [7.5, 12.5]
        low_break_point = [2.5, 5.0]
        maxavail = [5.0, 10.0]
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min,
                                             'RAMPDOWNRATE': ramp_rate, 'LOWBREAKPOINT': low_break_point,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_ramp_down_rate(input_data)
        assert_frame_equal(output_data, expected_output_data)

    def test_dont_scale_because_wrong_bid_type(self):
        enablement_max = [10, 15]
        high_break_point = [5.0, 10.0]
        low_break_point = [5.0, 10.0]
        enablement_min = [0, 0]
        maxavail = [10, 20]
        ramp_rate = [5 * 12, 10 * 12]
        bid_type = 'RAISEREG'
        input_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                   'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min, 'RAMPDOWNRATE': ramp_rate,
                                   'LOWBREAKPOINT': low_break_point, 'BIDTYPE': bid_type})
        expected_output_data = pd.DataFrame({'ENABLEMENTMAX': enablement_max, 'HIGHBREAKPOINT': high_break_point,
                                             'MAXAVAIL': maxavail, 'ENABLEMENTMIN': enablement_min,
                                             'RAMPDOWNRATE': ramp_rate, 'LOWBREAKPOINT': low_break_point,
                                             'BIDTYPE': bid_type})
        output_data = pre_process_bids.fcas_trapezium_scaling_on_ramp_down_rate(input_data)
        assert_frame_equal(output_data, expected_output_data)


class TestApplyFcasEnablementCriteria(unittest.TestCase):
    def test_all_units_meet_criteria(self):
        duid = ['a', 'b', 'c']
        initial_mw = [50, 111, 98]
        agc_status = [1, 1, 1]
        initial_cons = pd.DataFrame({'DUID': duid, 'INITIALMW': initial_mw, 'AGCSTATUS': agc_status})

        duid = ['a', 'b', 'c']
        bid_type = ['RAISEREG', 'LOWERREG', 'ENERGY']
        enablement_min = [25, 80, 70]
        enablement_max = [275, 150, 120]
        max_energy = [30, 90, 60]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'ENABLEMENTMIN': enablement_min,
                                      'ENABLEMENTMAX': enablement_max, 'MAXENERGY': max_energy})
        expected_output_data = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'ENABLEMENTMIN': enablement_min,
                                      'ENABLEMENTMAX': enablement_max, 'MAXENERGY': max_energy})
        output_data = pre_process_bids.apply_fcas_enablement_criteria(capacity_bids, initial_cons)
        assert_frame_equal(output_data, expected_output_data)

    def test_reg_services_filtered_out_if_agc_not_1(self):
        duid = ['a', 'b', 'c']
        initial_mw = [50, 111, 98]
        agc_status = [0, 0, 0]
        initial_cons = pd.DataFrame({'DUID': duid, 'INITIALMW': initial_mw, 'AGCSTATUS': agc_status})

        duid = ['a', 'b', 'c']
        bid_type = ['RAISEREG', 'LOWERREG', 'ENERGY']
        enablement_min = [25, 80, 70]
        enablement_max = [275, 150, 120]
        max_energy = [30, 90, 60]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'ENABLEMENTMIN': enablement_min,
                                      'ENABLEMENTMAX': enablement_max, 'MAXENERGY': max_energy})
        expected_output_data = capacity_bids[capacity_bids['DUID']=='c']
        output_data = pre_process_bids.apply_fcas_enablement_criteria(capacity_bids, initial_cons)
        assert_frame_equal(output_data, expected_output_data)

    def test_filter_a_because_initial_lower_than_enablement_min(self):
        duid = ['a', 'b', 'c']
        initial_mw = [20, 111, 98]
        agc_status = [1, 1, 1]
        initial_cons = pd.DataFrame({'DUID': duid, 'INITIALMW': initial_mw, 'AGCSTATUS': agc_status})

        duid = ['a', 'b', 'c']
        bid_type = ['RAISEREG', 'LOWERREG', 'ENERGY']
        enablement_min = [25, 80, 70]
        enablement_max = [275, 150, 120]
        max_energy = [30, 90, 60]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'ENABLEMENTMIN': enablement_min,
                                      'ENABLEMENTMAX': enablement_max, 'MAXENERGY': max_energy})
        expected_output_data = capacity_bids[capacity_bids['DUID'].isin(['b', 'c'])]
        output_data = pre_process_bids.apply_fcas_enablement_criteria(capacity_bids, initial_cons)
        assert_frame_equal(output_data, expected_output_data)

    def test_filter_b_because_initial_higher_than_enablement_max(self):
        duid = ['a', 'b', 'c']
        initial_mw = [50, 160, 98]
        agc_status = [1, 1, 1]
        initial_cons = pd.DataFrame({'DUID': duid, 'INITIALMW': initial_mw, 'AGCSTATUS': agc_status})

        duid = ['a', 'b', 'c']
        bid_type = ['RAISEREG', 'LOWERREG', 'ENERGY']
        enablement_min = [25, 80, 70]
        enablement_max = [275, 150, 120]
        max_energy = [30, 90, 60]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'ENABLEMENTMIN': enablement_min,
                                      'ENABLEMENTMAX': enablement_max, 'MAXENERGY': max_energy})
        expected_output_data = capacity_bids[capacity_bids['DUID'].isin(['a', 'c'])]
        output_data = pre_process_bids.apply_fcas_enablement_criteria(capacity_bids, initial_cons)
        assert_frame_equal(output_data, expected_output_data)

    def test_dont_filter_energy_even_enablement_outside_bounds(self):
        duid = ['a', 'b', 'c']
        initial_mw = [50, 160, 60]
        agc_status = [1, 1, 1]
        initial_cons = pd.DataFrame({'DUID': duid, 'INITIALMW': initial_mw, 'AGCSTATUS': agc_status})

        duid = ['a', 'b', 'c']
        bid_type = ['RAISEREG', 'LOWERREG', 'ENERGY']
        enablement_min = [25, 80, 70]
        enablement_max = [275, 150, 120]
        max_energy = [30, 90, 60]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'ENABLEMENTMIN': enablement_min,
                                      'ENABLEMENTMAX': enablement_max, 'MAXENERGY': max_energy})
        expected_output_data = capacity_bids[capacity_bids['DUID'].isin(['a', 'c'])]
        output_data = pre_process_bids.apply_fcas_enablement_criteria(capacity_bids, initial_cons)
        assert_frame_equal(output_data, expected_output_data)

    def test_filter_if_max_energy_less_than_enablement_min(self):
        duid = ['a', 'b', 'c']
        initial_mw = [50, 160, 80]
        agc_status = [1, 1, 1]
        initial_cons = pd.DataFrame({'DUID': duid, 'INITIALMW': initial_mw, 'AGCSTATUS': agc_status})

        duid = ['a', 'b', 'c']
        bid_type = ['RAISEREG', 'LOWERREG', 'ENERGY']
        max_energy = [10, 15, 60]
        enablement_min = [25, 80, 70]
        enablement_max = [275, 150, 120]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'ENABLEMENTMIN': enablement_min,
                                      'ENABLEMENTMAX': enablement_max, 'MAXENERGY': max_energy})
        expected_output_data = capacity_bids[capacity_bids['DUID'].isin(['c'])]
        output_data = pre_process_bids.apply_fcas_enablement_criteria(capacity_bids, initial_cons)
        assert_frame_equal(output_data, expected_output_data)


class TestFilterAndScale(unittest.TestCase):
    def test_run_with_no_change(self):
        duid = ['a', 'b']
        initial_mw = [50, 150]
        agc_status = [1, 1]
        raise_reg_enablement_max = [280, 0]
        raise_reg_enablement_min = [20, 0]
        lower_reg_enablement_max = [0, 160]
        lower_reg_enablement_min = [0, 70]
        availability = [100, 200]
        ramp_up_rate = [50 * 12, 90 * 12]
        ramp_down_rate = [50 * 12, 40 * 12]
        initial_cons = pd.DataFrame({'DUID': duid, 'INITIALMW': initial_mw, 'AGCSTATUS': agc_status,
                                     'RAISEREGENABLEMENTMAX': raise_reg_enablement_max,
                                     'RAISEREGENABLEMENTMIN': raise_reg_enablement_min,
                                     'LOWERREGENABLEMENTMAX': lower_reg_enablement_max,
                                     'LOWERREGENABLEMENTMIN': lower_reg_enablement_min,
                                     'AVAILABILITY': availability, 'RAMPUPRATE': ramp_up_rate,
                                     'RAMPDOWNRATE': ramp_down_rate})

        duid = ['a', 'b', 'a', 'b']
        bid_type = ['RAISEREG', 'LOWERREG', 'ENERGY', 'ENERGY']
        max_avail = [25, 30, 100, 110]
        enablement_min = [25, 80, 0, 0]
        enablement_max = [275, 150, 0, 0]
        low_break_point = [45, 90, 0, 0]
        high_break_point = [250, 130, 0, 0]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'ENABLEMENTMIN': enablement_min,
                                      'ENABLEMENTMAX': enablement_max, 'MAXAVAIL': max_avail,
                                      'HIGHBREAKPOINT': high_break_point, 'LOWBREAKPOINT': low_break_point})

        expected_output_data = pd.merge(capacity_bids, initial_cons, on='DUID', how='left')
        expected_output_data = expected_output_data.drop(
            ['RAISEREGENABLEMENTMAX', 'RAISEREGENABLEMENTMIN', 'LOWERREGENABLEMENTMAX', 'LOWERREGENABLEMENTMIN',
             'INITIALMW', 'AGCSTATUS', 'AVAILABILITY']
            , axis=1)
        expected_output_data['MAXENERGY'] = [100, 110, 100, 110]
        expected_output_data['MINENERGY'] = [0, 110, 0, 110]
        output_data = pre_process_bids.filter_and_scale(capacity_bids, initial_cons)
        expected_output_data = expected_output_data.loc[:, output_data.columns]
        expected_output_data = expected_output_data.sort_values('DUID').reset_index(drop=True)
        assert_frame_equal(expected_output_data, output_data)
