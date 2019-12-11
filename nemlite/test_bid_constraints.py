import unittest
from nemlite import bid_constraints
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

        calculated_answer = bid_constraints.add_max_unit_energy(cap_bids, initial_cons)
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

        calculated_answer = bid_constraints.add_max_unit_energy(cap_bids, initial_cons)
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

        calculated_answer = bid_constraints.add_min_unit_energy(cap_bids, initial_cons)
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

        calculated_answer = bid_constraints.rationalise_max_energy_constraint(cap_bids)
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

        calculated_answer = bid_constraints.remove_energy_bids_with_max_energy_zero(cap_bids)
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

        calculated_answer = bid_constraints.remove_energy_bids_with_max_energy_zero(cap_bids)
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

        calculated_answer = bid_constraints.remove_energy_bids_with_max_energy_zero(cap_bids)
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

        calculated_answer = bid_constraints.remove_fcas_bids_with_max_avail_zero(cap_bids)
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

        calculated_answer = bid_constraints.remove_fcas_bids_with_max_avail_zero(cap_bids)
        assert_frame_equal(calculated_answer.reset_index(drop=True), result)


