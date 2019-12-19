import unittest
from nemlite import joint_fcas_energy_constraints
import pandas as pd
from pandas.util.testing import assert_frame_equal


class TestCreateJointRampingConstraints(unittest.TestCase):
    def test_apply_no_duids_due_to_raise_reg_bid_and_ramping_up(self):
        duid = ['a', 'b']
        energy = [1.0, 1.0]
        raise_reg = [1.0, 0.0]
        bid_type_check = pd.DataFrame({'DUID': duid, 'ENERGY': energy, 'RAISEREG': raise_reg})
        duid = ['a', 'b']
        ramp_up_rate = [0.0, 100]
        initial_cons = pd.DataFrame({'DUID': duid, 'RAMPUPRATE': ramp_up_rate})
        expected_output = []
        output = joint_fcas_energy_constraints.get_duids_that_joint_ramping_constraints_apply_to(
            bid_type_check, initial_cons, 'RAISEREG')
        self.assertListEqual(expected_output, output)

    def test_apply_no_duids_due_to_raise_reg_bid_and_ramping_down(self):
        duid = ['a', 'b']
        energy = [1.0, 1.0]
        raise_reg = [1.0, 0.0]
        bid_type_check = pd.DataFrame({'DUID': duid, 'ENERGY': energy, 'LOWERREG': raise_reg})
        duid = ['a', 'b']
        ramp_up_rate = [0.0, 100]
        initial_cons = pd.DataFrame({'DUID': duid, 'RAMPDOWNRATE': ramp_up_rate})
        expected_output = []
        output = joint_fcas_energy_constraints.get_duids_that_joint_ramping_constraints_apply_to(
            bid_type_check, initial_cons, 'LOWERREG')
        self.assertListEqual(expected_output, output)

    def test_to_one_duid_lower_reg(self):
        duid = ['a', 'b']
        energy = [1.0, 1.0]
        raise_reg = [1.0, 0.0]
        bid_type_check = pd.DataFrame({'DUID': duid, 'ENERGY': energy, 'LOWERREG': raise_reg})
        duid = ['a', 'b']
        ramp_up_rate = [1, 100]
        initial_cons = pd.DataFrame({'DUID': duid, 'RAMPDOWNRATE': ramp_up_rate})
        expected_output = ['a']
        output = joint_fcas_energy_constraints.get_duids_that_joint_ramping_constraints_apply_to(
            bid_type_check, initial_cons, 'LOWERREG')
        self.assertListEqual(expected_output, output)

    def test_to_one_duid_raise_reg(self):
        duid = ['a', 'b']
        energy = [1.0, 1.0]
        raise_reg = [1.0, 0.0]
        bid_type_check = pd.DataFrame({'DUID': duid, 'ENERGY': energy, 'RAISEREG': raise_reg})
        duid = ['a', 'b']
        ramp_up_rate = [1, 100]
        initial_cons = pd.DataFrame({'DUID': duid, 'RAMPUPRATE': ramp_up_rate})
        expected_output = ['a']
        output = joint_fcas_energy_constraints.get_duids_that_joint_ramping_constraints_apply_to(
            bid_type_check, initial_cons, 'RAISEREG')
        self.assertListEqual(expected_output, output)

    def test_create_constraint_indexes_one_duid(self):
        duid = ['a', 'a']
        energy = ['RAISEREG', 'ENERGY']
        bids_and_indexes = pd.DataFrame({'DUID': duid, 'BIDTYPE': energy})
        expected_output = bids_and_indexes
        expected_output['ROWINDEX'] = [1, 1]
        output = joint_fcas_energy_constraints.setup_data_to_calc_joint_ramping_constraints(
            ['a'], bids_and_indexes, 'RAISEREG', 0)
        assert_frame_equal(expected_output, output)

    def test_create_constraint_indexes_two_duids(self):
        duid = ['b', 'a', 'a', 'b']
        energy = ['RAISEREG', 'ENERGY', 'RAISEREG', 'ENERGY']
        bids_and_indexes = pd.DataFrame({'DUID': duid, 'BIDTYPE': energy})
        expected_output = bids_and_indexes
        expected_output['ROWINDEX'] = [2, 1, 1, 2]
        output = joint_fcas_energy_constraints.setup_data_to_calc_joint_ramping_constraints(
            ['a', 'b'], bids_and_indexes, 'RAISEREG', 0)
        assert_frame_equal(expected_output, output)

    def test_constraint_value_calcs_raise_reg(self):
        duid = ['b', 'a', 'a', 'b']
        bid_type = ['RAISEREG', 'ENERGY', 'RAISEREG', 'ENERGY']
        ramp_up_rate = [10, 5, 5, 10]
        initial_mw = [11, 8, 8, 11]
        bids_and_indexes = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'INITIALMW': initial_mw,
                                         'RAMPUPRATE': ramp_up_rate})
        expected_output = bids_and_indexes.copy()
        expected_output['CONSTRAINTTYPE'] = '<='
        expected_output['RHSCONSTANT'] = bids_and_indexes['INITIALMW'] + \
                                          bids_and_indexes['RAMPUPRATE'] / 12
        expected_output['LHSCOEFFICIENTS'] = 1
        output = joint_fcas_energy_constraints.calc_joint_ramping_constraint_values(bids_and_indexes,
                                                                                    'RAISEREG')
        assert_frame_equal(expected_output, output)

    def test_constraint_value_calcs_lower_reg(self):
        duid = ['b', 'a', 'a', 'b']
        bid_type = ['LOWERREG', 'ENERGY', 'LOWERREG', 'ENERGY']
        ramp_down_rate = [10, 5, 5, 10]
        initial_mw = [11, 8, 8, 11]
        bids_and_indexes = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'INITIALMW': initial_mw,
                                         'RAMPDOWNRATE': ramp_down_rate})
        expected_output = bids_and_indexes.copy()
        expected_output['CONSTRAINTTYPE'] = '>='
        expected_output['RHSCONSTANT'] = bids_and_indexes['INITIALMW'] - \
                                          bids_and_indexes['RAMPDOWNRATE'] / 12
        expected_output['LHSCOEFFICIENTS'] = [-1, 1, -1, 1]
        expected_output = expected_output.astype(dtype={'LHSCOEFFICIENTS': 'int32'})
        output = joint_fcas_energy_constraints.calc_joint_ramping_constraint_values(bids_and_indexes,
                                                                                    'LOWERREG')
        assert_frame_equal(expected_output, output)