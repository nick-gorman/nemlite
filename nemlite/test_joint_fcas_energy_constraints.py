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

    def test_create_constraint_indexes_two_duids(self):
        duid = ['b', 'a', 'a', 'b']
        energy = ['RAISEREG', 'ENERGY', 'RAISEREG', 'ENERGY']
        bids_and_indexes = pd.DataFrame({'DUID': duid, 'BIDTYPE': energy})
        initial_conditions = pd.DataFrame({
            'DUID': ['a', 'b'],
            'INITIALMW': [11, 12],
            'RAMPUPRATE': [15, 16],
            'RAMPDOWNRATE': [13, 14],
        })
        expected_output = bids_and_indexes.copy()
        expected_output['INITIALMW'] = [12, 11, 11, 12]
        expected_output['RAMPUPRATE'] = [16, 15, 15, 16]
        expected_output['RAMPDOWNRATE'] = [14, 13, 13, 14]
        expected_output['ROWINDEX'] = [2, 1, 1, 2]
        output = joint_fcas_energy_constraints.setup_data_to_calc_joint_ramping_constraints(
            ['a', 'b'], bids_and_indexes, initial_conditions, 'RAISEREG', 0)
        assert_frame_equal(expected_output.sort_values('DUID', ascending=False).reset_index(drop=True), output)

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

    def test_joint_ramping_end_to_end(self):
        duid = ['a', 'b']
        energy = [1.0, 1.0]
        raise_reg = [1.0, 1.0]
        bid_type_check = pd.DataFrame({'DUID': duid, 'ENERGY': energy, 'RAISEREG': raise_reg})
        duid = ['b', 'a', 'a', 'b']
        energy = ['RAISEREG', 'ENERGY', 'RAISEREG', 'ENERGY']
        index = [2, 1, 1, 2]
        bids_and_indexes = pd.DataFrame({'DUID': duid, 'BIDTYPE': energy, 'INDEX': index})
        initial_conditions = pd.DataFrame({
            'DUID': ['a', 'b'],
            'INITIALMW': [11, 12],
            'RAMPUPRATE': [15, 16],
            'RAMPDOWNRATE': [13, 14]
        })
        expected_output = pd.DataFrame()
        expected_output['INDEX'] = [2, 1, 1, 2]
        expected_output['ROWINDEX'] = [2, 1, 1, 2]
        expected_output['LHSCOEFFICIENTS'] = [1, 1, 1, 1]
        expected_output['CONSTRAINTTYPE'] = '<='
        expected_output['RHSCONSTANT'] = [12 + 16/12, 11 + 15/12, 11 + 15/12, 12 + 16/12]
        expected_output = expected_output.astype(dtype={'RHSCONSTANT': 'float64'})
        output = joint_fcas_energy_constraints.create_joint_ramping_constraints(
            bids_and_indexes, initial_conditions, 0, 'RAISEREG', bid_type_check)
        assert_frame_equal(expected_output.sort_values(['INDEX', 'ROWINDEX'], ascending=False).reset_index(drop=True),
                           output[0].sort_values(['INDEX', 'ROWINDEX'], ascending=False).reset_index(drop=True))


class TestCreateJointCapacityConstraints(unittest.TestCase):
    def test_get_unit_to_constrain_joint_capacity(self):
        bid_type_check = pd.DataFrame({
            'DUID': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            'ENERGY': [1, 1, 0, 1, 0, 1, 0],
            'LOWERREG': [1, 1, 1, 0, 0, 0, 1],
            'LOWERCON': [1, 0, 1, 1, 1, 0, 0]
        })
        bids_and_indexes = pd.DataFrame({
            'DUID': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'F', 'G'],
            'BIDTYPE': ['ENERGY', 'LOWERREG', 'LOWERCON', 'ENERGY', 'LOWERREG', 'LOWERREG', 'LOWERCON', 'ENERGY',
                        'LOWERCON', 'LOWERCON', 'ENEGRY', 'LOWERREG']
        })
        expected_output = pd.DataFrame({
            'DUID': ['A', 'A', 'A', 'D', 'D'],
            'BIDTYPE': ['ENERGY', 'LOWERREG', 'LOWERCON', 'ENERGY', 'LOWERCON']
        })
        output_frame, duid_list = joint_fcas_energy_constraints.get_units_to_constrain_joint_capacity(
            bid_type_check, bids_and_indexes, 'LOWERCON', 'LOWERREG')
        assert_frame_equal(expected_output.reset_index(drop=True), output_frame.reset_index(drop=True))
        self.assertListEqual(['A', 'D'], list(duid_list))

    def test_set_rows(self):
        units_to_constrain = pd.DataFrame({
            'DUID': ['X1', 'X1', 'X2'],
            'INDEX': [1, 2, 3],
            'BIDTYPE': ['ENERGY', 'LOWERREG', 'LOWERCON'],
            'LOWERSLOPE': [0.1, 0.1, 0.1],
            'ENABLEMENTMIN': [0, 0, 0],
            'LHSCOEFFICIENTS': [1.0, -1.0, -0.1],
            'RHSCONSTANT': [0, 0, 0],
            'CONSTRAINTTYPE': ['>=', '>=', '>=']
        })
        duids = ['X1', 'X2']
        max_con = 0
        expected_output = pd.DataFrame({
            'INDEX': [1, 2, 3],
            'ROWINDEX': [1, 1, 2],
            'LHSCOEFFICIENTS': [1.0, -1.0, -0.1],
            'CONSTRAINTTYPE': ['>=', '>=', '>='],
            'RHSCONSTANT': [0, 0, 0]
        })
        output = joint_fcas_energy_constraints.set_row_index(units_to_constrain, duids, max_con)
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))


class TestCreateJointCapacityConstraintsLowerSlope(unittest.TestCase):
    def test_calc_slope(self):
        input = pd.DataFrame({
            'DUID': ['X1', 'X1'],
            'BIDTYPE': ['A', 'D'],
            'LOWBREAKPOINT': [1, 1],
            'ENABLEMENTMIN': [0, 0],
            'MAXAVAIL': [10, 20]
        })
        expected_output = pd.DataFrame({
            'DUID': ['X1'],
            'LOWERSLOPE': [0.1],
            'ENABLEMENTMIN': [0]
        })
        output = joint_fcas_energy_constraints.calc_slope_joint_capacity_lower(input, 'A')
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))

    def test_define_constraint_values_lower_slope(self):
        lower_slope_cofficients = pd.DataFrame({
            'DUID': ['X1'],
            'LOWERSLOPE': [0.1],
            'ENABLEMENTMIN': [0]
        })
        unit_to_constraint = pd.DataFrame({
            'DUID': ['X1', 'X1', 'X1'],
            'BIDTYPE': ['ENERGY', 'LOWERREG', 'LOWERCON']
        })
        expected_output = pd.DataFrame({
            'DUID': ['X1', 'X1', 'X1'],
            'BIDTYPE': ['ENERGY', 'LOWERREG', 'LOWERCON'],
            'LOWERSLOPE': [0.1, 0.1, 0.1],
            'ENABLEMENTMIN': [0, 0, 0],
            'LHSCOEFFICIENTS': [1.0, -1.0, -0.1],
            'RHSCONSTANT': [0, 0, 0],
            'CONSTRAINTTYPE': ['>=', '>=', '>=']
        })
        output = joint_fcas_energy_constraints.define_joint_capacity_constraint_values_lower_slope(
            unit_to_constraint, lower_slope_cofficients)
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))


class TestCreateJointCapacityConstraintsUpperSlope(unittest.TestCase):
    def test_calc_slope(self):
        input = pd.DataFrame({
            'DUID': ['X1', 'X1'],
            'BIDTYPE': ['A', 'D'],
            'HIGHBREAKPOINT': [1, 1],
            'ENABLEMENTMAX': [0, 0],
            'MAXAVAIL': [10, 20]
        })
        expected_output = pd.DataFrame({
            'DUID': ['X1'],
            'UPPERSLOPE': [-0.1],
            'ENABLEMENTMAX': [0]
        })
        output = joint_fcas_energy_constraints.calc_slope_joint_capacity_upper(input, 'A')
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))

    def test_define_constraint_values_upper_slope(self):
        upper_slope_cofficients = pd.DataFrame({
            'DUID': ['X1'],
            'UPPERSLOPE': [0.1],
            'ENABLEMENTMAX': [0]
        })
        unit_to_constraint = pd.DataFrame({
            'DUID': ['X1', 'X1', 'X1'],
            'BIDTYPE': ['ENERGY', 'RAISEREG', 'RAISECON']
        })
        expected_output = pd.DataFrame({
            'DUID': ['X1', 'X1', 'X1'],
            'BIDTYPE': ['ENERGY', 'RAISEREG', 'RAISECON'],
            'UPPERSLOPE': [0.1, 0.1, 0.1],
            'ENABLEMENTMAX': [0, 0, 0],
            'LHSCOEFFICIENTS': [1.0,  1.0, 0.1],
            'RHSCONSTANT': [0, 0, 0],
            'CONSTRAINTTYPE': ['<=', '<=', '<=']
        })
        output = joint_fcas_energy_constraints.define_joint_capacity_constraint_values_upper_slope(
            unit_to_constraint, upper_slope_cofficients)
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))


class TestJointEnergyAndRegConstraints(unittest.TestCase):
    def test_joint_energy_and_reg_get_units_to_constrain(self):
        bid_type_check = pd.DataFrame({
            'DUID': ['A', 'B', 'C'],
            'ENERGY': [1, 1, 0],
            'LOWERREG': [1, 0, 1],
        })
        bids_and_indexes = pd.DataFrame({
            'DUID': ['A', 'A', 'B', 'C'],
            'BIDTYPE': ['ENERGY', 'LOWERREG', 'ENERGY', 'LOWERREG']
        })
        expected_output = pd.DataFrame({
            'DUID': ['A', 'A'],
            'BIDTYPE': ['ENERGY', 'LOWERREG']
        })
        output_frame, duid_list = joint_fcas_energy_constraints.joint_energy_and_reg_get_units_to_constrain(
            bids_and_indexes, 'LOWERREG', bid_type_check)
        assert_frame_equal(expected_output.reset_index(drop=True), output_frame.reset_index(drop=True))
        self.assertListEqual(['A'], list(duid_list))

    def test_calc_slope(self):
        input = pd.DataFrame({
            'DUID': ['X1', 'X1'],
            'BIDTYPE': ['A', 'D'],
            'LOWBREAKPOINT': [1, 1],
            'ENABLEMENTMIN': [0, 0],
            'HIGHBREAKPOINT': [1, 1],
            'ENABLEMENTMAX': [0, 0],
            'MAXAVAIL': [10, 20]
        })
        expected_output = pd.DataFrame({
            'DUID': ['X1'],
            'UPPERSLOPE': [-0.1],
            'LOWERSLOPE': [0.1],
            'ENABLEMENTMAX': [0],
            'ENABLEMENTMIN': [0]
        })
        output = joint_fcas_energy_constraints.joint_energy_and_reg_slope_coefficients(input, 'A', ['X1'])
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))

    def test_define_constraint_values_upper_slope(self):
        unit_to_constraint = pd.DataFrame({
            'DUID': ['X1', 'X1'],
            'INDEX': [1, 2],
            'BIDTYPE': ['ENERGY', 'RAISEREG'],
            'UPPERSLOPE': [0.1, 0.1],
            'ENABLEMENTMAX': [0, 0]
        })
        expected_output = pd.DataFrame({
            'INDEX': [1, 2],
            'ROWINDEX': [1, 1],
            'LHSCOEFFICIENTS': [1.0,  0.1],
            'CONSTRAINTTYPE': ['<=', '<='],
            'RHSCONSTANT': [0, 0]

        })
        output = joint_fcas_energy_constraints.joint_energy_and_reg_upper_slope_constraints(
            unit_to_constraint, 'RAISEREG', 0)
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))

    def test_define_constraint_values_lower_slope(self):
        unit_to_constraint = pd.DataFrame({
            'DUID': ['X1', 'X1'],
            'INDEX': [1, 2],
            'BIDTYPE': ['ENERGY', 'LOWERREG'],
            'LOWERSLOPE': [0.1, 0.1],
            'ENABLEMENTMIN': [0, 0]
        })
        expected_output = pd.DataFrame({
            'INDEX': [1, 2],
            'ROWINDEX': [1, 1],
            'LHSCOEFFICIENTS': [1.0,  -0.1],
            'CONSTRAINTTYPE': ['>=', '>='],
            'RHSCONSTANT': [0, 0]

        })
        output = joint_fcas_energy_constraints.joint_energy_and_reg_lower_slope_constraints(
            unit_to_constraint, 'LOWERREG', 0)
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))

