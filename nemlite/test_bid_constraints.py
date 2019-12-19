import unittest
from nemlite import bid_constraints
import pandas as pd
from pandas.util.testing import assert_frame_equal


class TestCreateBiddingIndex(unittest.TestCase):
    def test_2_duid_case(self):
        duid = ['a', 'b', 'b']
        bid_type = ['ENERGY', 'ENERGY', 'RAISE60SEC']
        b1 = [10, 0, 0]
        b2 = [0, 0, 0]
        b3 = [6, 0, 0]
        b4 = [0, 0, 7]
        b5 = [0, 0, 0]
        b6 = [0, 8, 9]
        b7 = [0, 0, 0]
        b8 = [0, 0, 0]
        b9 = [11, 0, 0]
        b10 = [0, 0, 1]
        input_data = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'BANDAVAIL1': b1, 'BANDAVAIL2': b2,
                                   'BANDAVAIL3': b3, 'BANDAVAIL4': b4, 'BANDAVAIL5': b5, 'BANDAVAIL6': b6,
                                   'BANDAVAIL7': b7, 'BANDAVAIL8': b8, 'BANDAVAIL9': b9, 'BANDAVAIL10': b10})

        duid = ['a', 'a', 'a', 'b', 'b', 'b', 'b']
        bid_type = ['ENERGY', 'ENERGY', 'ENERGY', 'ENERGY', 'RAISE60SEC', 'RAISE60SEC', 'RAISE60SEC']
        bid = [10, 6, 11, 8, 7, 9, 1]
        capacity_band = ['BANDAVAIL1', 'BANDAVAIL3', 'BANDAVAIL9', 'BANDAVAIL6', 'BANDAVAIL4', 'BANDAVAIL6',
                         'BANDAVAIL10']
        expected_output = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'CAPACITYBAND': capacity_band, 'BID': bid})
        expected_output = expected_output.sort_values(['DUID', 'CAPACITYBAND'])
        expected_output['INDEX'] = [0, 1, 2, 3, 4, 5, 6]

        output = bid_constraints.create_bidding_index(input_data)
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))


class TestCreateConstraints(unittest.TestCase):
    def test_create_min_energy_constraint(self):
        duid = ['a', 'b', 'b']
        bid_type = ['ENERGY', 'ENERGY', 'RAISE60SEC']
        b1 = [10, 0, 0]
        b2 = [0, 0, 0]
        b3 = [6, 0, 0]
        b4 = [0, 0, 7]
        b5 = [0, 0, 0]
        b6 = [0, 8, 9]
        b7 = [0, 0, 0]
        b8 = [0, 0, 0]
        b9 = [11, 0, 0]
        b10 = [0, 0, 1]
        min_energy = [10, 15, 0]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'BANDAVAIL1': b1, 'BANDAVAIL2': b2,
                                   'BANDAVAIL3': b3, 'BANDAVAIL4': b4, 'BANDAVAIL5': b5, 'BANDAVAIL6': b6,
                                   'BANDAVAIL7': b7, 'BANDAVAIL8': b8, 'BANDAVAIL9': b9, 'BANDAVAIL10': b10,
                                   'MINENERGY': min_energy})

        duid = ['a', 'a', 'a', 'b', 'b', 'b', 'b']
        bid_type = ['ENERGY', 'ENERGY', 'ENERGY', 'ENERGY', 'RAISE60SEC', 'RAISE60SEC', 'RAISE60SEC']
        bid = [10, 6, 11, 8, 7, 9, 1]
        capacity_band = ['BANDAVAIL1', 'BANDAVAIL3', 'BANDAVAIL9', 'BANDAVAIL6', 'BANDAVAIL4', 'BANDAVAIL6',
                         'BANDAVAIL10']
        bids_and_indexes = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'CAPACITYBAND': capacity_band, 'BID': bid})
        bids_and_indexes = bids_and_indexes.sort_values(['DUID', 'CAPACITYBAND'])
        bids_and_indexes['INDEX'] = [0, 1, 2, 3, 4, 5, 6]

        row_indexes = [1, 1, 1, 2]
        expected_output = bids_and_indexes[bids_and_indexes['BIDTYPE'] == 'ENERGY']
        expected_output['ROWINDEX'] = row_indexes
        expected_output['ENABLEMENTTYPE'] = 'MINENERGY'
        expected_output['LHSCOEFFICIENTS'] = 1
        expected_output['RHSCONSTANT'] = [10, 10, 10, 15]
        expected_output['ENQUALITYTYPE'] = '>='
        expected_output = expected_output.loc[:, ['INDEX', 'ROWINDEX', 'DUID', 'ENABLEMENTTYPE', 'BIDTYPE',
                                                  'RHSCONSTANT', 'LHSCOEFFICIENTS', 'ENQUALITYTYPE']]
        output = bid_constraints.create_constraints(bids_and_indexes, capacity_bids, bid_types=['ENERGY'],
                                                    max_row_index=0, rhs_col='MINENERGY', direction='>=')
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))

    def test_create_max_fcas_constraint(self):
        duid = ['a', 'b', 'b']
        bid_type = ['ENERGY', 'ENERGY', 'RAISE60SEC']
        b1 = [10, 0, 0]
        b2 = [0, 0, 0]
        b3 = [6, 0, 0]
        b4 = [0, 0, 7]
        b5 = [0, 0, 0]
        b6 = [0, 8, 9]
        b7 = [0, 0, 0]
        b8 = [0, 0, 0]
        b9 = [11, 0, 0]
        b10 = [0, 0, 1]
        min_energy = [10, 15, 55]
        capacity_bids = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'BANDAVAIL1': b1, 'BANDAVAIL2': b2,
                                   'BANDAVAIL3': b3, 'BANDAVAIL4': b4, 'BANDAVAIL5': b5, 'BANDAVAIL6': b6,
                                   'BANDAVAIL7': b7, 'BANDAVAIL8': b8, 'BANDAVAIL9': b9, 'BANDAVAIL10': b10,
                                   'MAXAVAIL': min_energy})

        duid = ['a', 'a', 'a', 'b', 'b', 'b', 'b']
        bid_type = ['ENERGY', 'ENERGY', 'ENERGY', 'ENERGY', 'RAISE60SEC', 'RAISE60SEC', 'RAISE60SEC']
        bid = [10, 6, 11, 8, 7, 9, 1]
        capacity_band = ['BANDAVAIL1', 'BANDAVAIL3', 'BANDAVAIL9', 'BANDAVAIL6', 'BANDAVAIL4', 'BANDAVAIL6',
                         'BANDAVAIL10']
        bids_and_indexes = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'CAPACITYBAND': capacity_band, 'BID': bid})
        bids_and_indexes = bids_and_indexes.sort_values(['DUID', 'CAPACITYBAND'])
        bids_and_indexes['INDEX'] = [0, 1, 2, 3, 4, 5, 6]

        row_indexes = [1, 1, 1]
        expected_output = bids_and_indexes[bids_and_indexes['BIDTYPE'] != 'ENERGY']
        expected_output['ROWINDEX'] = row_indexes
        expected_output['ENABLEMENTTYPE'] = 'MAXAVAIL'
        expected_output['LHSCOEFFICIENTS'] = 1
        expected_output['RHSCONSTANT'] = [55, 55, 55]
        expected_output['ENQUALITYTYPE'] = '<='
        expected_output = expected_output.loc[:, ['INDEX', 'ROWINDEX', 'DUID', 'ENABLEMENTTYPE', 'BIDTYPE',
                                                  'RHSCONSTANT', 'LHSCOEFFICIENTS', 'ENQUALITYTYPE']]
        output = bid_constraints.create_constraints(bids_and_indexes, capacity_bids,
                                                    bid_types=['LOWER5MIN', 'LOWER60SEC', 'LOWER6SEC', 'RAISE5MIN',
                                                               'RAISE60SEC', 'RAISE6SEC', 'LOWERREG', 'RAISEREG'],
                                                    max_row_index=0, rhs_col='MAXAVAIL', direction='<=')
        assert_frame_equal(expected_output.reset_index(drop=True), output.reset_index(drop=True))


class TestCreateBiddingContributionToConstraintMatrix(unittest.TestCase):
    def test_create_bidding_constraints(self):
        duid = ['a', 'b', 'b']
        bid_type = ['ENERGY', 'ENERGY', 'RAISE60SEC']
        max_energy = [45, 55, 65]
        min_energy = [1, 2, 63]
        max_avail = [0, 0, 17]
        b1 = [10, 0, 0]
        b2 = [0, 0, 0]
        b3 = [6, 0, 0]
        b4 = [0, 0, 7]
        b5 = [0, 0, 0]
        b6 = [0, 8, 9]
        b7 = [0, 0, 0]
        b8 = [0, 0, 0]
        b9 = [11, 0, 0]
        b10 = [0, 0, 1]
        input_data = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type, 'BANDAVAIL1': b1, 'BANDAVAIL2': b2,
                                   'BANDAVAIL3': b3, 'BANDAVAIL4': b4, 'BANDAVAIL5': b5, 'BANDAVAIL6': b6,
                                   'BANDAVAIL7': b7, 'BANDAVAIL8': b8, 'BANDAVAIL9': b9, 'BANDAVAIL10': b10,
                                   'MAXENERGY': max_energy, 'MINENERGY': min_energy, 'MAXAVAIL': max_avail})

        duid = ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']
        bid_type = ['ENERGY', 'ENERGY', 'ENERGY', 'ENERGY', 'ENERGY', 'ENERGY', 'ENERGY','ENERGY', 'RAISE60SEC',
                    'RAISE60SEC', 'RAISE60SEC']
        expected_output = pd.DataFrame({'DUID': duid, 'BIDTYPE': bid_type})
        expected_output = expected_output.sort_values(['DUID'])
        expected_output['INDEX'] = [0, 1, 2, 0, 1, 2, 5, 5, 3, 4, 6]
        expected_output['ROWINDEX'] = [1, 1, 1, 3, 3, 3, 2, 4, 5, 5, 5]
        expected_output['ENABLEMENTTYPE'] = ['MAXENERGY', 'MAXENERGY', 'MAXENERGY', 'MINENERGY', 'MINENERGY',
                                             'MINENERGY', 'MAXENERGY', 'MINENERGY', 'MAXAVAIL', 'MAXAVAIL', 'MAXAVAIL']
        expected_output['RHSCONSTANT'] = [45, 45, 45, 1, 1, 1, 55, 2, 17, 17, 17]
        expected_output = expected_output.loc[:, ['INDEX', 'ROWINDEX', 'DUID', 'ENABLEMENTTYPE',
                                             'BIDTYPE', 'RHSCONSTANT']]
        expected_output['LHSCOEFFICIENTS'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        expected_output['ENQUALITYTYPE'] = ['<=', '<=', '<=', '>=', '>=', '>=', '<=', '>=', '<=', '<=', '<=']
        output, _ = bid_constraints.create_bidding_contribution_to_constraint_matrix(input_data)
        assert_frame_equal(expected_output.sort_values(['INDEX', 'ROWINDEX']).reset_index(drop=True),
                           output.sort_values(['INDEX', 'ROWINDEX']).reset_index(drop=True))



