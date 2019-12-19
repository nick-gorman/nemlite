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
