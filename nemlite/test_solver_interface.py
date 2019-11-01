import unittest
import pandas as pd
from nemlite import solver_interface

class TestOneRegionTwoBids(unittest.TestCase):
    def setUp(self):
        self.number_of_con = 10
        self.bid_bounds = pd.DataFrame.from_dict({
            'BID':[50, 50],
            'INDEX':[1, 2],
            'CAPACITYBAND':['ENERGY', 'ENERGY'],
        })
        self.inter_bounds = pd.DataFrame.from_dict({
            'INDEX':[],
            'UPPERBOUND':[]
        })
        self.constraint_matrix = pd.DataFrame.from_dict({
            1: [1],
            2: [1]
        })
        self.objective_coefficients = (10, 20)
        self.enquality_types = []
        self.row_rhs_values = pd.DataFrame({
            'ROWINDEX':[1],
            'RHSCONSTANT':[75],
            'ENQUALITYTYPE':['equal_or_greater']
        })
        self.indices = []
        self.region_req_by_row = pd.DataFrame.from_dict({
            'ROWINDEX':[1],
            'BIDTYPE':['ENEGY'],
            'REGIONID':['BIG']
        })
        self.regions_to_price = ['BIG']


    def test_dispatch_tables_start_of_month(self):
        dispatch, inter_flow = solver_interface.solve_lp(number_of_constraints=self.number_of_con,
                                                         bid_bounds=self.bid_bounds,
                                                         inter_bounds=self.inter_bounds,
                                                         constraint_matrix=self.constraint_matrix,
                                                         objective_coefficients=self.objective_coefficients,
                                                         row_rhs_values=self.row_rhs_values,
                                                         inequality_types=self.enquality_types)