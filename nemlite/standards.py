
class Naming:
    def __init__(self, base_filename, col_datetime_applicable: str, col_date_applicable: str,
                 col_date_changed: str, cols_bid_cap_name_list: list,
                 cols_bid_price_name_list: list, col_unit_name: str, col_unit_max_output: str, col_enablement_min: str,
                 col_enablement_max: str, col_bid_type: str, col_capacity_band_number: str, col_price_band_number: str,
                 col_bid_value: str,
                 col_enablement_type: str, col_enablement_value: str, col_low_break_point: str,
                 col_high_break_point: str, col_lhs_coefficients: str, col_rhs_constant: str, col_upper_bound: str,
                 col_lower_bound: str, col_variable_index: str, col_energy_variable_index: str,
                 col_max_unit_energy: str, col_constraint_row_index: str, col_fcas_integer_variable: str,
                 type_energy: str, col_inter_id: str, col_import_limit: str, col_export_limit: str,
                 col_fcas_import_limit: str, col_fcas_export_limit: str, list_fcas_types: list, type_fcas: str,
                 col_gen_req: str, col_lower_5min_req: str, col_lower_60sec_req: str, col_lower_6sec_req: str,
                 col_raise_5min_req: str, col_raise_60sec_req: str, col_raise_6sec_req: str, col_lower_reg_req: str,
                 col_raise_req_req: str, col_region_id: str, col_region_constraint_type: str,
                 col_region_constraint_value: str, col_end_date: str, col_contribution_coefficients: str,
                 col_region_from: str, col_region_to: str, col_direction: str, col_limit_value,
                 col_enquality_type: str, col_price: str, col_loss_factor: str, col_dispatch_type: str, type_load:str,
                 type_gen: str, col_pool_id: str):

        self.base_filename = base_filename
        self.col_datetime_applicable = col_datetime_applicable
        self.col_date_applicable = col_date_applicable
        self.col_date_changed = col_date_changed
        self.cols_bid_cap_name_list = cols_bid_cap_name_list
        self.cols_bid_price_name_list = cols_bid_price_name_list
        self.col_unit_name = col_unit_name
        self.col_unit_max_output = col_unit_max_output
        self.col_enablement_min = col_enablement_min
        self.col_enablement_max = col_enablement_max
        self.col_bid_type = col_bid_type
        self.col_capacity_band_number = col_capacity_band_number
        self.col_price_band_number = col_price_band_number
        self.col_bid_value = col_bid_value
        self.col_enablement_type = col_enablement_type
        self.col_enablement_value = col_enablement_value
        self.col_low_break_point = col_low_break_point
        self.col_high_break_point = col_high_break_point
        self.col_lhs_coefficients = col_lhs_coefficients
        self.col_rhs_constant = col_rhs_constant
        self.col_upper_bound = col_upper_bound
        self.col_lower_bound = col_lower_bound
        self.col_variable_index = col_variable_index
        self.col_energy_variable_index = col_energy_variable_index
        self.col_max_unit_energy = col_max_unit_energy
        self.col_constraint_row_index = col_constraint_row_index
        self.col_fcas_integer_variable = col_fcas_integer_variable
        self.type_energy = type_energy
        self.type_fcas = type_fcas
        self.col_inter_id = col_inter_id
        self.col_import_limit = col_import_limit
        self.col_export_limit = col_export_limit
        self.col_fcas_import_limit = col_fcas_import_limit
        self.col_fcas_export_limit = col_fcas_export_limit
        self.list_region_req_cols = [col_gen_req, col_lower_5min_req, col_lower_60sec_req, col_lower_6sec_req,
                                     col_raise_5min_req, col_raise_60sec_req, col_raise_6sec_req, col_lower_reg_req,
                                     col_raise_req_req]
        self.list_fcas_types = list_fcas_types
        self.col_region_id = col_region_id
        self.col_region_constraint_type = col_region_constraint_type
        self.col_region_constraint_value = col_region_constraint_value
        self.req_suffix = 'LOCALDISPATCH'
        self.col_end_date = col_end_date
        self.col_contribution_coefficients = col_contribution_coefficients
        self.col_region_from = col_region_from
        self.col_region_to = col_region_to
        self.col_direction = col_direction
        self.col_gen_req = col_gen_req
        self.col_limit_value = col_limit_value
        self.col_enquality_type = col_enquality_type
        self.col_price = col_price
        self.col_loss_factor = col_loss_factor
        self.col_dispatch_type = col_dispatch_type
        self.type_load = type_load
        self.type_gen = type_gen
        self.col_pool_id = col_pool_id
