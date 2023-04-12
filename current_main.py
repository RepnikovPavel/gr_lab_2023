import torch
import de_config
import config
from pprint import pprint

from gen_des import F_vec

from gen_F import make_params_vec_from_params_dict, get_start_point_values

if __name__ == '__main__':
    to_new_var_name = torch.load(config.map_from_old_param_name_to_new_name_dict_filename)
    params_vec = make_params_vec_from_params_dict(source_params_values=de_config.params_values,
                                                  from_source_param_name_to_new=to_new_var_name)
    to_new_y_name = torch.load(config.map_from_old_y_name_to_new_name_dict_filename)

    start_point = get_start_point_values(source_y_dict_with_start_point=de_config.start_point,
                                         from_source_y_name_to_new=to_new_y_name)

    F_func = F_vec




