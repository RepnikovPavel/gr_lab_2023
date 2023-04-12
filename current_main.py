import torch
import de_config
import config
from pprint import pprint

from gen_des import F_vec

from gen_F import make_params_vec_from_params_dict


if __name__ =='__main__':
    to_new_var_name = torch.load(config.map_from_old_to_new_name_dict_filename)
    params_vec = make_params_vec_from_params_dict(source_params_values=de_config.params_values,
                                                 from_source_param_name_to_new=to_new_var_name)
    F_func = F_vec
