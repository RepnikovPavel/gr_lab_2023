from ParsingSystem.parse_and_build import *
import myocyte_de_config
import os

from support_func import create_dir_if_not_exist
import config as config
import torch

if __name__ == '__main__':
    aliases_str, des_str = myocyte_de_config.aliases_, myocyte_de_config.des_str_
    des_str = substitute_aliases(des_str, aliases_str)
    source_des = deepcopy(des_str)
    map_to_Y_name, map_to_source_name = build_Y_name_mapping(get_all_Y_names(des_str))
    des_str = replace_substrings(des_str, map_to_Y_name)
    to_new_var_name, to_source_var_name = build_var_name_mapping(des_str)
    
    # # pprint(map_to_Y_name)
    # pprint(to_new_var_name)
    # raise SystemExit

    des_str = replace_substrings(des_str, to_new_var_name)
    # pprint(des_str)
    gen_python_code(gen_file = os.path.join(myocyte_de_config.write_generated_code_to,
                                            myocyte_de_config.name_of_generated_pyhton_file),
                    des_dict_ = des_str,
                    sys_funcs = myocyte_de_config.system_functions,
                    from_y_name_to_source_name=map_to_source_name)
    create_dir_if_not_exist(config.myocyte_translators_of_names_path)
    torch.save(obj=to_new_var_name, f=config.myocyte_map_from_old_param_name_to_new_name_dict_filename)
    torch.save(obj=map_to_Y_name, f=config.myocyte_map_from_old_y_name_to_new_name_dict_filename)



    # params_vec = make_params_vec_from_params_dict(source_params_values=de_config.params_values,
    #                                              from_source_param_name_to_new=to_new_var_name)


    # pprint(des_str)
    # latex_str = get_latex_view_of_dict(des_str)
    latex_str,latex_equations = get_latex_view_of_dict(source_des)
    torch.save(latex_equations,config.latex_eq_path)


    view = HtmlGenerator()
    view.write_to_html(latex_str)
    view.render(path_to_rendered_html=os.path.join(myocyte_de_config.write_latex_view_of_system_to,'equations.html'))
