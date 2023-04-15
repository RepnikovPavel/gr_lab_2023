import os
from support_func import create_dir_if_not_exist
problem_folder = 'C:/ode_dir'
translators_of_names_path = os.path.join(problem_folder, 'translators')
map_from_old_param_name_to_new_name_dict_filename = os.path.join(translators_of_names_path, 'map_from_old_param_name_to_new_name_dict.txt')
map_from_old_y_name_to_new_name_dict_filename = os.path.join(translators_of_names_path, 'map_from_old_y_name_to_new_name_dict.txt')


plotly_base_path = './frontend/plotly/'
plotly_plotting_html = './frontend/plotly/plotly_.html'
create_dir_if_not_exist(plotly_base_path)