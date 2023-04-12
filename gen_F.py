import json
import os
import re
from pprint import pprint
from typing import Dict, List, Any, Tuple
from copy import deepcopy

import numpy as np
import sympy
from sympy.parsing.latex import parse_latex

from django.template import Template, Context
from django.conf import settings
from django.template.loader import get_template
import django

import de_config
import mysite.settings

import torch

from support_func import create_dir_if_not_exist

import config

def get_all_Y_names(des_str: Dict[str, str]):
    return list(des_str.keys())


def build_Y_name_mapping(Y_names: List[str]):
    index = 0
    map_to_Y = {}
    map_to_name = {}
    for name in Y_names:
        y_name = 'y_{'+str(index)+'}'
        map_to_Y.update({name: y_name})
        map_to_name.update({y_name: name})
        index += 1
    return map_to_Y, map_to_name


def try_replace(source_str: str, mapping_dict: Dict[str, str]):
    output = deepcopy(source_str)
    # is key exists in source_str
    for key_ in mapping_dict:
        v_ = mapping_dict[key_]
        if key_ in source_str:
            output = output.replace(key_, v_)

    return output


def replace_substrings(dict_: Dict[str, str], mapping_dict: Dict[str, str]):
    o_dict = {}
    for key_ in dict_:
        # try to replace
        k = key_
        v = dict_[key_]
        replaced_value = try_replace(v, mapping_dict=mapping_dict)
        replaced_key = try_replace(k, mapping_dict=mapping_dict)

        o_dict.update({replaced_key: replaced_value})

    return o_dict

def get_all_variables(dict_:Dict[str,str],free_functions_: List[str], y_functions_:List[str]):
    # на входе строки без кода latex
    # разбить строку по математическим символам
    # достать все, что не является свободной функцией и Y-ом
    for key_ in dict_:
        k = key_
        v = dict_[key_]


def substitute_aliases(where_to_substitute: Dict[str, str], what_to_substitute: Dict[str,str]):
    output = {}
    for key_ in where_to_substitute:
        value_to_be_replaced = where_to_substitute[key_]
        replaced_value = try_replace(value_to_be_replaced, what_to_substitute)
        output.update({key_: replaced_value})
    return output


def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]


def build_var_name_mapping(deqs_: Dict[str,str]):
    variables = []
    for key_ in deqs_:
        y_i = key_
        f_i = deqs_[key_]
        positions = findOccurrences(s=f_i,ch='$')
        if len(positions)%2!=0:
            print('error')
        for i in range(0, len(positions), 2):
            pos1 = positions[i]
            pos2 = positions[i+1]
            variable = f_i[pos1:pos2+1]
            variables.append(variable)
    variables = np.unique(variables).tolist()

    k_ = 0
    from_source_to_new = {}
    from_new_to_source = {}
    for var in variables:
        new_ = r'a_'+'{'+str(k_)+'}'
        from_source_to_new.update({var:new_})
        from_new_to_source.update({new_:var})
        k_+=1

    return from_source_to_new, from_new_to_source


class HtmlGenerator:
    t: None

    __list_of_equations: List[str]

    def __init__(self):
        settings.configure(TEMPLATES=mysite.settings.TEMPLATES)
        django.setup()
        self.t = get_template('frontend_template.html')
        self.__list_of_equations = []
    def write_to_html(self, string_to_write):
        self.__list_of_equations.append(string_to_write)
        # c = Context({"equations": "equations here"})
    def render(self):
        str_to_render = ''
        for el in self.__list_of_equations:
            str_to_render += el

        rendered = self.t.render({"equations": str_to_render})
        file1 = open('frontend/templates/frontend.html', "w")
        file1.write(rendered)
        file1.close()



def get_latex_view_of_dict(dict_: Dict[str, str]):

    clear_latex = {}

    for k,v in dict_.items():
        clear_k = k.replace('#', '')
        clear_k = clear_k.replace('$', '')
        clear_k = clear_k.replace('@', '')
        clear_v = v.replace('#', '')
        clear_v = clear_v.replace('$', '')
        clear_v = clear_v.replace('@', '')
        clear_latex.update({clear_k:clear_v})

    latex_str = ''
    for k, v in clear_latex.items():
        eq_ = '$$'
        eq_ += r'\frac{d}{dt}'
        eq_ += k
        eq_ += '='
        eq_ += v
        eq_ += '$$'
        latex_str += eq_
        latex_str += '\n'
    return latex_str


def get_substitutions(latex_str_, sys_funcs)->Tuple[str, List[Dict[str,str]]]:

    all_functions_tags_symbols = findOccurrences(s=latex_str_, ch='#')

    if len(all_functions_tags_symbols) == 0:
        return ['',[]]

    detected_functions = []
    for i in range(0, len(all_functions_tags_symbols), 2):
        pos1 = all_functions_tags_symbols[i]
        pos2 = all_functions_tags_symbols[i + 1]
        variable = latex_str_[pos1:pos2 + 1]
        detected_functions.append(variable)

    copy_of_latex_str = deepcopy(latex_str_)

    substitutions = []

    k_ = 0
    for free_function_name in sys_funcs:
        for detected_function in detected_functions:
            if ('#'+free_function_name) in detected_function:
                # нашли совпадение того, что есть в latex строке и в словаре функций
                # вырезать функцию, ее аргументы
                # {'py_name': 'sigmoid', 'params': ''}
                # вырезать параметры latex
                copy_of_detected_func = deepcopy(detected_function)
                copy_of_detected_func = copy_of_detected_func.replace('#', '')
                copy_of_detected_func = copy_of_detected_func.replace('(', '')
                copy_of_detected_func = copy_of_detected_func.replace(')', '')
                copy_of_detected_func = copy_of_detected_func.replace(free_function_name, '')
                latex_args_of_one_detected_function = copy_of_detected_func.split(',')
                code_gen_for_latex_str = {
                    'latex_name': 'f_{'+str(k_)+'}',
                    'py_name': sys_funcs[free_function_name]['py_name'],
                    'latex_args': latex_args_of_one_detected_function
                }
                substitutions.append(code_gen_for_latex_str)
                copy_of_latex_str = copy_of_latex_str.replace(detected_function, 'f_{'+str(k_)+'}')
                k_ += 1

    return copy_of_latex_str, substitutions

def find_all_symbol_i(symbol, py_code_str: str):
    start_stop_positions_of_e_i = []
    for i in range(len(py_code_str)-2):
        ch = py_code_str[i]
        if ch == symbol and py_code_str[i+1] == '_':
            len_of_numer = 0
            last_index = i+2
            for j in range(i+2, len(py_code_str)):
                if not py_code_str[j].isdigit():
                    break
                else:
                    len_of_numer += 1
                    last_index = j
            if len_of_numer > 0:
                start_stop_positions_of_e_i.append([i, last_index])


    return start_stop_positions_of_e_i


def insert_str_instead_of_a_substr(source_str,
                                   start_stop_positions_to_insert,
                                   what_to_insert):
    i1 =start_stop_positions_to_insert[0]
    i2= start_stop_positions_to_insert[1]
    left_side = source_str[:i1]
    right_side = source_str[i2+1:]
    middle = what_to_insert
    output = left_side+middle+right_side
    return output


def latex_str_to_py_code(latex_str,
                         substitutions):
    copy_of_latex_str = deepcopy(latex_str)
    #get_code_of_source_str
    # F_i = parse_latex(f_)
    # F_i_code = sympy.pycode(F_i)
    sympy_expr_of_source_str = parse_latex(latex_str)
    sympy_expr_substitutions = [parse_latex(el['latex_name']) for el in substitutions]
    py_code_of_source_str = sympy.pycode(sympy_expr_of_source_str)
    py_code_of_substitutions = [sympy.pycode(el) for el in sympy_expr_substitutions]
    # тут тоже может возникнуть ошибка с подстановкой, например, e_1, e_11 и им подобные
    # нужно определить все выражения вида e_number
    start_stop_positions = find_all_symbol_i('f', py_code_of_source_str)
    i_=0
    while len(start_stop_positions) != 0:
        start_stop = start_stop_positions[0]
        # вряд ли заведется
        function_args = [sympy.pycode(parse_latex(el)) for el in substitutions[i_]['latex_args']]
        if len(function_args)==0:
            print('error')
            print('задали функцию и не указали или не передали аргумент')
            raise SystemExit
        function_args_str = ''
        for arg_i in range(len(function_args)-1):
            arg = function_args[arg_i]
            function_args_str += arg
            function_args_str += ','
        function_args_str += function_args[-1]

        what_to_insert = substitutions[i_]['py_name']+'('+function_args_str+')'
        py_code_of_source_str = insert_str_instead_of_a_substr(py_code_of_source_str, start_stop, what_to_insert)
        start_stop_positions = find_all_symbol_i('f', py_code_of_source_str)
        i_+=1

    #get_code_of_letex_names from list
    #replace

    return py_code_of_source_str

def get_index_of_y(y_latex_name):
    # y_{index}
    return y_latex_name[3:-1]

def get_index_of_param(param_py_code):
    # a_index
    return param_py_code[2:]

def replace_parameters_with_an_array_element(py_code):
    py_code_copy = deepcopy(py_code)
    start_stop_positions = find_all_symbol_i('a', py_code_copy)
    while len(start_stop_positions) != 0:
        start_stop_pair = start_stop_positions[0]
        i1 = start_stop_pair[0]
        i2 = start_stop_pair[1]
        a_str = py_code_copy[i1:i2+1]
        a_index = get_index_of_param(a_str)
        py_code_copy = insert_str_instead_of_a_substr(py_code_copy, [i1,i2], what_to_insert='param_vec['+str(a_index)+']')
        start_stop_positions = find_all_symbol_i('a', py_code_copy)
    return py_code_copy

def replace_y_with_an_array_element(py_code):
    py_code_copy = deepcopy(py_code)
    start_stop_positions = find_all_symbol_i('y', py_code_copy)
    while len(start_stop_positions) != 0:
        start_stop_pair = start_stop_positions[0]
        i1 = start_stop_pair[0]
        i2 = start_stop_pair[1]
        y_str = py_code_copy[i1:i2+1]
        y_index = get_index_of_param(y_str)
        py_code_copy = insert_str_instead_of_a_substr(py_code_copy, [i1,i2], what_to_insert='y_vec['+str(y_index)+']')
        start_stop_positions = find_all_symbol_i('y', py_code_copy)
    return py_code_copy



def gen_python_code(gen_file: str, des_dict_: Dict[str, str], sys_funcs, from_y_name_to_source_name: Dict[str, str]):

    # получить словарь с уравнениями вида {"index of equation": F_i(t,y_vec,...)}
    equations = {}
    # получить словарь вида {"index of equation": "source_y_name"}
    get_source_name_by_index_of_equation = {}


    number_of_equations = 0
    for y, f in des_dict_.items():
        y_ = y
        index_of_equation = get_index_of_y(y_)
        source_name = from_y_name_to_source_name[y_]
        f_ = f
        latex_str, substitutions = get_substitutions(f_, sys_funcs)
        code = ''
        if len(latex_str) == 0:
            code = sympy.pycode(parse_latex(f_))
        else:
            code = latex_str_to_py_code(latex_str, substitutions)
        code = replace_parameters_with_an_array_element(py_code=code)
        code = replace_y_with_an_array_element(py_code=code)
        key_ = index_of_equation
        value_ = code
        equations.update({key_: value_})
        get_source_name_by_index_of_equation.update({index_of_equation: source_name})
        number_of_equations += 1

    # генерирование строки для подключения определнных где-то функций
    import_str = ''
    import_str += 'import numpy as np\n'
    for k, v in sys_funcs.items():
        module = v['module']
        py_name = v['py_name']
        import_str += 'from '+module+' import '+py_name+'\n'

    F_str = '\n\n'
    F_str += 'def F_vec(y_vec,t, param_vec):\n'
    F_str += '\tbuffer = np.zeros(shape=('+str(number_of_equations)+',))\n'
    for index_of_equation, right_side_of_eq in equations.items():
        F_str += '\t'+ 'buffer['+index_of_equation+']'+' = '+right_side_of_eq+'\n'
    F_str += '\treturn buffer'

    with open(gen_file, 'w') as file:
        file.write(import_str)
        file.write(F_str)

def get_index_of_latex_param(latex_str):
    # a_{number}
    return latex_str[3:-1]

def make_params_vec_from_params_dict(source_params_values:Dict[str, float], from_source_param_name_to_new:Dict[str,str]):
    number_of_params = len(source_params_values)
    params_vec = np.zeros(shape=(number_of_params, ))
    for k, v in source_params_values.items():
        source_latex_name = k
        new_latex_name = from_source_param_name_to_new[source_latex_name]
        index_of_param = get_index_of_latex_param(new_latex_name)
        params_vec[int(index_of_param)] = v
    return params_vec

def get_start_point_values(source_y_dict_with_start_point:Dict[str,float], from_source_y_name_to_new:Dict[str,str]):
    number_of_y = len(source_y_dict_with_start_point)
    start_point = np.zeros(shape=(number_of_y , ))
    for k, v in source_y_dict_with_start_point.items():
        source_latex_name = k
        new_latex_name = from_source_y_name_to_new[source_latex_name]
        index_of_y = get_index_of_y(new_latex_name)
        start_point[int(index_of_y)] = v
    return start_point


def get_y_latex_name_from_index(index: int):
    index = str(index)
    return 'y_{'+index+'}'

def get_y_name_by_index_of_solution(index_of_solution:int, from_new_to_old_y_names_dict):
    y_latex_name = get_y_latex_name_from_index(index_of_solution)
    for k, v in from_new_to_old_y_names_dict.items():
        if v==y_latex_name:
            return k




if __name__ == '__main__':
    aliases_str, des_str = de_config.aliases_, de_config.des_str_
    des_str = substitute_aliases(des_str, aliases_str)
    source_des = deepcopy(des_str)
    map_to_Y_name, map_to_source_name = build_Y_name_mapping(get_all_Y_names(des_str))
    des_str = replace_substrings(des_str, map_to_Y_name)
    to_new_var_name, to_source_var_name = build_var_name_mapping(des_str)
    #
    # # pprint(map_to_Y_name)
    # pprint(to_new_var_name)
    # raise SystemExit

    des_str = replace_substrings(des_str, to_new_var_name)
    # pprint(des_str)
    gen_python_code(gen_file='gen_des.py',
                    des_dict_=des_str,
                    sys_funcs=de_config.system_functions,
                    from_y_name_to_source_name=map_to_source_name)
    create_dir_if_not_exist(config.translators_of_names_path)
    torch.save(obj=to_new_var_name, f=config.map_from_old_param_name_to_new_name_dict_filename)
    torch.save(obj=map_to_Y_name, f=config.map_from_old_y_name_to_new_name_dict_filename)



    # params_vec = make_params_vec_from_params_dict(source_params_values=de_config.params_values,
    #                                              from_source_param_name_to_new=to_new_var_name)


    # pprint(des_str)
    # latex_str = get_latex_view_of_dict(des_str)
    latex_str = get_latex_view_of_dict(source_des)


    view = HtmlGenerator()
    view.write_to_html(latex_str)
    view.render()
