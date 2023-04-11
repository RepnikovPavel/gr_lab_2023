import json
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
    latex_str = ''
    for k, v in dict_.items():
        eq_ = '$$'
        eq_ += r'\frac{d}{dt}'
        eq_ += k
        eq_ += '='
        eq_ += v
        eq_ += '$$'
        latex_str += eq_
        latex_str += '\n'
    return latex_str


def replace_functions_in_latex_str(latex_str_, free_functions, external_links):
    output_latex_str = ''
    function_args = ''
    all_functions_tags_symbols = findOccurrences(s=latex_str_, ch='#')

    if len(all_functions_tags_symbols) == 0:
        return output_latex_str,function_args

    detected_functions = []
    for i in range(0, len(all_functions_tags_symbols), 2):
        pos1 = all_functions_tags_symbols[i]
        pos2 = all_functions_tags_symbols[i + 1]
        variable = latex_str_[pos1:pos2 + 1]
        detected_functions.append(variable)

    k_ = 0
    for free_function_name in free_functions:
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
                    'latex_name': 'e_{'+str(k_)+'}',
                    'py_name': free_functions[free_function_name]['py_name'],
                    'latex_args': latex_args_of_one_detected_function
                }
                k_ += 1


        # if free_function_name in latex_str_:
        #     print(free_function_name)
        #     print(latex_str)
    return output_latex_str, function_args

def gen_python_code(gen_file: str, des_dict_: Dict[str, str], free_functions, external_links):
    Y_i_dict = {}
    F_i_dict = {}
    Y_0_dict = {}
    k_ = 0
    for y, f in des_dict_.items():
        y_ = y
        f_ = f
        f_, function_args = replace_functions_in_latex_str(f_,free_functions,external_links)
        # F_i = parse_latex(f_)
        # F_i_code = sympy.pycode(F_i)
        # Y_i = parse_latex(y_)
        # Y_i_code = sympy.pycode(Y_i)
        # Y_i_dict.update({str(k_): Y_i_code})
        # F_i_dict.update({str(k_): F_i_code})
        # k_ += 1
    #
    # with open(gen_file, 'w') as file:
    #     file.write('Y = {\n')
    #     for k, v in Y_i_dict.items():
    #         file.write('\t'+k +':\t'+v+',\n')
    #     file.write('}\n')
    #     file.write('F = {\n')
    #     for k, v in F_i_dict.items():
    #         file.write('\t'+k +':\t'+v+',\n')
    #     file.write('}\n')


        # file.write(json.dumps([Y_i_dict, F_i_dict]))




if __name__ == '__main__':
    aliases_str, des_str = de_config.aliases_, de_config.des_str_
    des_str = substitute_aliases(des_str, aliases_str)
    map_to_Y_name, map_to_source_name = build_Y_name_mapping(get_all_Y_names(des_str))
    des_str = replace_substrings(des_str, map_to_Y_name)
    to_new_var_name, to_source_var_name = build_var_name_mapping(des_str)

    des_str = replace_substrings(des_str, to_new_var_name)
    pprint(des_str)
    gen_python_code(gen_file='gen_des.py',
                    des_dict_=des_str,
                    free_functions=de_config.free_functions,
                    external_links=de_config.external_links)

    # pprint(des_str)
    latex_str = get_latex_view_of_dict(des_str)



    view = HtmlGenerator()
    view.write_to_html(latex_str)
    view.render()
