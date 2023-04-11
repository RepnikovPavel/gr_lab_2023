from pprint import pprint
from typing import Dict, List, Any, Tuple
from copy import deepcopy

import de_config


def get_all_Y_names(des_str: Dict[str, str]):
    return list(des_str.keys())


def build_Y_name_mapping(Y_names: List[str]):
    index = 0
    map_to_Y = {}
    map_to_name = {}
    for name in Y_names:
        map_to_Y.update({name: 'y_' + str(index)})
        map_to_name.update({'y_' + str(index): name})
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

if __name__ == '__main__':
    aliases_str, des_str = de_config.aliases_, de_config.des_str_
    map_to_Y_name, map_to_source_name = build_Y_name_mapping(get_all_Y_names(des_str))
    des_str = substitute_aliases(des_str, aliases_str)
    des_str = replace_substrings(des_str, map_to_Y_name)
    pprint(des_str)
