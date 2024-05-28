#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import os
import re
import glob
import json
import argparse


DATA_TPYE_DICT = {
    'float32': 0,
    'float16': 1,
    'int8': 2,
    'int16': 6,
    'uint16': 7,
    'uint8': 4,
    'int32': 3,
    'int64': 9,
    'uint32': 8,
    'uint64': 10,
    'bool': 12,
    'double': 11,
    'complex64': 16,
    'complex128': 17,
    'qint8': 18,
    'qint16': 19,
    'qint32': 20,
    'quint8': 21,
    'quint16': 22,
    'resource': 23,
    'string': 24,
    'dual': 25,
    'variant': 26,
    'bf16': 27,
    'bfloat16': 27,
    'undefined': 28,
    'int4': 29,
    'uint1': 30,
    'int2': 31
}

FORMAT_DICT = {
    'NCHW': 0,
    'NHWC': 1,
    'ND': 2,
    'NC1HWC0': 3,
    'FRACTAL_Z': 4,
    'NC1C0HWPAD': 5,
    'NHWC1C0': 6,
    'FSR_NCHW': 7,
    'FRACTAL_DECONV': 8,
    'C1HWNC0': 9,
    'FRACTAL_DECONV_TRANSPOSE': 10,
    'FRACTAL_DECONV_SP_STRIDE_TRANS': 11,
    'NC1HWC0_C04': 12,
    'FRACTAL_Z_C04': 13,
    'CHWN': 14,
    'FRACTAL_DECONV_SP_STRIDE8_TRANS': 15,
    'HWCN': 16,
    'NC1KHKWHWC0': 17,
    'BN_WEIGHT': 18,
    'FILTER_HWCK': 19,
    'HASHTABLE_LOOKUP_LOOKUPS': 20,
    'HASHTABLE_LOOKUP_KEYS': 21,
    'HASHTABLE_LOOKUP_VALUE': 22,
    'HASHTABLE_LOOKUP_OUTPUT': 23,
    'HASHTABLE_LOOKUP_HITS': 24,
    'C1HWNCoC0': 25,
    'MD': 26,
    'NDHWC': 27,
    'FRACTAL_ZZ': 28,
    'FRACTAL_NZ': 29,
    'NCDHW': 30,
    'DHWCN': 31,
    'NDC1HWC0': 32,
    'FRACTAL_Z_3D': 33,
    'CN': 34,
    'NC': 35,
    'DHWNC': 36,
    'FRACTAL_Z_3D_TRANSPOSE': 37,
    'FRACTAL_ZN_LSTM': 38,
    'FRACTAL_Z_G': 39,
    'RESERVED': 40,
    'ALL': 41,
    'NULL': 42,
    'ND_RNN_BIAS': 43,
    'FRACTAL_ZN_RNN': 44,
    'NYUV': 45,
    'NYUV_A': 46
}


def load_json(json_file: str):
    with open(json_file, encoding='utf-8') as file:
        json_content = json.load(file)
    return json_content


def get_specified_suffix_file(root_dir, suffix):
    specified_suffix = os.path.join(root_dir, '**/*.{}'.format(suffix))
    all_suffix_files = glob.glob(specified_suffix, recursive=True)
    return all_suffix_files


def get_deterministic_value(support_info):
    deterministic_key = 'deterministic'
    if deterministic_key not in support_info:
        return 0
    deterministic_value = support_info.get(deterministic_key)
    if deterministic_value == 'true':
        return 1
    else:
        return 0


def get_precision_value(support_info):
    precision_key = 'implMode'
    precision_value = support_info.get(precision_key)
    if precision_value == 'high_performance':
        _value = 1
    elif precision_value == 'high_precision':
        _value = 2
    else:
        _value = 0
    return _value


def get_overflow_value(support_info):
    return 0


def get_parameters(info):
    if info:
        if 'dtype' in info:
            data_type = info['dtype']
            data_type_value = DATA_TPYE_DICT.get(data_type)
        else:
            data_type_value = 0
        if 'format' in info:
            _format = info['format']
            _format_value = FORMAT_DICT.get(_format)
        else:
            _format_value = 0
    else:
        data_type_value = 0
        _format_value = 0
    return str(data_type_value), str(_format_value)


def get_dynamic_parameters(info):
    # 动态输入时只需获取第一个参数
    return get_parameters(info[0])


def get_all_parameters(support_info, _type):
    result_list = list()
    info_lists = support_info.get(_type)
    if info_lists:
        for _info in info_lists:
            # 输入为列表时是动态输入
            if isinstance(_info, (list, tuple)):
                data_type_value, _format_value = get_dynamic_parameters(_info)
            else:
                data_type_value, _format_value = get_parameters(_info)
            result_list.append("{},{}".format(data_type_value, _format_value))
    return result_list


def get_all_input_parameters(support_info):
    result = get_all_parameters(support_info, 'inputs')
    return '/'.join(result)


def insert_content_into_file(input_file, content):
    with open(input_file, 'r+') as file:
        lines = file.readlines()
        for index, line in enumerate(lines):
            match_result = re.search(r'"staticKey":', line)
            if match_result:
                count = len(line) - len(line.lstrip())
                new_content = "{}{}".format(' ' * count, content)
                # 插入到前一行，防止插入最后时还需要考虑是否添加逗号
                lines.insert(index, new_content)
                break
        file.seek(0)
        file.write(''.join(lines))


def insert_simplified_keys(json_file):
    contents = load_json(json_file)
    # 不存在'binFileName'或者'supportInfo'字段时，非需要替换的解析json文件
    if ('binFileName' not in contents) or ('supportInfo' not in contents):
        return
    support_info = contents.get('supportInfo')
    bin_file_name = contents.get('binFileName')
    # 'simplifiedKey'字段已经存在时，直接返回，不重复生成
    if 'simplifiedKey' in support_info:
        return
    op_type = bin_file_name.split('_')[0]
    deterministic = str(get_deterministic_value(support_info))
    precision = str(get_precision_value(support_info))
    overflow = str(get_overflow_value(support_info))
    input_parameters = get_all_input_parameters(support_info)
    key = '{}/d={},p={},o={}/{}/'.format(
        op_type,
        deterministic,
        precision,
        overflow,
        input_parameters)
    result = '"simplifiedKey": "' + key + '",\n'
    insert_content_into_file(json_file, result)


def insert_all_simplified_keys(root_dir):
    suffix = 'json'
    all_json_files = get_specified_suffix_file(root_dir, suffix)
    for _json in all_json_files:
        insert_simplified_keys(_json)


def args_prase():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--path',
                        nargs='?',
                        required=True,
                        help='Parse the path of the json file.')
    return parser.parse_args()


def main():
    args = args_prase()
    insert_all_simplified_keys(args.path)


if __name__ == '__main__':
    main()
