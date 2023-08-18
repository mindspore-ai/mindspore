#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import os
import glob
import json
import argparse
import const_var


def load_json(json_file: str):
    """
    load json
    """
    with open(json_file, encoding='utf-8') as file:
        json_content = json.load(file)
    return json_content


def get_specified_suffix_file(root_dir, suffix):
    """
    get_specified_suffix_file
    """
    specified_suffix = os.path.join(root_dir, '**/*.{}'.format(suffix))
    all_suffix_files = glob.glob(specified_suffix, recursive=True)
    return all_suffix_files


def add_simplified_config(op_type, key, core_type, objfile, config):
    """
    add_simplified_config
    """
    simple_cfg = config.get('binary_info_config.json')
    op_cfg = simple_cfg.get(op_type)
    if not op_cfg:
        op_cfg = {}
        op_cfg['dynamicRankSupport'] = True
        op_cfg['simplifiedKeyMode'] = 0
        op_cfg['binaryList'] = []
        simple_cfg[op_type] = op_cfg
    bin_list = op_cfg.get('binaryList')
    bin_list.append(
        {'coreType': core_type, 'simplifiedKey': key, 'binPath': objfile})


def add_op_config(op_file, bin_info, config):
    """
    add_op_config
    """
    op_cfg = config.get(op_file)
    if not op_cfg:
        op_cfg = {}
        op_cfg['binList'] = []
        config[op_file] = op_cfg
    op_cfg.get('binList').append(bin_info)


def gen_ops_config(json_file, soc, config):
    """
    gen_ops_config
    """
    core_type_map = {"MIX": 0, "AiCore": 1, "VectorCore": 2}
    contents = load_json(json_file)
    if ('binFileName' not in contents) or ('supportInfo' not in contents):
        return
    json_base_name = os.path.basename(json_file)
    op_dir = os.path.basename(os.path.dirname(json_file))
    support_info = contents.get('supportInfo')
    bin_name = contents.get('binFileName')
    bin_suffix = contents.get('binFileSuffix')
    core_type = core_type_map.get(contents.get("coreType"))
    bin_file_name = bin_name + bin_suffix
    op_type = bin_name.split('_')[0]
    op_file = op_dir + '.json'
    bin_info = {}
    keys = support_info.get('simplifiedKey')
    if keys:
        bin_info['simplifiedKey'] = keys
        for key in keys:
            add_simplified_config(op_type, key, core_type, os.path.join(
                soc, op_dir, bin_file_name), config)
    bin_info['staticKey'] = support_info.get('staticKey')
    bin_info['int64Mode'] = support_info.get('int64Mode')
    bin_info['inputs'] = support_info.get('inputs')
    bin_info['outputs'] = support_info.get('outputs')
    if support_info.get('attrs'):
        bin_info['attrs'] = support_info.get('attrs')
    bin_info['binInfo'] = {
        'jsonFilePath': os.path.join(soc, op_dir, json_base_name)}
    add_op_config(op_file, bin_info, config)


def gen_all_config(root_dir, soc):
    """Generate all config"""
    suffix = 'json'
    config = {}
    config['binary_info_config.json'] = {}
    all_json_files = get_specified_suffix_file(root_dir, suffix)
    for _json in all_json_files:
        gen_ops_config(_json, soc, config)
    for cfg_key, _ in config:
        cfg_file = os.path.join(root_dir, cfg_key)
        with os.fdopen(os.open(cfg_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            json.dump(config.get(cfg_key), fd, indent='  ')


def args_prase():
    """Parse the args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--path',
                        nargs='?',
                        required=True,
                        help='Parse the path of the json file.')
    parser.add_argument('-s',
                        '--soc',
                        nargs='?',
                        required=True,
                        help='Parse the soc_version of ops.')
    return parser.parse_args()


def main():
    """main"""
    args = args_prase()
    gen_all_config(args.path, args.soc)


if __name__ == '__main__':
    main()
