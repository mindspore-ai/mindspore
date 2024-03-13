#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import sys
import os


OP_ALL = '__ALLOP__'
SOC_ALL = '__ALLSOC__'
SOC_TO_SHORT_SOC_MAP = {
    "ascend910a": "ascend910",
    "ascend910proa": "ascend910",
    "ascend910b": "ascend910",
    "ascend910prob": "ascend910",
    "ascend910premiuma": "ascend910",
    "ascend910b1": "ascend910b",
    "ascend910b2": "ascend910b",
    "ascend910b2c": "ascend910b",
    "ascend910b3": "ascend910b",
    "ascend910b4": "ascend910b",
    "ascend910c1": "ascend910c",
    "ascend910c2": "ascend910c",
    "ascend910c3": "ascend910c",
    "ascend910c4": "ascend910c",
    "ascend310p1": "ascend310p",
    "ascend310p3": "ascend310p",
    "ascend310p3vir01": "ascend310p",
    "ascend310p3vir02": "ascend310p",
    "ascend310p3vir04": "ascend310p",
    "ascend310p3vir08": "ascend310p",
    "ascend310b1": "ascend310b",
    "bs9sx1aa": "bs9sx1a"
}


class OpDesc:
    def __init__(self: any, op_type: str):
        self.op_type = op_type
        self.attr_list = []
        self.attr_val = {}
        self.input_name = []
        self.input_type = []
        self.input_dtype = []
        self.input_fmt = []
        self.output_name = []
        self.output_type = []
        self.output_dtype = []
        self.output_fmt = []
        self.op_fmt_sel = False
        self.op_chk_support = False
        self.op_intf = ''
        self.kern_name = ''
        self.op_file = ''
        self.op_replay_flag = False
        self.op_replay_batch = False
        self.input_idx = -1
        self.output_idx = -1
        self.max_block_dim = 32
        self.max_shape_size = 268435456
        self.dynamic_shape = False
        self.op_range_limit = ''
        self.custom_compile_options = {}
        self.custom_all_compile_options = {}

    @staticmethod
    def _parse_digit(conf: str) -> int:
        return int(conf.split('=')[1])

    @staticmethod
    def _parse_flag(conf: str) -> bool:
        if 'true' == conf.split('=')[1]:
            return True
        return False

    @staticmethod
    def _parse_str(conf: str) -> str:
        return conf.split('=')[1]

    @staticmethod
    def _parse_list(conf: str) -> list:
        return conf.split('=')[1].split(',')

    def parse_input(self: any, conf: str):
        if conf.startswith('input{}.name'.format(int(self.input_idx) + 1)):
            self.input_idx += 1
            self.input_name.append(self._parse_str(conf) + '_in__')
        elif conf.startswith('input{}.paramType'.format(int(self.input_idx))):
            self.input_type.append(self._parse_str(conf))
        elif conf.startswith('input{}.dtype'.format(int(self.input_idx))):
            self.input_dtype.append(self._parse_str(conf))
        elif conf.startswith('input{}.format'.format(int(self.input_idx))):
            self.input_fmt.append(self._parse_str(conf))
        else:
            return

    def parse_output(self: any, conf: str):
        if conf.startswith('output{}.name'.format(int(self.output_idx) + 1)):
            self.output_idx += 1
            self.output_name.append(self._parse_str(conf) + '_out_')
        elif conf.startswith('output{}.paramType'.format(int(self.output_idx))):
            self.output_type.append(self._parse_str(conf))
        elif conf.startswith('output{}.dtype'.format(int(self.output_idx))):
            self.output_dtype.append(self._parse_str(conf))
        elif conf.startswith('output{}.format'.format(int(self.output_idx))):
            self.output_fmt.append(self._parse_str(conf))
        else:
            return

    def parse_op_format(self: any, conf: str):
        self.op_fmt_sel = self._parse_flag(conf)

    def parse_check_support(self: any, conf: str):
        self.op_chk_support = self._parse_flag(conf)

    def parse_range_limit(self: any, conf: str):
        self.op_range_limit = self._parse_str(conf)

    def parse_kern_name(self: any, conf: str):
        self.kern_name = self._parse_str(conf)

    def parse_op_intf(self: any, conf: str):
        self.op_intf = self._parse_str(conf)

    def parse_op_file(self: any, conf: str):
        self.op_file = self._parse_str(conf)

    def parse_dynamic_shape(self: any, conf: str):
        self.dynamic_shape = self._parse_flag(conf)

    def parse_attr_list(self: any, conf: str):
        self.attr_list = self._parse_list(conf)

    @staticmethod
    def _camel_to_snake(camel_case_str: str):
        snake_case_str = ''
        for i, c in enumerate(camel_case_str):
            if i == 0:
                snake_case_str += c.lower()
            elif c.isupper():
                snake_case_str += '_' + c.lower()
            else:
                snake_case_str += c
        return snake_case_str

    def parse_attr_val(self: any, conf: str):
        for attr in self.attr_list:
            if self.attr_val.get(attr) is None:
                self.attr_val[attr] = {}
            if conf.startswith('attr_{}.type'.format(attr)):
                self.attr_val.get(attr)['type'] = self._camel_to_snake(self._parse_str(conf))
            elif conf.startswith('attr_{}.paramType'.format(attr)):
                self.attr_val.get(attr)['paramType'] = self._parse_str(conf)
            elif conf.startswith('attr_{}.defaultValue'.format(attr)):
                self.attr_val.get(attr)['defaultValue'] = self._parse_str(conf)

    def parse_replay_val(self: any, batch_list: list, iterator_list: list):
        if self.op_type in batch_list:
            self.op_replay_flag = True
            self.op_replay_batch = True
        elif self.op_type in iterator_list:
            self.op_replay_flag = True
            self.op_replay_batch = False


def _is_op_type_in_opdesc(op_descs: list, op_type: str):
    for op in op_descs:
        if op_type == op.op_type:
            return True
    return False


def _set_all_options_to_opdescs(op_descs, soc_ver_compile_options):
    for op in op_descs:
        op.custom_all_compile_options = soc_ver_compile_options


def _set_options_to_opdesc(op_descs, op_type, soc_ver_compile_options):
    for op in op_descs:
        if op.op_type != op_type:
            continue
        op.custom_compile_options = soc_ver_compile_options


def _trans_soc_ver_to_short(soc_ver: str):
    low_soc_ver = soc_ver.lower()
    if low_soc_ver not in SOC_TO_SHORT_SOC_MAP:
        print(f'WARNING: caution: {soc_ver} will trans into ascend910, if not your intention,'
              f'use ascend910b1~4 instead')
    return SOC_TO_SHORT_SOC_MAP[low_soc_ver]


def _get_op_custom_options(op_descs: list, auto_gen_dir: str):
    if auto_gen_dir is None:
        return {}
    file = os.path.join(auto_gen_dir, "custom_compile_options.ini")
    if not os.path.exists(file):
        print(f'WARNING: cannot find {auto_gen_dir}/custom_compile_options.ini')
        return {}
    with open (file, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            param_list = str.split(line.rstrip('\n'), ',')
            if len(param_list) != 3:
                raise Exception(f'ERROR: custom compile option {param_list} len is not 3')
            op_type = param_list[0]
            if op_type.upper() == 'ALL':
                op_type = OP_ALL
            if op_type != OP_ALL and _is_op_type_in_opdesc(op_descs, op_type) == False:
                print(f'WARNING: op: {op_type} are not exists in this project')
                continue
            soc_ver_compile_options = {}
            soc_ver = param_list[1]
            options_str = param_list[2]
            options = str.split(options_str, ';')
            if soc_ver == '':
                soc_ver_compile_options[SOC_ALL] = options
            else:
                soc_ver_list = str.split(soc_ver, ';')
                for ver in soc_ver_list:
                    short_ver = _trans_soc_ver_to_short(ver)
                    soc_ver_compile_options[short_ver] = options
            if op_type == OP_ALL:
                _set_all_options_to_opdescs(op_descs, soc_ver_compile_options)
            else:
                _set_options_to_opdesc(op_descs, op_type, soc_ver_compile_options)


def get_op_desc(file: str, batch_list: list, iterator_list: list, builder: any,
                op_type: list, auto_gen_dir: str = None) -> list:
    op_descs = []
    op_match = False
    with open (file, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('['):
                name = line[1:-1]
                if op_type is None or name in op_type:
                    op_match = True
                    op_desc = builder(name)
                    op_desc.parse_replay_val(batch_list, iterator_list)
                    op_descs.append(op_desc)
                else:
                    op_match = False
                    if op_type is not None and len(op_descs) == len(op_type):
                        return op_descs
                continue
            if not op_match:
                continue
            if line.startswith('input'):
                op_desc.parse_input(line)
            elif line.startswith('output'):
                op_desc.parse_output(line)
            elif line.startswith('dynamicFormat.flag'):
                op_desc.parse_op_format(line)
            elif line.startswith('needCheckSupport.flag'):
                op_desc.parse_check_support(line)
            elif line.startswith('rangeLimit.value'):
                op_desc.parse_range_limit(line)
            elif line.startswith('opInterface.value'):
                op_desc.parse_op_intf(line)
            elif line.startswith('kernel.name'):
                op_desc.parse_kern_name(line)
            elif line.startswith('opFile.value'):
                op_desc.parse_op_file(line)
            elif line.startswith('dynamicShapeSupport.flag'):
                op_desc.parse_dynamic_shape(line)
            elif line.startswith('attr.list'):
                op_desc.parse_attr_list(line)
            elif line.startswith('attr_'):
                op_desc.parse_attr_val(line)
    _get_op_custom_options(op_descs, auto_gen_dir)
    return op_descs
