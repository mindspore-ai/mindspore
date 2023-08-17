#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""


class OpDesc:
    """
    OpDesc
    """
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

    @staticmethod
    def _parse_digit(conf: str) -> int:
        return int(conf.split('=')[1])

    @staticmethod
    def _parse_flag(conf: str) -> bool:
        if conf.split('=')[1] == 'true':
            return True
        return False

    @staticmethod
    def _parse_str(conf: str) -> str:
        return conf.split('=')[1]

    @staticmethod
    def _parse_list(conf: str) -> list:
        return conf.split('=')[1].split(',')

    def parse_input(self: any, conf: str):
        """parse_input
        """
        if conf.startswith('input{}.name'.format(int(self.input_idx) + 1)):
            self.input_idx += 1
            self.input_name.append(self._parse_str(conf))
        elif conf.startswith('input{}.paramType'.format(int(self.input_idx))):
            self.input_type.append(self._parse_str(conf))
        elif conf.startswith('input{}.dtype'.format(int(self.input_idx))):
            self.input_dtype.append(self._parse_str(conf))
        elif conf.startswith('input{}.format'.format(int(self.input_idx))):
            self.input_fmt.append(self._parse_str(conf))
        else:
            return

    def parse_output(self: any, conf: str):
        """parse_output
        """
        if conf.startswith('output{}.name'.format(int(self.output_idx) + 1)):
            self.output_idx += 1
            self.output_name.append(self._parse_str(conf))
        elif conf.startswith('output{}.paramType'.format(int(self.output_idx))):
            self.output_type.append(self._parse_str(conf))
        elif conf.startswith('output{}.dtype'.format(int(self.output_idx))):
            self.output_dtype.append(self._parse_str(conf))
        elif conf.startswith('output{}.format'.format(int(self.output_idx))):
            self.output_fmt.append(self._parse_str(conf))
        else:
            return

    def parse_op_format(self: any, conf: str):
        """parse_op_format
        """
        self.op_fmt_sel = self._parse_flag(conf)

    def parse_check_support(self: any, conf: str):
        """parse_check_support"""
        self.op_chk_support = self._parse_flag(conf)

    def parse_range_limit(self: any, conf: str):
        """parse_range_limit"""
        self.op_range_limit = self._parse_str(conf)

    def parse_kern_name(self: any, conf: str):
        """parse_kern_name"""
        self.kern_name = self._parse_str(conf)

    def parse_op_intf(self: any, conf: str):
        """parse_op_intf"""
        self.op_intf = self._parse_str(conf)

    def parse_op_file(self: any, conf: str):
        """parse_op_file"""
        self.op_file = self._parse_str(conf)

    def parse_dynamic_shape(self: any, conf: str):
        """parse_dynamic_shape"""
        self.dynamic_shape = self._parse_flag(conf)

    def parse_attr_list(self: any, conf: str):
        """parse_attr_list"""
        self.attr_list = self._parse_list(conf)

    def parse_attr_val(self: any, conf: str):
        """parse_attr_val"""
        for attr in self.attr_list:
            if self.attr_val.get(attr) is None:
                self.attr_val[attr] = {}
            if conf.startswith('attr_{}.type'.format(attr)):
                self.attr_val.get(attr)['type'] = self._parse_str(conf)
            elif conf.startswith('attr_{}.paramType'.format(attr)):
                self.attr_val.get(attr)['paramType'] = self._parse_str(conf)
            elif conf.startswith('attr_{}.defaultValue'.format(attr)):
                self.attr_val.get(attr)['defaultValue'] = self._parse_str(conf)

    def parse_replay_val(self: any, batch_list: list, iterator_list: list):
        """parse_replay_val"""
        if self.op_type in batch_list:
            self.op_replay_flag = True
            self.op_replay_batch = True
        elif self.op_type in iterator_list:
            self.op_replay_flag = True
            self.op_replay_batch = False


def get_op_desc(file: str, batch_list: list, iterator_list: list, builder: any, op_type: list) -> list:
    """get_op_desc"""
    op_descs = []
    op_match = False
    with open(file, 'r') as fd:
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
    return op_descs
