#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import sys
import os
import json
import hashlib
import const_var
import opdesc_parser

PYF_PATH = os.path.dirname(os.path.realpath(__file__))


class BinParamBuilder(opdesc_parser.OpDesc):
    def __init__(self: any, op_type: str):
        super().__init__(op_type)
        self.soc = ''
        self.out_path = ''

    def set_soc_version(self: any, soc: str):
        self.soc = soc

    def set_out_path(self: any, out_path: str):
        self.out_path = out_path

    def gen_input_json(self: any):
        key_map = {}
        count = len(self.input_dtype[0].split(','))
        for i in range(0, count):
            inputs = []
            outputs = []
            attrs = []
            op_node = {}
            for idx in range(0, len(self.input_name)):
                idtypes = self.input_dtype[idx].split(',')
                ifmts = self.input_fmt[idx].split(',')
                itype = self.input_type[idx]
                para = {}
                para['name'] = self.input_name[idx][:-5]
                para['index'] = idx
                para['dtype'] = idtypes[i]
                para['format'] = ifmts[i]
                para['paramType'] = itype
                para['shape'] = [-2]
                if itype == 'dynamic':
                    inputs.append([para])
                else:
                    inputs.append(para)
            for idx in range(0, len(self.output_name)):
                odtypes = self.output_dtype[idx].split(',')
                ofmts = self.output_fmt[idx].split(',')
                otype = self.output_type[idx]
                para = {}
                para['name'] = self.output_name[idx][:-5]
                para['index'] = idx
                para['dtype'] = odtypes[i]
                para['format'] = ofmts[i]
                para['paramType'] = otype
                para['shape'] = [-2]
                if otype == 'dynamic':
                    outputs.append([para])
                else:
                    outputs.append(para)
            for attr in self.attr_list:
                att = {}
                att['name'] = attr
                atype = self.attr_val.get(attr).get('type').lower()
                att['dtype'] = atype
                att['value'] = const_var.ATTR_DEF_VAL.get(atype)
                attrs.append(att)
            op_node['bin_filename'] = ''
            op_node['inputs'] = inputs
            op_node['outputs'] = outputs
            if len(attrs) > 0:
                op_node['attrs'] = attrs
            param = {}
            param['op_type'] = self.op_type
            param['op_list'] = [op_node]
            objstr = json.dumps(param, indent='  ')
            md5sum = hashlib.md5(objstr.encode('utf-8')).hexdigest()
            while key_map.get(md5sum) is not None:
                objstr += '1'
                md5sum = hashlib.md5(objstr.encode('utf-8')).hexdigest()
            key_map[md5sum] = md5sum
            bin_file = self.op_type + '_' + md5sum
            op_node['bin_filename'] = bin_file
            param_file = os.path.join(self.out_path, bin_file + '_param.json')
            param_file = os.path.realpath(param_file)
            with os.fdopen(os.open(param_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
                json.dump(param, fd, indent='  ')
            self._write_buld_cmd(param_file, bin_file, i)

    def _write_buld_cmd(self: any, param_file: str, bin_file: str, index: int):
        hard_soc = const_var.SOC_MAP_EXT.get(self.soc)
        if not hard_soc:
            hard_soc = self.soc.capitalize()
        name_com = [self.op_type, self.op_file, str(index)]
        compile_file = os.path.join(self.out_path, '-'.join(name_com) + '.sh')
        compile_file = os.path.realpath(compile_file)
        with os.fdopen(os.open(compile_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            fd.write('#!/bin/bash\n')
            fd.write('echo "[{}] Generating {} ..."\n'.format(hard_soc, bin_file))
            cmd = const_var.BIN_CMD.format(fun=self.op_intf, soc=hard_soc, param=param_file, impl='""')
            fd.write(cmd)
            chk = const_var.CHK_CMD.format(res_file=bin_file + '.json')
            fd.write(chk)
            chk = const_var.CHK_CMD.format(res_file=bin_file + '.o')
            fd.write(chk)
            fd.write('echo "[{}] Generating {} Done"\n'.format(hard_soc, bin_file))


def gen_bin_param_file(cfgfile: str, out_dir: str, soc: str):
    op_descs = opdesc_parser.get_op_desc(cfgfile, [], [], BinParamBuilder, None)
    for op_desc in op_descs:
        op_desc.set_soc_version(soc)
        op_desc.set_out_path(out_dir)
        op_desc.gen_input_json()


if __name__ == '__main__':
    if len(sys.argv) <= 3:
        raise RuntimeError('arguments must greater than 3')
    gen_bin_param_file(sys.argv[1], sys.argv[2], sys.argv[3])
