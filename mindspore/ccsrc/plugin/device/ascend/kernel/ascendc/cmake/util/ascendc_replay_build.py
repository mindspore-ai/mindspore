#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import sys
import os
import opdesc_parser
import replay_codegen
import const_var
from replay_codegen import ReplayCodeGenParams

PYF_PATH = os.path.dirname(os.path.realpath(__file__))


class ReplayBuilder(opdesc_parser.OpDesc):
    def __init__(self: any, op_type: str):
        super().__init__(op_type)

    def gen_replay_source(self: any, impl_path: str, out_path: str, ops_product: str):
        if not self.op_replay_flag:
            print('{} replay not enabled'.format(self.op_type))
            return
        argn = len(self.input_name) + len(self.output_name) + 1
        if self.op_replay_batch:
            print('{} replay in batch mode'.format(self.op_type))
        else:
            print('{} replay in normal mode'.format(self.op_type))
        if impl_path.endswith('op_kernel'):
            implf = os.path.join(impl_path, self.op_file + '.cpp')
            tiling_file = os.path.join(impl_path, "../op_host", self.op_file + '_tiling.h')
        else:
            if self.dynamic_shape:
                dyn_path = 'dynamic'
            else:
                dyn_path = ''
            implf = os.path.join(impl_path, dyn_path, self.op_file + '.cpp')
            tiling_file = os.path.join(impl_path, "../../op_tiling", self.op_file + '_tiling.h')
        rep_conf = replay_codegen.ReplayCodeGen(ReplayCodeGenParams(self.op_type, implf, tiling_file, self.op_file, \
            self.op_intf, argn, self.op_replay_batch, self.max_block_dim, self.max_shape_size))
        rep_conf.set_batch(self.op_replay_batch)
        rep_conf.set_outdir(out_path)
        rep_conf.gen_replay(ops_product)


def gen_replay(cfgfile: str, cfgs: dict, dirs: dict, ops_product: str, ops: list = None):
    batch_lists = cfgs.get(const_var.REPLAY_BATCH).split(';')
    iterator_lists = cfgs.get(const_var.REPLAY_ITERATE).split(';')
    op_descs = opdesc_parser.get_op_desc(cfgfile, batch_lists, iterator_lists, ReplayBuilder, ops)
    for op_desc in op_descs:
        op_desc.gen_replay_source(dirs.get(const_var.CFG_IMPL_DIR), dirs.get(const_var.CFG_OUT_DIR), ops_product)


if __name__ == '__main__':
    if len(sys.argv) <= 6:
        raise RuntimeError('arguments must greater than 6')
    rep_cfg = {}
    rep_cfg[const_var.REPLAY_BATCH] = sys.argv[2]
    rep_cfg[const_var.REPLAY_ITERATE] = sys.argv[3]
    rep_dir = {}
    rep_dir[const_var.CFG_IMPL_DIR] = sys.argv[4]
    rep_dir[const_var.CFG_OUT_DIR] = sys.argv[5]
    gen_replay(sys.argv[1], rep_cfg, rep_dir, sys.argv[6])
