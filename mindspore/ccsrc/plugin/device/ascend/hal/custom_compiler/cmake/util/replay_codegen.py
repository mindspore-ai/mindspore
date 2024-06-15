#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import os
import stat
import collections
import kernel_entry as keb
from tiling_data_def_build import gen_tiling
import code_channel_infer
import const_var

PYF_PATH = os.path.dirname(__file__)

ReplayCodeGenParams = collections.namedtuple('ReplayCodeGenParams',\
['op_type', 'impl', 'tiling_file', 'kernel', 'entry', 'argn', 'op_replay_batch', 'max_block_dim', 'max_shape_size'])


class ReplayCodeGen:
    def __init__(self, replayCodeGenParams):
        self.op_type = replayCodeGenParams.op_type
        self.impl = replayCodeGenParams.impl
        self.tiling_file = replayCodeGenParams.tiling_file
        self.tiling_data_file = ''
        self.kernel = replayCodeGenParams.kernel
        self.entry = replayCodeGenParams.entry
        self.argn = replayCodeGenParams.argn
        self.batch = False
        self.outdir = ''
        self.data_type = 'uint8_t'
        self.blknum = 32
        self.op_replay_batch = replayCodeGenParams.op_replay_batch
        self.max_block_dim = replayCodeGenParams.max_block_dim
        self.max_shape_size = replayCodeGenParams.max_shape_size

    def set_batch(self, is_batch):
        self.batch = is_batch

    def set_outdir(self, outdir):
        self.outdir = outdir

    def gen_replay(self, ops_product: str):
        kerentry = os.path.join(self.outdir, self.kernel + '_entry.cce')
        kerimpl = os.path.join(self.outdir, self.kernel + '_impl.cpp')
        replayimpl = os.path.join(self.outdir, self.kernel + '_replay.cpp')
        if self.batch:
            reptmp = os.path.join(PYF_PATH, 'batch_replay_impl.temp')
        else:
            reptmp = os.path.join(PYF_PATH, 'replay_impl.temp')
        kertmp = os.path.join(PYF_PATH, 'kernel_impl.temp')
        self._gen_kentry(kerentry)
        self._gen_kimpl_code(kerimpl, kertmp)
        self._gen_tiling_data_header()
        self._gen_replay_code(replayimpl, reptmp, ops_product)

    def _gen_tiling_data_header(self):
        self.tiling_data_file = os.path.join(self.outdir, self.kernel + '_tiling_data.h')
        gen_tiling(self.tiling_file, self.tiling_data_file)

    def _gen_kimpl_code(self, src, tmpfile):
        with open(tmpfile, 'r') as fd:
            temp = fd.read()
            temp = temp.replace('__CCE_FILE__', self.impl)
        with os.fdopen(os.open(src, const_var.WFLAGS, const_var.WMODES), 'w') as ofd:
            ofd.write(temp)

    def _gen_replay_code(self, src, tmpfile, ops_product: str):
        with open(tmpfile, 'r') as fd:
            temp = fd.read()
            temp = temp.replace('__ARG_NUM__', str(self.argn))
            argdef = []
            kargs = []
            for i in range(0, self.argn):
                argdef.append('{} *'.format(self.data_type))
                kargs.append('({} *)GetArg({})'.format(self.data_type, i))
            temp = temp.replace('__ARGS_DEF__', ', '.join(argdef))
            temp = temp.replace('__KERNEL_ARGS__', ', '.join(kargs))
            temp = temp.replace('__KERNEL_FUN__', self.entry)
            core_type_infer = 'core_type'
            code_channel = code_channel_infer.infer_code_channel(code_channel_infer.InfoCodeChanelParams(self.impl,\
                self.tiling_data_file, self.kernel, self.outdir, ops_product, None))
            if code_channel == code_channel_infer.CODE_VEC:
                core_type_infer = '0'
            elif code_channel == code_channel_infer.CODE_CUBE:
                core_type_infer = '1'
            temp = temp.replace('__CORE_TYPE__', core_type_infer)
            # regist function
            temp = temp.replace('__OPS_PRODUCT__', ops_product)
            temp = temp.replace('__OPTYPE__', self.op_type)
        with os.fdopen(os.open(src, const_var.WFLAGS, const_var.WMODES), 'w') as ofd:
            ofd.write(temp)

    def _gen_kentry(self, src):
        kf = ''
        pre_alloc_str = 'A' * 256
        if self.batch:
            kf += keb.batch_code_gen("K{:02d}_{}{}".format(0, self.entry, pre_alloc_str), self.argn, self.data_type)
        else:
            kf += keb.mc_code_gen("K{:02d}_{}{}".format(0, self.entry, pre_alloc_str),\
            self.argn, self.data_type, self.blknum)
        with os.fdopen(os.open(src, const_var.WFLAGS, const_var.WMODES), 'w') as ofd:
            ofd.write(kf)
