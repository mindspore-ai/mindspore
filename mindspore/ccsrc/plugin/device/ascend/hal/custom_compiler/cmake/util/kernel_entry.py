#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""


def gen_fun_def(title, kernel, argn, arg_type, arg_name):
    entry = []
    entry.append(title)
    entry.append(kernel)
    entry.append('(')
    args = []
    for i in range(0, argn):
        args.append(arg_type + ' ' + arg_name + str(i))
    entry.append(', '.join(args))
    entry.append(')')
    return ' '.join(entry)


def gen_batch_kernel_body(fname, argn, arg_name):
    body = []
    body.append('{')
    fun = []
    fun.append(fname)
    fun.append('(')
    args = []
    for i in range(0, argn):
        args.append(arg_name + str(i))
    fun.append(', '.join(args))
    fun.append(');')
    body.append(' '.join(fun))
    body.append('}')
    return '\n'.join(body)


def gen_mc_kernel_body(kn, argn, arg_name, blknum):
    body = []
    body.append('{')
    body.append('    switch(block_idx) {')
    for blk in range(0, blknum):
        fun = []
        fun.append('{}_blk{:02d}'.format(kn, blk))
        fun.append('(')
        args = []
        for i in range(0, argn):
            args.append(arg_name + str(i))
        fun.append(', '.join(args))
        fun.append(')')
        body.append('        case {}: {}; break;'.format(blk, ' '.join(fun)))
    body.append('        default: break;')
    body.append('    }')
    body.append('}')
    return '\n'.join(body)


def gen_proc_body(argn, arg_name):
    body = []
    body.append('{')
    args = []
    for i in range(0, argn):
        args.append(arg_name + str(i))
    body.append('uint64_t __x = (uint64_t)' + ' + (uint64_t)'.join(args) + ';')
    body.append('__asm__ ("NOP");')
    body.append('__asm__ ("NOP");')
    body.append('__asm__ ("NOP");')
    body.append('}')
    return '\n'.join(body)


def batch_code_gen(kn, argn, argt):
    codes = []
    kernel_name = kn
    proc_name = kernel_name + '_percore'
    arg_num = int(argn)
    data_type = argt
    arg_type = '__gm__ ' + data_type + '* __restrict__'
    arg_name = 'arg'
    kernel_title = 'extern \"C\" __global__ __aicore__ void'
    proc_title = 'extern \"C\" __attribute__((noinline)) __aicore__ void'
    codes.append('#ifndef __aicore__')
    codes.append('#define __aicore__ [aicore]')
    codes.append('#endif')
    codes.append(gen_fun_def(proc_title, proc_name, arg_num, arg_type, arg_name) + ';')
    codes.append(gen_fun_def(kernel_title, kernel_name, arg_num, arg_type, arg_name))
    codes.append(gen_batch_kernel_body(proc_name, arg_num, arg_name))
    codes.append(gen_fun_def(proc_title, proc_name, arg_num, arg_type, arg_name))
    codes.append(gen_proc_body(arg_num, arg_name))
    return '\n'.join(codes) + '\n'


def mc_code_gen(kn, argn, argt, blknum):
    codes = []
    kernel_name = kn
    core_num = int(blknum)
    arg_num = int(argn)
    data_type = argt
    arg_type = '__gm__ ' + data_type + '* __restrict__'
    arg_name = 'arg'
    kernel_title = 'extern \"C\" __global__ __aicore__ void'
    proc_title = 'extern \"C\" __attribute__((noinline)) __aicore__ void'
    codes.append('#ifndef __aicore__')
    codes.append('#define __aicore__ [aicore]')
    codes.append('#endif')
    for i in range(0, core_num):
        proc_name = '{}_blk{:02d}'.format(kernel_name, i)
        codes.append(gen_fun_def(proc_title, proc_name, arg_num, arg_type, arg_name) + ';')
    codes.append(gen_fun_def(kernel_title, kernel_name, arg_num, arg_type, arg_name))
    codes.append(gen_mc_kernel_body(kernel_name, arg_num, arg_name, core_num))
    for i in range(0, core_num):
        proc_name = '{}_blk{:02d}'.format(kernel_name, i)
        codes.append(gen_fun_def(proc_title, proc_name, arg_num, arg_type, arg_name))
        codes.append(gen_proc_body(arg_num, arg_name))
    return '\n'.join(codes) + '\n'
