#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""
import os
import stat
import ctypes
import collections
import shutil
import subprocess
import copy

"""CODE_* is used to cube/vector api is called in operator code
CODE_MIX means both cube and vector api is called
CODE_CUBE means only cube api is called
CODE_VEC means only vector api is called
"""
CODE_MIX = 0
CODE_CUBE = 1
CODE_VEC = 2


def _is_v220(op_product: str):
    """return if current soc version is V220

    Returns:
        res: True means V220
    """
    if op_product in ["ascend910b", "ascend910c"]:
        return True
    return False


InfoCodeChanelParams = collections.namedtuple('InfoCodeChanelParams',\
['src_file', 'tiling_header', 'kernel_name', 'outdir', 'op_product', 'compile_options'])


def infer_code_channel(params: InfoCodeChanelParams):
    """get code channel for v220, return CODE_MIX if soc version is not V220

    Args:
        src_file (str): AscendC operator code file
        src_file (str): AscendC operator tiling header file
        kernel_name (str): kernel function name
        optype (str): operator type
        compile_options (list): compile options for bisheng cmd

    Raises:
        Exception: if not exist L1/L0/UB if code, it's not a aicore code

    Returns:
        res (int): CODE_MIX/CODE_CUBE/CODE_VEC
    """
    if not _is_v220(params.op_product):
        return CODE_MIX
    return CODE_VEC
