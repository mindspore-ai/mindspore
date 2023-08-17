#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""
import collections


#CODE_* is used to cube/vector api is called in operator code
#CODE_MIX means both cube and vector api is called
#CODE_CUBE means only cube api is called
#CODE_VEC means only vector api is called

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


InfoCodeChanelParams = collections.namedtuple('InfoCodeChanelParams',
                                              ['src_file', 'tiling_header',
                                               'kernel_name', 'outdir', 'op_product',
                                               'compile_options'])
