#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
copyright 2020 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License == distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

matmul
"""
from __future__ import absolute_import

from collections import namedtuple
import logging

from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

cus_matmul_cube_fracz_right_mul_op_info = TBERegOp("CusMatMulCubeFraczRightMul") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmulcubefraczrightmul.so") \
    .compute_cost(10) \
    .kernel_name("cus_matmul_cube_fraczrightmul") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "required", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_FracZ, DataType.F16_Default, DataType.F32_Default,
                  DataType.F32_FracZ) \
    .get_op_info()


@op_info_register(cus_matmul_cube_fracz_right_mul_op_info)
def cus_matmul_cube_fraczrightmul(input_x1, input_x2, input_x3, output_y=None, kernel_name="matmulcube"):
    """CusMatMulCubeFraczRightMul"""
    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))

    input_x1_shape = input_x1.get("shape")
    input_x1_dtype = input_x1.get("dtype").lower()
    input_x2_shape = input_x2.get("shape")
    input_x2_dtype = input_x2.get("dtype").lower()
    input_x3_shape = input_x3.get("shape")
    input_x3_dtype = input_x3.get("dtype").lower()
    output_shape = output_y.get("shape")
    supported = [((72, 8, 16, 16), "float16", (72, 72, 16, 16), "float16", (1,), "float32"),
                 ((32, 8, 16, 16), "float16", (32, 32, 16, 16), "float16", (1,), "float32"),
                 ((8, 32, 16, 16), "float16", (8, 8, 16, 16), "float16", (1,), "float32"),
                 ((4, 4, 16, 16), "float16", (4, 4, 16, 16), "float16", (1,), "float32"),
                 ((4, 16, 16, 16), 'float16', (4, 4, 16, 16), 'float16', (1,), 'float32'),
                 ((49, 4, 16, 16), 'float16', (49, 49, 16, 16), 'float16', (1,), 'float32'),
                 ((36, 4, 16, 16), 'float16', (36, 36, 16, 16), 'float16', (1,), 'float32'),
                 ((64, 16, 16, 16), 'float16', (64, 64, 16, 16), 'float16', (1,), 'float32'),
                 ((32, 64, 16, 16), 'float16', (32, 32, 16, 16), 'float16', (1,), 'float32'),
                 ((32, 16, 16, 16), 'float16', (32, 32, 16, 16), 'float16', (1,), 'float32'),
                 ((16, 32, 16, 16), 'float16', (16, 16, 16, 16), 'float16', (1,), 'float32'),
                 ((16, 8, 16, 16), 'float16', (16, 16, 16, 16), 'float16', (1,), 'float32'),
                 ((16, 4, 16, 16), 'float16', (16, 16, 16, 16), 'float16', (1,), 'float32'),
                 ((288, 32, 16, 16), 'float16', (288, 288, 16, 16), 'float16', (1,), 'float32'),
                 ((144, 16, 16, 16), 'float16', (144, 144, 16, 16), 'float16', (1,), 'float32'),
                 ((128, 32, 16, 16), 'float16', (128, 128, 16, 16), 'float16', (1,), 'float32'),
                 ((64, 128, 16, 16), 'float16', (64, 64, 16, 16), 'float16', (1,), 'float32'),
                 ((32, 128, 16, 16), 'float16', (32, 32, 16, 16), 'float16', (1,), 'float32'),
                 ((64, 32, 16, 16), 'float16', (64, 64, 16, 16), 'float16', (1,), 'float32'),
                 ((16, 64, 16, 16), 'float16', (16, 16, 16, 16), 'float16', (1,), 'float32')]
    input_shape = (tuple(input_x1_shape), input_x1_dtype, tuple(input_x2_shape),
                   input_x2_dtype, tuple(input_x3_shape), input_x3_dtype)
    if input_shape not in supported:
        raise RuntimeError("input_shape %s is not supported" % str(input_shape))

    input_x1 = tik_instance.Tensor("float16", input_x1_shape, name="left_matrix", scope=tik.scope_gm)
    input_x2 = tik_instance.Tensor("float16", input_x2_shape, name="right_matrix", scope=tik.scope_gm)
    input_x3 = tik_instance.Tensor("float32", input_x3_shape, name="matrix_max", scope=tik.scope_gm)
    resmatmul = tik_instance.Tensor("float32", output_shape, name="output", scope=tik.scope_gm)
    cus_cube_matmul_right_mul(tik_instance, input_x1, input_x2, input_x3, resmatmul)
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x1, input_x2, input_x3], outputs=[resmatmul])
    return tik_instance


def get_cus_tile_info(input_x1, input_x2, input_x3):
    """get_cus_tile_info"""
    _, mo, _, _ = input_x1.shape
    no, _, _, _ = input_x2.shape
    c0 = input_x1.shape[-1]
    diag_outer = 128 // c0
    input_shape = (tuple(input_x1.shape), input_x1.dtype, tuple(input_x2.shape), input_x2.dtype,
                   tuple(input_x3.shape), input_x3.dtype)
    tile_map = {
        # no diag opt:
        ((8, 32, 16, 16), "float16", (8, 8, 16, 16), "float16", (1,), "float32"): (4, 8, 2, 8, 4),
        ((4, 4, 16, 16), "float16", (4, 4, 16, 16), "float16", (1,), "float32"): (1, 4, 1, 4, 4),
        ((4, 16, 16, 16), 'float16', (4, 4, 16, 16), 'float16', (1,), 'float32'): (1, 4, 2, 16, 2),
        ((49, 4, 16, 16), 'float16', (49, 49, 16, 16), 'float16', (1,), 'float32'): (1, 7, 7, 4, 7),
        ((36, 4, 16, 16), 'float16', (36, 36, 16, 16), 'float16', (1,), 'float32'): (2, 6, 3, 2, 12),
        # diag opt:
        ((288, 32, 16, 16), 'float16', (288, 288, 16, 16), 'float16', (1,), 'float32'): (16, 8, 8, 2, 12),
    }
    maxblocknum = 32
    diag_opt = False
    if input_x2.shape[0] * input_x2.shape[3] > 128 and input_x2.shape[0] % diag_outer == 0:
        diag_opt = True
    if input_shape in tile_map:
        mo_tile_, ko_tile_, no_tile_, core_m_num_, core_n_num_ = tile_map.get(input_shape)
    elif diag_opt:
        ko_tile_ = diag_outer
        no_tile_ = ko_tile_
        core_n_num_ = no // no_tile_
        core_m_num_max = maxblocknum // core_n_num_
        mo_tile_ = -1
        core_m_num_ = -1
        for i in range(core_m_num_max, 0, -1):
            if mo % i == 0:
                core_m_num_ = i
                mo_tile_ = mo // i
                break
        if mo_tile_ == -1:
            raise ValueError("no valid tile be found!")
        while mo_tile_ > 16:
            mo_tile_ = mo_tile_ // 2
    else:
        raise ValueError("please add tile config to the tile_map")
    logging.info(
        "shape: %s, tile: %s", input_shape, str((mo_tile_, ko_tile_, no_tile_, core_m_num_, core_n_num_,
                                                 diag_opt)))
    cus_tile_info = namedtuple('cus_tile_info', ['mo_tile_', 'ko_tile_', 'no_tile_', 'core_m_num_',
                                                 'core_n_num_', 'diag_opt'])
    return cus_tile_info(mo_tile_, ko_tile_, no_tile_, core_m_num_, core_n_num_, diag_opt)


def cus_cube_matmul_right_mul(tik_instance, input_x1, input_x2, input_x3,
                              res):
    """cus_cube_matmul_right_mul"""
    ko, mo, _, _ = input_x1.shape
    no, ko, _, _ = input_x2.shape
    c0 = input_x1.shape[-1]
    diag_outer = 128 // c0
    if [input_x1.shape[-1], input_x1.shape[-2], input_x2.shape[-1], input_x2.shape[-2]] != [c0, c0, c0, c0]:
        raise ValueError("shape of input_x1 or input_x2 is not supported!")

    mo_tile, ko_tile, no_tile, core_m_num, core_n_num, diag_opt = get_cus_tile_info(input_x1, input_x2, input_x3)
    fp32_size = 4
    fp16_size = 2
    blocksize = 32
    vectorfp32_size = 64
    loop_n_num_total = no // no_tile
    loop_m_num_total = mo // mo_tile
    if loop_n_num_total % core_n_num != 0 or loop_m_num_total % core_m_num != 0:
        raise ValueError("Does not support this scenario!")
    loop_n_num = loop_n_num_total // core_n_num
    loop_m_num = loop_m_num_total // core_m_num
    block_num = core_n_num * core_m_num
    loop_k_num = ko // ko_tile
    if diag_opt:
        loop_k_num = diag_outer // ko_tile
    # double buffer:
    thread_num_k = 2
    if ko_tile % 2 == 0:
        loop_k_num *= thread_num_k
        ko_tile_inner = ko_tile // thread_num_k
    else:
        ko_tile_inner = ko_tile
        ko_tile *= thread_num_k
    with tik_instance.for_range(0, block_num, block_num=block_num) as block_idx, \
            tik_instance.for_range(0, loop_m_num) as cc_m, tik_instance.for_range(0, loop_n_num) as cc_n:
        core_m = block_idx // core_n_num
        core_n = block_idx % core_n_num
        res_l0c = tik_instance.Tensor("float32", [no_tile, mo_tile, c0, c0],
                                      name="resmatmul_L0C", scope=tik.scope_cc)
        with tik_instance.for_range(0, loop_k_num, thread_num=thread_num_k) as thread_idx_k:
            if diag_opt:
                k_idx = (core_n * loop_n_num + cc_n) * no_tile + thread_idx_k * ko_tile_inner
            else:
                k_idx = thread_idx_k * ko_tile_inner
            # input_x1 -> input_x1_l1
            input_x1_l1 = tik_instance.Tensor(input_x1.dtype, [ko_tile_inner, mo_tile, c0, c0],
                                              name="input_x1_l1", scope=tik.scope_cbuf)
            tik_instance.data_move(input_x1_l1,
                                   input_x1[k_idx,
                                            (core_m * loop_m_num + cc_m) * mo_tile, 0, 0],
                                   0, ko_tile_inner, mo_tile * c0 * c0 * fp16_size // blocksize,
                                   (mo - mo_tile) * c0 * c0 * fp16_size // blocksize, 0)
            # input_x2 -> input_x2_l1
            input_x2_l1 = tik_instance.Tensor("float16", [no_tile, ko_tile_inner, c0, c0],
                                              name="input_x2_l1", scope=tik.scope_cbuf)
            tik_instance.data_move(input_x2_l1,
                                   input_x2[(core_n * loop_n_num + cc_n) * no_tile,
                                            k_idx, 0, 0],
                                   0, no_tile, ko_tile_inner * c0 * c0 * fp16_size // blocksize,
                                   (ko - ko_tile_inner) * c0 * c0 * fp16_size // blocksize, 0)
            # input_x1_l1 -> input_x1_l0a
            input_x1_l0a = tik_instance.Tensor(input_x1.dtype, [mo_tile, ko_tile_inner, c0, c0],
                                               name="input_x1_l0a", scope=tik.scope_ca)
            with tik_instance.for_range(0, mo_tile) as cc1:
                tik_instance.load2dv1(input_x1_l0a[cc1, 0, 0, 0], input_x1_l1[0, cc1, 0, 0], 0, ko_tile_inner,
                                      mo_tile, 0, False)
            # input_x2_l1 -> input_x2_l0b
            input_x2_l0b = tik_instance.Tensor("float16", [ko_tile_inner, no_tile, c0, c0],
                                               name="input_x2_l0b", scope=tik.scope_cb)
            with tik_instance.for_range(0, ko_tile_inner) as cc2:
                tik_instance.load2dv1(input_x2_l0b[cc2, 0, 0, 0], input_x2_l1[0, cc2, 0, 0], 0, no_tile,
                                      ko_tile_inner,
                                      0, True)
            with tik_instance.if_scope(thread_idx_k == 0):
                tik_instance.mmad(res_l0c, input_x1_l0a, input_x2_l0b, mo_tile * c0,
                                  ko_tile_inner * c0, no_tile * c0, 0)
            with tik_instance.else_scope():
                tik_instance.mmad(res_l0c, input_x1_l0a, input_x2_l0b, mo_tile * c0,
                                  ko_tile_inner * c0, no_tile * c0, 1)
        res_ub = tik_instance.Tensor("float32", [no_tile, mo_tile, c0, c0],
                                     name="resmatmul_ub", scope=tik.scope_ubuf)
        tik_instance.data_move(res_ub, res_l0c, 0, 1, no_tile * mo_tile, 0, 0)

        input_3_local_ub = tik_instance.Tensor("float32", (8,), scope=tik.scope_ubuf, name="input_3_local_ub")
        tik_instance.data_move(input_3_local_ub, input_x3, 0, 1, 1, 0, 0)
        matrix_max_scalar = tik_instance.Scalar("float32")
        matrix_max_scalar.set_as(input_3_local_ub[0])
        repeate_num = no_tile * mo_tile * c0 * c0 // vectorfp32_size
        repeate_times_max = 255
        count = 0
        while repeate_num > repeate_times_max:
            tik_instance.vmuls(vectorfp32_size,
                               res_ub[count * repeate_times_max * vectorfp32_size],
                               res_ub[count * repeate_times_max * vectorfp32_size],
                               matrix_max_scalar, repeate_times_max, 1, 1, 8, 8)
            repeate_num -= repeate_times_max
            count += 1
        tik_instance.vmuls(vectorfp32_size,
                           res_ub[count * repeate_times_max * vectorfp32_size],
                           res_ub[count * repeate_times_max * vectorfp32_size],
                           matrix_max_scalar, repeate_num, 1, 1, 8, 8)

        tik_instance.data_move(res[(core_n * loop_n_num + cc_n) * no_tile,
                                   (core_m * loop_m_num + cc_m) * mo_tile, 0, 0],
                               res_ub, 0, no_tile,
                               mo_tile * c0 * c0 * fp32_size // blocksize, 0,
                               (mo - mo_tile) * c0 * c0 * fp32_size // blocksize)
