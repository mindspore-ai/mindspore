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

import collections

import te.platform.cce_params as cce
from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
from mindspore.ops._op_impl._custom_op._basic import _shape_check, _get_bias, _get_input_shape

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)

matmul_cube_fracz_left_cast_op_info = TBERegOp("CusMatMulCubeFraczLeftCast") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmulcubefraczleftcast.so") \
    .compute_cost(10) \
    .kernel_name("cus_matmul_cube_fraczleftcast") \
    .partial_flag(True) \
    .attr("trans_a", "optional", "bool", "all", "false") \
    .attr("trans_b", "optional", "bool", "all", "false") \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F32_FracZ, DataType.F16_Default, DataType.F16_FracZ) \
    .get_op_info()


def _clip_num(num):
    """clip number"""
    if num == 0:
        num = 1
    return num


def _get_block(trans_a, trans_b, m_shape, n_shape, km_shape, kn_shape):
    """_get_block"""
    block_in0 = cce.BLOCK_IN
    block_out0 = cce.BLOCK_OUT
    if trans_a and km_shape == 1:
        block_in0 = cce.BLOCK_VECTOR

    if not trans_a and m_shape == 1:
        block_in0 = cce.BLOCK_VECTOR

    if trans_b and kn_shape == 1:
        block_out0 = cce.BLOCK_VECTOR

    if not trans_b and n_shape == 1:
        block_out0 = cce.BLOCK_VECTOR
    return block_in0, block_out0


@op_info_register(matmul_cube_fracz_left_cast_op_info)
def cus_matmul_cube_fraczleftcast(input_x1, input_x2, bias=None, output_y=None, trans_a=False, trans_b=False,
                                  kernel_name="cus_matmul_cube_fraczleftcast"):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    shape_aa: list or tuple
            Shape of the first tensor a with rank > 1
    shape_bb:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_aa, shape_bb must be 2 dims
    src_dtype: str
            The data type of input, support "float32", "float16"
    dst_dtype: str
            The data type of output, support "float32", "float16"
    trans_a: bool
            If True, shape_aa == transposed before multiplication
    trans_b: bool
            If True, shape_bb == transposed before multiplication
    is_fractal: bool
            If True, the input data format of a and b must be fractal format
    shape_bbias: list or tuple
            Shape of bias, only support the input data format with ND

    Returns
    -------
    None
    """
    shape_aa = input_x1.get("ori_shape")
    shape_bb = input_x2.get("ori_shape")
    if input_x2.get("format") == "FRACTAL_Z":
        n, c, h, w = shape_bb
        c0 = 16
        c1 = c // c0
        c1 = _clip_num(c1)
        shape_bb = [n, c1 * h * w * c0]
        shape_aa = [n, n]

    if input_x1.get("format") == "FRACTAL_Z":
        n, c, h, w = shape_aa
        c0 = 16
        c1 = c // c0
        c1 = _clip_num(c1)
        shape_aa = [n, c1 * h * w * c0]
        shape_bb = [c1 * h * w * c0, c1 * h * w * c0]

    if input_x2.get("format") == "FRACTAL_NZ":
        shape_aa = [shape_bb[0], shape_bb[0]]

    if input_x1.get("format") == "FRACTAL_NZ":
        shape_bb = [shape_aa[1], shape_aa[1]]

    shape_aa = list(shape_aa)
    shape_bb = list(shape_bb)

    shape_aa = _get_input_shape(shape_aa)
    shape_bb = _get_input_shape(shape_bb)

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_aa)
    util.check_shape_rule(shape_bb)
    util.check_shape_size(shape_aa, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_bb, SHAPE_SIZE_LIMIT)

    shape_aa = [shape_aa[1], shape_aa[0]]
    trans_a = bool(1 - trans_a)

    shape_bb = [shape_bb[1], shape_bb[0]]
    trans_b = bool(1 - trans_b)

    shape_bbias = ()
    if bias is not None and bool(bias):
        shape_bbias = bias.get("shape")
        shape_bbias = list(shape_bbias)
        shape_bbias = _get_bias(shape_bbias)

    src_dtype = input_x1.get("dtype").lower()
    _shape_check(shape_aa, shape_bb, shape_bbias, src_dtype, trans_a, trans_b)

    m_shape = shape_aa[len(shape_aa) - 2]
    km_shape = shape_aa[len(shape_aa) - 1]
    kn_shape = shape_bb[len(shape_aa) - 2]
    n_shape = shape_bb[len(shape_aa) - 1]

    if src_dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE

    block_in0, block_out0 = _get_block(trans_a, trans_b, m_shape, n_shape, km_shape, kn_shape)

    if trans_a:
        shape_aa_tmp = (m_shape // block_reduce, km_shape // block_in0, block_reduce, block_in0)
    else:
        shape_aa_tmp = (m_shape // block_in0, km_shape // block_reduce, block_in0, block_reduce)

    if trans_b:
        shape_bb_tmp = (kn_shape // block_out0, n_shape // block_reduce, block_reduce, block_out0)
    else:
        shape_bb_tmp = (kn_shape // block_reduce, n_shape // block_out0, block_out0, block_reduce)
    shape_aa_tmp = (shape_aa_tmp[0], shape_aa_tmp[1], shape_aa_tmp[2], shape_aa_tmp[3])
    shape_bb_tmp = (shape_bb_tmp[0], shape_bb_tmp[1], shape_bb_tmp[2], shape_bb_tmp[3])

    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
    input_x1 = tik_instance.Tensor(input_x1.get("dtype"), shape_aa_tmp, name="left_matrix", scope=tik.scope_gm)
    input_x2 = tik_instance.Tensor(input_x2.get("dtype"), shape_bb_tmp, name="right_matrix", scope=tik.scope_gm)
    res_matmul0 = tik_instance.Tensor(output_y.get("dtype"), output_y.get("shape"), name="output", scope=tik.scope_gm)
    mo_tile, ko_tile, no_tile, dig_opt = get_cus_tile_info(input_x1, input_x2, 128)
    cus_cube_matmul_cast(tik_instance, input_x1, trans_a, input_x2, trans_b, res_matmul0,
                         mo_tile=mo_tile, ko_tile=ko_tile, no_tile=no_tile,
                         diag_opt=dig_opt, diag_size=128)
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x1, input_x2], outputs=[res_matmul0])
    return tik_instance


def get_cus_tile_info(input_x1, input_x2, diag_size):
    """get_cus_tile_info"""
    tile_map = {
        ((32, 32, 16, 16), (128, 32, 16, 16)): (8, 8, 16),
        ((8, 8, 16, 16), (72, 8, 16, 16)): (8, 8, 4),
        ((32, 32, 16, 16), (288, 32, 16, 16)): (8, 8, 12),
        ((128, 128, 16, 16), (32, 128, 16, 16)): (8, 8, 16),
        ((16, 16, 16, 16), (144, 16, 16, 16)): (8, 8, 9),
        ((64, 64, 16, 16), (16, 64, 16, 16)): (8, 8, 4),
        ((16, 16, 16, 16), (64, 16, 16, 16)): (8, 8, 4),
        ((32, 32, 16, 16), (8, 32, 16, 16)): (8, 8, 1),
        ((128, 128, 16, 16), (64, 128, 16, 16)): (8, 8, 16),
        ((16, 16, 16, 16), (4, 16, 16, 16)): (8, 8, 1),
        ((16, 16, 16, 16), (32, 16, 16, 16)): (8, 8, 2),
        ((64, 64, 16, 16), (32, 64, 16, 16)): (8, 8, 8),
        ((32, 32, 16, 16), (64, 32, 16, 16)): (8, 8, 8),
        ((32, 32, 16, 16), (16, 32, 16, 16)): (8, 8, 2),
        ((8, 8, 16, 16), (32, 8, 16, 16)): (8, 8, 1),
        ((8, 8, 16, 16), (16, 8, 16, 16)): (4, 8, 1),
        ((4, 4, 16, 16), (16, 4, 16, 16)): (2, 4, 1),
        ((4, 4, 16, 16), (4, 4, 16, 16)): (1, 4, 1),
        ((4, 4, 16, 16), (36, 4, 16, 16)): (2, 4, 3),
        ((4, 4, 16, 16), (49, 4, 16, 16)): (1, 4, 7)
    }
    shape_info = (tuple(input_x1.shape), tuple(input_x2.shape))
    diag_opt = False
    if input_x1.shape[0] * input_x1.shape[3] > diag_size:
        diag_opt = True
    if shape_info not in tile_map:
        raise ValueError("shape %s is not supported" % str(shape_info))
    mo_tile, ko_tile, no_tile = tile_map.get(shape_info)
    cus_tile_info = collections.namedtuple('cus_tile_info', ['mo_tile', 'ko_tile', 'no_tile', 'diag_opt'])
    return cus_tile_info(mo_tile, ko_tile, no_tile, diag_opt)


def cus_cube_matmul_cast(tik_instance, input_x1, trans_a, input_x2, trans_b,
                         res, mo_tile, ko_tile, no_tile, diag_opt=False, diag_size=128):
    """cus_cube_matmul_cast"""
    ko, mo, _, _ = input_x1.shape
    no, ko, _, _ = input_x2.shape
    c0 = input_x1.shape[-1]
    diag_outer = diag_size // c0
    blocksize = 32
    vectorfp32_size = 64
    if [input_x1.shape[-1], input_x1.shape[-2], input_x2.shape[-1], input_x2.shape[-2]] != [c0, c0, c0, c0]:
        raise ValueError("shape of input_x1 or input_x2 is not supported!")
    if not trans_a or not trans_b:
        raise ValueError("only trans_a=False and trans_b=False be supported!")

    core_m_num = mo // mo_tile
    loop_n_num = no // no_tile
    if loop_n_num * core_m_num <= 32:
        core_n_num = loop_n_num
    else:
        core_n_num = 32 // core_m_num
    if core_n_num > 0 and loop_n_num % core_n_num == 0:
        loop_n_num = loop_n_num // core_n_num
    else:
        raise ValueError("Does not support this scenario!")
    block_num = core_m_num * core_n_num

    loop_k_num = ko // ko_tile
    if diag_opt:
        loop_k_num = diag_outer // ko_tile
    # double buffer:
    thread_num_k = 2
    loop_k_num *= thread_num_k
    ko_tile_inner = ko_tile // thread_num_k
    with tik_instance.for_range(0, block_num, block_num=block_num) as block_idx, \
            tik_instance.for_range(0, loop_n_num) as cc_n:
        core_m = block_idx // core_n_num
        core_n = block_idx % core_n_num
        res_l0c = tik_instance.Tensor("float32", [no_tile, mo_tile, c0, c0],
                                      name="resMatmul_L0C", scope=tik.scope_cc)
        with tik_instance.for_range(0, loop_k_num, thread_num=thread_num_k) as thread_idx_k:
            # input_x2 -> input_x2_ub -(fp322fp16)-> input_x2_cast_ub -> input_x2_l1
            input_x2_ub = tik_instance.Tensor("float32", [no_tile, ko_tile_inner, c0, c0], name="input_x2_ub",
                                              scope=tik.scope_ubuf)
            if diag_opt:
                k_idx = core_m * mo_tile + thread_idx_k * ko_tile_inner
            else:
                k_idx = thread_idx_k * ko_tile_inner
            tik_instance.data_move(input_x2_ub,
                                   input_x2[(core_n * loop_n_num + cc_n) * no_tile,
                                            k_idx, 0, 0],
                                   0, no_tile, ko_tile_inner * c0 * c0 * 4 // blocksize,
                                   (ko - ko_tile_inner) * c0 * c0 * 4 // blocksize, 0)
            input_x2_cast_ub = tik_instance.Tensor("float16", [no_tile, ko_tile_inner, c0, c0],
                                                   name="input_x2_cast_ub", scope=tik.scope_ubuf)
            repeate_num = no_tile * ko_tile_inner * c0 * c0 // vectorfp32_size
            repeate_times_max = 255
            count = 0
            while repeate_num > repeate_times_max:
                tik_instance.vconv(vectorfp32_size, 'none',
                                   input_x2_cast_ub[count * repeate_times_max * vectorfp32_size],
                                   input_x2_ub[count * repeate_times_max * vectorfp32_size],
                                   repeate_times_max,
                                   1, 1, 4, 8)
                repeate_num -= repeate_times_max
                count += 1
            tik_instance.vconv(vectorfp32_size, 'none',
                               input_x2_cast_ub[count * repeate_times_max * vectorfp32_size],
                               input_x2_ub[count * repeate_times_max * vectorfp32_size], repeate_num,
                               1, 1, 4, 8)
            input_x2_l1 = tik_instance.Tensor("float16", [no_tile, ko_tile_inner, c0, c0],
                                              name="input_x2_l1", scope=tik.scope_cbuf)
            tik_instance.data_move(input_x2_l1, input_x2_cast_ub, 0, 1,
                                   no_tile * ko_tile_inner * c0 * c0 * 2 // blocksize, 0, 0)
            # input_x1 -> input_x1_l1
            input_x1_l1 = tik_instance.Tensor(input_x1.dtype, [ko_tile_inner, mo_tile, c0, c0],
                                              name="input_x1_l1", scope=tik.scope_cbuf)
            tik_instance.data_move(input_x1_l1,
                                   input_x1[k_idx,
                                            core_m * mo_tile, 0, 0],
                                   0, ko_tile_inner, mo_tile * c0 * c0 * 2 // blocksize,
                                   (mo - mo_tile) * c0 * c0 * 2 // blocksize, 0)
            # input_x2_l1 -> input_x2_l0b
            input_x2_l0b = tik_instance.Tensor("float16", [ko_tile_inner, no_tile, c0, c0],
                                               name="input_x2_l0b", scope=tik.scope_cb)
            with tik_instance.for_range(0, ko_tile_inner) as cc2:
                tik_instance.load2dv1(input_x2_l0b[cc2, 0, 0, 0], input_x2_l1[0, cc2, 0, 0], 0, no_tile,
                                      ko_tile_inner,
                                      0, True)
            # input_x1_l1 -> input_x1_l0a
            input_x1_l0a = tik_instance.Tensor(input_x1.dtype, [mo_tile, ko_tile_inner, c0, c0],
                                               name="input_x1_l0a", scope=tik.scope_ca)
            with tik_instance.for_range(0, mo_tile) as cc1:
                tik_instance.load2dv1(input_x1_l0a[cc1, 0, 0, 0], input_x1_l1[0, cc1, 0, 0], 0, ko_tile_inner,
                                      mo_tile, 0, False)
            with tik_instance.if_scope(thread_idx_k == 0):
                tik_instance.mmad(res_l0c, input_x1_l0a, input_x2_l0b, mo_tile * c0,
                                  ko_tile_inner * c0, no_tile * c0, 0)
            with tik_instance.else_scope():
                tik_instance.mmad(res_l0c, input_x1_l0a, input_x2_l0b, mo_tile * c0,
                                  ko_tile_inner * c0, no_tile * c0, 1)
        res_ub = tik_instance.Tensor(input_x1.dtype, [no_tile, mo_tile, c0, c0],
                                     name="resMatmul_ub", scope=tik.scope_ubuf)
        tik_instance.data_move(res_ub, res_l0c, 0, 1, no_tile * mo_tile, 0, 0, 1)
        tik_instance.data_move(res[(core_n * loop_n_num + cc_n) * no_tile, core_m * mo_tile, 0, 0],
                               res_ub, 0, no_tile,
                               mo_tile * c0 * c0 * 2 // blocksize, 0,
                               (mo - mo_tile) * c0 * c0 * 2 // blocksize)
