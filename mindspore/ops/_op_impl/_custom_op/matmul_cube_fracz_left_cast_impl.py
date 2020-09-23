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
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
import te.platform.cce_params as cce
from te import tik
from topi.cce import util

# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
NoneType = type(None)

matmul_cube_fracz_left_cast_op_info = TBERegOp("CusMatMulCubeFraczLeftCast") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmulcubefraczleftcast.so") \
    .compute_cost(10) \
    .kernel_name("CusMatMulCubeFraczLeftCast") \
    .partial_flag(True) \
    .input(0, "x1", False, "required", "all") \
    .input(1, "x2", False, "required", "all") \
    .input(2, "x3", False, "optional", "all") \
    .output(0, "y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F32_FracZ, DataType.F16_Default, DataType.F16_FracZ) \
    .get_op_info()


def _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a, trans_b):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
src_dtype: str
            The data type of input, support "float32", "float16"
    trans_a: bool
            If True, shape_a == transposed before multiplication
    trans_b: bool
            If True, shape_b == transposed before multiplication

    Returns None
    """
    shape_len = len(shape_a)
    src_dtype = src_dtype.lower()
    k_block_size = cce.BLOCK_REDUCE

    check_list = ("float16")

    if src_dtype not in check_list:
        raise RuntimeError("matmul_cce only support %s while src_dtype == %s"
                           % (",".join(check_list), src_dtype))
    if shape_len != len(shape_b):
        raise RuntimeError("length of a and b are not equal")

    if shape_len != 2:
        raise RuntimeError(
            "length of shape must be 2, more than 2 dimensions should use batch_matmul now!")

    is_gevm = True if shape_a[-2] == 1 or shape_a[-1] == 1 else False
    is_gemv = True if shape_b[-2] == 1 or shape_b[-1] == 1 else False

    if trans_a:
        m_shape = shape_a[shape_len - 1]
        km_shape = shape_a[shape_len - 2]
    else:
        m_shape = shape_a[shape_len - 2]
        km_shape = shape_a[shape_len - 1]

    if trans_b:
        kn_shape = shape_b[shape_len - 1]
        n_shape = shape_b[shape_len - 2]
    else:
        kn_shape = shape_b[shape_len - 2]
        n_shape = shape_b[shape_len - 1]

    if m_shape == 1:
        if n_shape == 1:
            raise RuntimeError("input shape M and N can't both be 1")

    if km_shape != kn_shape:
        print(km_shape, kn_shape)
        raise RuntimeError("reduce axis not same")

    if m_shape % cce.BLOCK_IN != 0 and m_shape != 1:
        raise RuntimeError(
            "input shape M should be 1 or multiple of %d" % cce.BLOCK_IN)

    if m_shape != 1:
        if n_shape == 1:
            if km_shape % (cce.BLOCK_IN * cce.BLOCK_IN) != 0:
                raise RuntimeError("input shape K1 should be multiple of %d"
                                   % (cce.BLOCK_IN * cce.BLOCK_IN))
        elif km_shape % k_block_size != 0:
            raise RuntimeError(
                "input shape K1 should be multiple of %d" % cce.BLOCK_IN)
    else:
        if km_shape % (cce.BLOCK_IN * cce.BLOCK_IN) != 0:
            raise RuntimeError("input shape K1 should be multiple of %d"
                               % (cce.BLOCK_IN * cce.BLOCK_IN))

    if n_shape % cce.BLOCK_IN != 0 and n_shape != 1:
        raise RuntimeError("input shape N should be 1 or multiple of %d" % cce.BLOCK_IN)

    if shape_bias:
        if len(shape_bias) == 1:
            if is_gevm or is_gemv:
                if shape_bias[0] != m_shape * n_shape:
                    raise RuntimeError("broadcast case shape bias for gemv must be equal m*n")
            else:
                if shape_bias[0] != n_shape:
                    raise RuntimeError("broadcast bias shape must be equal to shape n")
        elif len(shape_bias) == shape_len:
            if [i for i in shape_bias[-2:]] != [m_shape, n_shape]:
                raise RuntimeError("non broadcast bias shape must be same as output shape")
        else:
            raise RuntimeError("Unsupported input shape now for batch bias case")


def _get_bias(shape_bias):
    """_get_bias"""
    bias_length = shape_bias[0]
    if bias_length % 16 == 0:
        return shape_bias
    bias_length = (bias_length // 16) * 16 + 16
    shape_bias = []
    shape_bias.append(bias_length)
    return shape_bias


def _get_input_shape(shape_x):
    """_get_input_shape"""
    dim_a = shape_x[0]
    dim_b = shape_x[1]
    res = []
    if dim_a % 16 != 0:
        dim_a = (dim_a // 16) * 16 + 16
        res.append(dim_a)
    else:
        res.append(dim_a)

    if dim_b % 16 != 0:
        dim_b = (dim_b // 16) * 16 + 16
        res.append(dim_b)
    else:
        res.append(dim_b)
    return res


def check_supported(input_x1, input_x2, bias=None, output_y={}, trans_a=False, trans_b=False, kernel_name="matmulcube"):
    """check_supported"""
    shape_a = input_x1.get("shape")
    shape_b = input_x2.get("shape")
    print("shape_a: ", shape_a)
    print("shape_b: ", shape_b)
    src_dtype = input_x1.get("dtype")
    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)
    util.check_shape_size(shape_a, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_b, SHAPE_SIZE_LIMIT)
    try:
        trans_a_f = bool(1 - trans_a)
        if src_dtype in ("floate32", "int32"):
            if len(shape_a) != 2 and len(shape_b) != 2:
                return False
            if trans_b:
                if shape_b[0] == 1:
                    return False
            else:
                if shape_b[1] == 1:
                    return False
            if trans_a:
                if trans_b:
                    if shape_a[0] != shape_b[1]:
                        return False
                elif shape_a[0] != shape_b[0]:
                    return False
            elif trans_b:
                if shape_a[1] != shape_b[1]:
                    return False
            elif shape_a[1] != shape_b[0]:
                return False

            if trans_a_f and trans_b and shape_b[1] == 1:
                return False

        if src_dtype == "float16":
            if len(shape_a) != 2 and len(shape_b) != 2:
                return False

            if trans_a:
                m_shape = shape_a[1]
                k_shape = shape_a[0]
            else:
                m_shape = shape_a[0]
                k_shape = shape_a[1]

            if trans_b:
                n_shape = shape_b[0]
                k_b_shape = shape_b[1]
            else:
                n_shape = shape_b[1]
                k_b_shape = shape_b[0]

            if k_shape != k_b_shape:
                return False

            if m_shape == 1 or n_shape == 1:
                if k_shape % 256 != 0:
                    return False

    except RuntimeError as e:
        print(e)
        return False

    return True


@op_info_register(matmul_cube_fracz_left_cast_op_info)
def CusMatMulCubeFraczLeftCast(input_x1, input_x2, bias=None, output_y={}, trans_a=False, trans_b=False,
                               kernel_name="CusMatMulCubeFraczLeftCast"):
    """
    calculating  matrix multiplication with bias, C = A*B + bias, support input
    data with fractal format.

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    src_dtype: str
            The data type of input, support "float32", "float16"
    dst_dtype: str
            The data type of output, support "float32", "float16"
    trans_a: bool
            If True, shape_a == transposed before multiplication
    trans_b: bool
            If True, shape_b == transposed before multiplication
    is_fractal: bool
            If True, the input data format of a and b must be fractal format
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND

    Returns
    -------
    None
    """
    shape_a = input_x1.get("ori_shape")
    shape_b = input_x2.get("ori_shape")
    print("============")
    print(input_x1.get("format"), input_x2.get("format"))
    print(shape_a, shape_b)
    print("============")
    if input_x2.get("format") == "FRACTAL_Z":
        n, c, h, w = shape_b
        c0 = 16
        c1 = c // c0
        if c1 == 0:
            c1 = 1
        shape_b = [n, c1 * h * w * c0]
        shape_a = [n, n]

    if input_x1.get("format") == "FRACTAL_Z":
        n, c, h, w = shape_a
        c0 = 16
        c1 = c // c0
        if c1 == 0:
            c1 = 1
        shape_a = [n, c1 * h * w * c0]
        shape_b = [c1 * h * w * c0, c1 * h * w * c0]

    if input_x2.get("format") == "FRACTAL_NZ":
        shape_a = [shape_b[0], shape_b[0]]
        shape_b = shape_b

    if input_x1.get("format") == "FRACTAL_NZ":
        shape_a = shape_a
        shape_b = [shape_a[1], shape_a[1]]

    shape_a = list(shape_a)
    shape_b = list(shape_b)

    shape_a = _get_input_shape(shape_a)
    shape_b = _get_input_shape(shape_b)

    util.check_kernel_name(kernel_name)
    util.check_shape_rule(shape_a)
    util.check_shape_rule(shape_b)
    util.check_shape_size(shape_a, SHAPE_SIZE_LIMIT)
    util.check_shape_size(shape_b, SHAPE_SIZE_LIMIT)

    shape_a = [shape_a[1], shape_a[0]]
    trans_a = bool(1 - trans_a)

    shape_b = [shape_b[1], shape_b[0]]
    trans_b = bool(1 - trans_b)

    shape_bias = ()
    if bias is not None and bool(bias):
        shape_bias = bias.get("shape")
        shape_bias = list(shape_bias)
        shape_bias = _get_bias(shape_bias)

    src_dtype = input_x1.get("dtype").lower()
    _shape_check(shape_a, shape_b, shape_bias, src_dtype, trans_a, trans_b)

    m_shape = shape_a[len(shape_a) - 2]
    km_shape = shape_a[len(shape_a) - 1]
    kn_shape = shape_b[len(shape_a) - 2]
    n_shape = shape_b[len(shape_a) - 1]

    if src_dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE

    block_in = cce.BLOCK_IN
    block_out = cce.BLOCK_OUT

    if trans_a and km_shape == 1:
        block_in = cce.BLOCK_VECTOR

    if not trans_a and m_shape == 1:
        block_in = cce.BLOCK_VECTOR

    if trans_b and kn_shape == 1:
        block_out = cce.BLOCK_VECTOR

    if not trans_b and n_shape == 1:
        block_out = cce.BLOCK_VECTOR

    if trans_a:
        shape_a_temp = (m_shape // block_reduce, km_shape // block_in, block_reduce, block_in)
    else:
        shape_a_temp = (m_shape // block_in, km_shape // block_reduce, block_in, block_reduce)

    if trans_b:
        shape_b_temp = (kn_shape // block_out, n_shape // block_reduce, block_reduce, block_out)
    else:
        shape_b_temp = (kn_shape // block_reduce, n_shape // block_out, block_out, block_reduce)
    shape_a_temp = (shape_a_temp[0], shape_a_temp[1], shape_a_temp[2], shape_a_temp[3])
    shape_b_temp = (shape_b_temp[0], shape_b_temp[1], shape_b_temp[2], shape_b_temp[3])

    if util.get_product_version() == util.VERSION_MINI:
        tik_instance = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_instance = tik.Tik(tik.Dprofile("v100", "cloud"))
    input_x1 = tik_instance.Tensor(input_x1.get("dtype"), shape_a_temp, name="left_matrix", scope=tik.scope_gm)
    input_x2 = tik_instance.Tensor(input_x2.get("dtype"), shape_b_temp, name="right_matrix", scope=tik.scope_gm)
    res_matmul = tik_instance.Tensor(output_y.get("dtype"), output_y.get("shape"), name="output", scope=tik.scope_gm)
    DIAG_SIZE = 128
    mo_tile, ko_tile, no_tile, diag_opt = get_cus_tile_info(input_x1, input_x2, DIAG_SIZE)
    cus_cube_matmul_cast(tik_instance, input_x1, trans_a, input_x2, trans_b, res_matmul,
                         mo_tile=mo_tile, ko_tile=ko_tile, no_tile=no_tile,
                         diag_opt=diag_opt, diag_size=DIAG_SIZE)
    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_x1, input_x2], outputs=[res_matmul])
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
    mo_tile, ko_tile, no_tile = tile_map[shape_info]
    return mo_tile, ko_tile, no_tile, diag_opt


def cus_cube_matmul_cast(tik_instance, input_x1, trans_a, input_x2, trans_b,
                         res, mo_tile, ko_tile, no_tile, diag_opt=False, diag_size=128):
    """cus_cube_matmul_cast"""
    ko, mo, _, _ = input_x1.shape
    no, ko, _, _ = input_x2.shape
    c0 = input_x1.shape[-1]
    diag_outer = diag_size // c0
    maxblocknum = 32
    fp32_size = 4
    fp16_size = 2
    blocksize = 32
    vectorfp32_size = 64
    if [input_x1.shape[-1], input_x1.shape[-2], input_x2.shape[-1], input_x2.shape[-2]] != [c0, c0, c0, c0]:
        raise ValueError("shape of input_x1 or input_x2 is not supported!")
    if not trans_a or not trans_b:
        raise ValueError("only trans_a=False and trans_b=False be supported!")

    core_m_num = mo // mo_tile
    loop_n_num = no // no_tile
    if loop_n_num * core_m_num <= maxblocknum:
        core_n_num = loop_n_num
    else:
        core_n_num = maxblocknum // core_m_num
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
    with tik_instance.for_range(0, block_num, block_num=block_num) as block_idx:
        core_m = block_idx // core_n_num
        core_n = block_idx % core_n_num
        with tik_instance.for_range(0, loop_n_num) as cc_n:
            res_L0C = tik_instance.Tensor("float32", [no_tile, mo_tile, c0, c0],
                                          name="resMatmul_L0C", scope=tik.scope_cc)
            with tik_instance.for_range(0, loop_k_num, thread_num=thread_num_k) as thread_idx_k:
                # input_x2 -> input_x2_ub -(fp322fp16)-> input_x2_cast_ub -> input_x2_L1
                input_x2_ub = tik_instance.Tensor("float32", [no_tile, ko_tile_inner, c0, c0], name="input_x2_ub",
                                                  scope=tik.scope_ubuf)
                if diag_opt:
                    k_idx = core_m * mo_tile + thread_idx_k * ko_tile_inner
                else:
                    k_idx = thread_idx_k * ko_tile_inner
                tik_instance.data_move(input_x2_ub,
                                       input_x2[(core_n * loop_n_num + cc_n) * no_tile,
                                                k_idx, 0, 0],
                                       0, no_tile, ko_tile_inner * c0 * c0 * fp32_size // blocksize,
                                       (ko - ko_tile_inner) * c0 * c0 * fp32_size // blocksize, 0)
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
                input_x2_L1 = tik_instance.Tensor("float16", [no_tile, ko_tile_inner, c0, c0],
                                                  name="input_x2_L1", scope=tik.scope_cbuf)
                tik_instance.data_move(input_x2_L1, input_x2_cast_ub, 0, 1,
                                       no_tile * ko_tile_inner * c0 * c0 * fp16_size // blocksize, 0, 0)
                # input_x1 -> input_x1_L1
                input_x1_L1 = tik_instance.Tensor(input_x1.dtype, [ko_tile_inner, mo_tile, c0, c0],
                                                  name="input_x1_L1", scope=tik.scope_cbuf)
                tik_instance.data_move(input_x1_L1,
                                       input_x1[k_idx,
                                                core_m * mo_tile, 0, 0],
                                       0, ko_tile_inner, mo_tile * c0 * c0 * fp16_size // blocksize,
                                       (mo - mo_tile) * c0 * c0 * fp16_size // blocksize, 0)
                # input_x2_L1 -> input_x2_L0B
                input_x2_L0B = tik_instance.Tensor("float16", [ko_tile_inner, no_tile, c0, c0],
                                                   name="input_x2_L0B", scope=tik.scope_cb)
                with tik_instance.for_range(0, ko_tile_inner) as cc2:
                    tik_instance.load2dv1(input_x2_L0B[cc2, 0, 0, 0], input_x2_L1[0, cc2, 0, 0], 0, no_tile,
                                          ko_tile_inner,
                                          0, True)
                # input_x1_L1 -> input_x1_L0A
                input_x1_L0A = tik_instance.Tensor(input_x1.dtype, [mo_tile, ko_tile_inner, c0, c0],
                                                   name="input_x1_L0A", scope=tik.scope_ca)
                with tik_instance.for_range(0, mo_tile) as cc1:
                    tik_instance.load2dv1(input_x1_L0A[cc1, 0, 0, 0], input_x1_L1[0, cc1, 0, 0], 0, ko_tile_inner,
                                          mo_tile, 0, False)
                with tik_instance.if_scope(thread_idx_k == 0):
                    tik_instance.mmad(res_L0C, input_x1_L0A, input_x2_L0B, mo_tile * c0,
                                      ko_tile_inner * c0, no_tile * c0, 0)
                with tik_instance.else_scope():
                    tik_instance.mmad(res_L0C, input_x1_L0A, input_x2_L0B, mo_tile * c0,
                                      ko_tile_inner * c0, no_tile * c0, 1)
            res_ub = tik_instance.Tensor(input_x1.dtype, [no_tile, mo_tile, c0, c0],
                                         name="resMatmul_ub", scope=tik.scope_ubuf)
            tik_instance.data_move(res_ub, res_L0C, 0, 1, no_tile * mo_tile, 0, 0, 1)
            tik_instance.data_move(res[(core_n * loop_n_num + cc_n) * no_tile, core_m * mo_tile, 0, 0],
                                   res_ub, 0, no_tile,
                                   mo_tile * c0 * c0 * fp16_size // blocksize, 0,
                                   (mo - mo_tile) * c0 * c0 * fp16_size // blocksize)
