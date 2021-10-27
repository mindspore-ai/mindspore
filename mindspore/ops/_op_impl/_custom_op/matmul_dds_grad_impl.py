# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""matmul dds impl"""
from te import tik
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

matmul_dds_grad_op_info = TBERegOp("MatmulDDSGrad") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmul_dds_grad.so") \
    .compute_cost(10) \
    .kernel_name("matmul_dds_grad") \
    .partial_flag(True) \
    .input(0, "q", False, "required", "all") \
    .input(1, "k", False, "required", "all") \
    .input(2, "local_prob", False, "required", "all") \
    .input(3, "global_prob", False, "required", "all") \
    .input(4, "local_prob_grad", False, "required", "all") \
    .input(5, "global_prob_grad", False, "required", "all") \
    .output(0, "dq", False, "required", "all") \
    .output(1, "dk", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(matmul_dds_grad_op_info)
def matmul_dds_grad(q,
                    k,
                    local_prob,
                    global_prob,
                    local_prob_grad,
                    global_prob_grad,
                    dq,
                    dk,
                    kernel_name="matmul_dds_grad"):
    """
    :param q: the dict of input q (bs*seq_len, embedding_size) zN
    :param k: the dict of input k (bs*seq_len, embedding_size) nZ
    :param local_mask:  the dict of input mask local  (bs*16*64, 64) zN
    :param global_mask:   the dict of input mask global   (heads*1024, 256)  zN
    :param local_prob: local output (bs, heads, block_num, block_size // 16, block_size // 16, 16, 16) zN
    :param global_prob: global output (bs, heads, block_num, global_size // 16, block_size // 16, 16, 16) zN
    :param local_prob_grad: local output grad (bs, heads, block_num, block_size // 16, block_size // 16, 16, 16) zN
    :param global_prob_grad: global output grad (bs, heads, block_num, global_size // 16, block_size // 16, 16, 16) zN
    """

    shape_q = q.get(
        'shape')
    shape_lc = local_prob.get(
        'shape')
    shape_gc = global_prob.get(
        'shape')
    bs = shape_lc[0]
    heads = shape_gc[1]
    global_size = shape_gc[3] * shape_gc[-1]
    block_size = shape_lc[4] * shape_lc[5]
    seq_len = shape_q[1] * shape_q[2] // bs
    block_num = seq_len // block_size
    size_per_head = shape_q[0] * shape_q[-1] // heads

    tik_inst = tik.Tik(tik.Dprofile('v100', 'cloud'))
    mat_q = tik_inst.Tensor("float16", (size_per_head * heads // 16, bs * seq_len // 16, 16, 16),
                            name="mat_q",
                            scope=tik.scope_gm)  # zN
    mat_k = tik_inst.Tensor("float16", (size_per_head * heads // 16, bs * seq_len // 16, 16, 16),
                            name="mat_k",
                            scope=tik.scope_gm)  # nZ
    mat_lc = tik_inst.Tensor("float16", (bs, heads, block_num, block_size // 16, block_size // 16, 16, 16),
                             name="mat_lc",
                             scope=tik.scope_gm)  # zN
    mat_gc = tik_inst.Tensor("float16", (bs, heads, block_num, global_size // 16, block_size // 16, 16, 16),
                             name="mat_gc",
                             scope=tik.scope_gm)  # zN
    mat_lc_grad = tik_inst.Tensor("float16", (bs, heads, block_num, block_size // 16, block_size // 16, 16, 16),
                                  name="mat_lc_grad",
                                  scope=tik.scope_gm)  # zN
    mat_gc_grad = tik_inst.Tensor("float16", (bs, heads, block_num, global_size // 16, block_size // 16, 16, 16),
                                  name="mat_gc_grad",
                                  scope=tik.scope_gm)  # zN
    mat_dq = tik_inst.Tensor("float16", (size_per_head * heads // 16, bs * seq_len // 16, 16, 16),
                             name="mat_dq",
                             scope=tik.scope_gm)  # zN
    mat_dk = tik_inst.Tensor("float16", (bs * seq_len // 16, size_per_head * heads // 16, 16, 16),
                             name="mat_dk",
                             scope=tik.scope_gm)  # zN

    channel_num = bs * heads
    with tik_inst.for_range(0, channel_num, block_num=channel_num) as block_index:
        # apply for tensor in L1 for fp 16 ones-like result (16, 320) zZ
        mat_l1_ones = tik_inst.Tensor("float16", (1, (global_size + block_size) // 16, 16, 16),
                                      name='mat_l1_ones',
                                      scope=tik.scope_cbuf)
        with tik_inst.new_stmt_scope():
            mat_ub_ones = tik_inst.Tensor("float16", (1, (global_size + block_size) // 16, 16, 16),
                                          name='mat_ub_ones',
                                          scope=tik.scope_ubuf)
            tik_inst.vec_dup(128, mat_ub_ones, 1.0,
                             (global_size + block_size) * 16 // 128, 8)
            tik_inst.data_move(mat_l1_ones[0, 0, 0, 0], mat_ub_ones[0, 0, 0, 0],
                               0, (global_size + block_size) // 16, 16, 0, 0)

        b = tik_inst.Scalar(dtype="int32")
        b.set_as(block_index // heads)

        head = tik_inst.Scalar(dtype="int32")
        head.set_as(block_index - b * heads)

        s = tik_inst.Scalar(dtype="int32")
        s.set_as(head // 4)
        # formula: global_idx = 3 - (head - 4 * s)  # global idx for global key extraction
        global_idx = tik_inst.Scalar(dtype="int32")
        global_idx.set_as(3 - (head - 4 * s))
        # apply tensor in l1 for global k (256, 128) nZ
        mat_l1_gk = tik_inst.Tensor("float16",
                                    (global_size // 16, size_per_head // 16, 16, 16),
                                    name="mat_l1_gk",
                                    scope=tik.scope_cbuf)
        # apply for tensor in L0C for global dk (128, 256) zN
        mat_l0c_dkg = tik_inst.Tensor("float32",
                                      (global_size // 16,
                                       size_per_head // 16, 16, 16),
                                      name="mat_l0c_dkg",
                                      scope=tik.scope_cc)
        with tik_inst.for_range(0, global_size // 16) as gb:
            # move global key from gm to L1 nZ
            # the shape of k is nZ, move (16, 256) in one loop, the stride between each (16, 16) is 3*(16,16)
            tik_inst.data_move(mat_l1_gk[gb, 0, 0, 0],
                               mat_k[
                                   head * size_per_head // 16, b * seq_len // 16 +
                                   global_idx + gb * block_size // 16, 0, 0],
                               0, size_per_head // 16, 16, bs * seq_len - 16, 0)
        with tik_inst.for_range(0, block_num) as block:
            # do backward softmax
            # formula: grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax
            # apply for tensor in ub for grad_x out (64, 320) zN
            mat_ub_lg_d = tik_inst.Tensor("float16",
                                          ((global_size + block_size) //
                                           16, block_size // 16, 16, 16),
                                          name='mat_ub_lg_d',
                                          scope=tik.scope_ubuf)
            with tik_inst.new_stmt_scope():
                # apply for tensor in ub for softmax out (64, 320) zN
                mat_ub_lg = tik_inst.Tensor("float16", ((global_size + block_size) // 16, block_size // 16, 16, 16),
                                            name='mat_ub_lg',
                                            scope=tik.scope_ubuf)
                # apply for tensor in ub for softmax out grad (64, 320) zN
                mat_ub_lg_grad = tik_inst.Tensor("float16",
                                                 ((global_size + block_size) //
                                                  16, block_size // 16, 16, 16),
                                                 name='mat_ub_lg_grad',
                                                 scope=tik.scope_ubuf)
                # move local out from gm to ub zN
                # the shape of local out in gm is zN
                # the shape of local out in UB is zN
                # the stride between each (64, 16) is 0
                # repeat 4 times
                tik_inst.data_move(mat_ub_lg[0, 0, 0, 0], mat_lc[b, head, block, 0, 0, 0, 0], 0,
                                   block_size // 16, block_size,
                                   0, 0)
                # move global out from gm to ub zN
                # the shape of global out in gm is zN
                # the shape of global out in UB is zN
                # the stride between each (64, 16) is 0
                # repeat 16 times
                tik_inst.data_move(mat_ub_lg[block_size // 16, 0, 0, 0], mat_gc[b, head, block, 0, 0, 0, 0], 0,
                                   global_size // 16, block_size,
                                   0, 0)
                # move local out grad from gm to ub zN
                # the shape of local out grad in gm is zN
                # the shape of local out grad in UB is zN
                # the stride between each (64, 16) is 0
                # repeat 4 times
                tik_inst.data_move(mat_ub_lg_grad[0, 0, 0, 0], mat_lc_grad[b, head, block, 0, 0, 0, 0], 0,
                                   block_size // 16, block_size,
                                   0, 0)
                # move global out grad from gm to ub zN
                # the shape of global out grad in gm is zN
                # the shape of global out grad in UB is zN
                # the stride between each (64, 16) is 0
                # repeat 16 times
                tik_inst.data_move(mat_ub_lg_grad[block_size // 16, 0, 0, 0],
                                   mat_gc_grad[b, head, block, 0, 0, 0, 0], 0,
                                   global_size // 16, block_size,
                                   0, 0)
                # apply for tensor in ub for softmax multiply out grad (64, 320) zN
                mat_ub_ssg = tik_inst.Tensor("float16",
                                             ((global_size + block_size) //
                                              16, block_size // 16, 16, 16),
                                             name='mat_ub_ssg',
                                             scope=tik.scope_ubuf)
                # calculate softmax * softmax_grad
                tik_inst.vmul(128, mat_ub_ssg[0, 0, 0, 0], mat_ub_lg_grad[0, 0, 0, 0], mat_ub_lg[0, 0, 0, 0],
                              (global_size + block_size) * block_size // 128,
                              1, 1, 1, 8, 8, 8)

                # apply for tensor in L1 for dsoftmax*softmax result (320, 64) nZ
                mat_l1_ssg_nz = tik_inst.Tensor("float16", ((global_size + block_size) // 16,
                                                            block_size // 16, 16, 16),
                                                name='mat_l1_ssg_nz',
                                                scope=tik.scope_cbuf)
                # move ones from ub to L1 for CUBE mmad
                # the shape of ones in ub is nZ
                # the shape of ones in L0A is nZ
                # the stride between each (16, 16) is 0
                # repeat 32 times
                tik_inst.data_move(mat_l1_ssg_nz[0, 0, 0, 0], mat_ub_ssg[0, 0, 0, 0], 0,
                                   (global_size + block_size) // 16, block_size, 0, 0)
                # apply tensor in l0c for exp sum (16, 64) zN
                mat_l0c_ssg_sum = tik_inst.Tensor("float32", (block_size // 16, 1, 16, 16),
                                                  name='mat_l0c_ssg_sum',
                                                  scope=tik.scope_cc)
                # apply tensor in ub for exp sum (16, 64) zN
                mat_ub_ssg_sum = tik_inst.Tensor("float32", (block_size // 16, 1, 16, 16),
                                                 name='mat_ub_ssg_sum',
                                                 scope=tik.scope_ubuf)
                # apply for tensor in L0A for q (16, 320) zZ
                mat_l0a_ones = tik_inst.Tensor('float16', (1, (global_size + block_size) // 16, 16, 16),
                                               name='mat_l0a_ones', scope=tik.scope_ca)
                # apply for tensor in L0B for exp (320, 64) nZ
                mat_l0b_ssg = tik_inst.Tensor('float16', ((global_size + block_size) // 16, block_size // 16, 16, 16),
                                              name='mat_l0b_exp', scope=tik.scope_cb)
                # move ones from l1 to L0A for CUBE mmad
                # the shape of ones in l1 is zZ
                # the shape of ones in L0A is zZ
                # the stride between each (16, 16) is 0
                # repeat 32 times
                tik_inst.load2dv1(mat_l0a_ones[0, 0, 0, 0], mat_l1_ones[0, 0, 0, 0], 0,
                                  (global_size + block_size) * 16 // (16 * 16), 1, 0, False)
                # move ssg from l1 to L0B for CUBE mmad
                # the shape of ssg in l1 is nZ
                # the shape of ssg in L0B is nZ
                # the stride between each (16, 16) is 0
                # repeat 128 times
                tik_inst.load2dv1(mat_l0b_ssg[0, 0, 0, 0], mat_l1_ssg_nz[0, 0, 0, 0], 0,
                                  (global_size + block_size) * block_size // (16 * 16), 1, 0, False)
                tik_inst.mmad(mat_l0c_ssg_sum, mat_l0a_ones, mat_l0b_ssg,
                              16, (global_size + block_size), block_size, 0)
                tik_inst.data_move(mat_ub_ssg_sum[0, 0, 0, 0], mat_l0c_ssg_sum[0, 0, 0, 0], 0,
                                   block_size // 16, 1, 0, 0)
                # apply for tensor in UB for global prob sum (64,)
                mat_ub_ssg_sums = tik_inst.Tensor("float32", (block_size,),
                                                  name='mat_ub_ssg_sums',
                                                  scope=tik.scope_ubuf)
                tik_inst.data_move(mat_ub_ssg_sums[0], mat_ub_ssg_sum[0, 0, 0, 0],
                                   0, block_size // 16, 1 * 2, 15 * 2, 0)
                # apply for tensor in UB for global prob sum (64,)
                mat_ub_ssg_sums_16 = tik_inst.Tensor("float16", (block_size,),
                                                     name='mat_ub_ssg_sums_16',
                                                     scope=tik.scope_ubuf)
                # convert fp32 to fp16
                tik_inst.vec_conv(
                    64, "", mat_ub_ssg_sums_16[0], mat_ub_ssg_sums[0], 1, 4, 8)

                mat_ub_ssgs = tik_inst.Tensor("float16",
                                              ((global_size + block_size) //
                                               16, block_size // 16, 16, 16),
                                              name='mat_ub_ssgs',
                                              scope=tik.scope_ubuf)

                with tik_inst.for_range(0, block_size) as bbs:
                    # apply for scalar in UB for prob sum rec
                    sum_ssg = tik_inst.Scalar("float16",
                                              name='sum_ssg',
                                              init_value=0)
                    # set value for scalar prob sum rec
                    sum_ssg.set_as(mat_ub_ssg_sums_16[bbs])
                    tik_inst.vec_muls(16, mat_ub_ssgs[0, bbs // 16, bbs % 16, 0],
                                      mat_ub_lg[0, bbs // 16, bbs %
                                                16, 0], sum_ssg,
                                      (global_size + block_size) // 16,
                                      block_size, block_size)

                tik_inst.vsub(128, mat_ub_lg_d[0, 0, 0, 0], mat_ub_ssg[0, 0, 0, 0], mat_ub_ssgs[0, 0, 0, 0],
                              (global_size + block_size) * block_size // 128,
                              1, 1, 1, 8, 8, 8)

            # local dq calculation
            # dw X K.T
            # apply tensor in l1 for local k (64, 128) nZ
            mat_l1_lk = tik_inst.Tensor("float16",
                                        (block_size // 16,
                                         size_per_head // 16, 16, 16),
                                        name="mat_l1_lk",
                                        scope=tik.scope_cbuf)
            # move k from gm to l1
            # the shape of local k in gm is nZ
            # the shape of local k in l1 is zZ
            # the stride between each (16, 16) is 1024*bs-64
            # repeat 8 times
            # LOOP 4 times
            with tik_inst.for_range(0, block_size // 16) as lb:
                tik_inst.data_move(mat_l1_lk[lb, 0, 0, 0],
                                   mat_k[head * size_per_head // 16, b * seq_len // 16 + (
                                       block * block_size) // 16 + lb, 0, 0],
                                   0, size_per_head // 16, 16, bs * seq_len - 16, 0)

            # apply tensor in l1 for local dw (64, 128) zZ
            mat_l1_ldw = tik_inst.Tensor("float16",
                                         (block_size // 16,
                                          block_size // 16, 16, 16),
                                         name="mat_l1_ldw",
                                         scope=tik.scope_cbuf)
            # move local d-softmax from ub to l1
            # the shape of d-softmax in ub is zN
            # the shape of d-softmax in l1 is zZ
            # the stride between each (16, 64) is 0
            # repeat 16 times
            with tik_inst.for_range(0, block_size // 16) as lb:
                tik_inst.data_move(mat_l1_ldw[lb, 0, 0, 0],
                                   mat_ub_lg_d[0, lb, 0, 0],
                                   0, block_size // 16, 16, block_size - 16, 0)
            # apply for tensor in L0C for local d-q (64, 128) zN
            mat_l0c_dq = tik_inst.Tensor("float32",
                                         (size_per_head // 16,
                                          block_size // 16, 16, 16),
                                         name="mat_l0c_dq",
                                         scope=tik.scope_cc)
            with tik_inst.new_stmt_scope():
                # apply for tensor in L0A for q (64, 64) zZ
                mat_l0a_ldw = tik_inst.Tensor('float16', (block_size // 16, block_size // 16, 16, 16),
                                              name='mat_l0a_ldw', scope=tik.scope_ca)
                # apply for tensor in L0B for global k (128, 256) nZ
                mat_l0b_lk = tik_inst.Tensor('float16', (block_size // 16, size_per_head // 16, 16, 16),
                                             name='mat_l0b_lk', scope=tik.scope_cb)
                # move q from l1 to L0A for CUBE mmad
                # the shape of q in l1 is zZ
                # the shape of q in L0A is zZ
                # the stride between each (16, 16) is 0
                # repeat 16 times
                tik_inst.load2dv1(mat_l0a_ldw[0, 0, 0, 0], mat_l1_ldw[0, 0, 0, 0], 0,
                                  block_size * block_size // (16 * 16), 1, 0, False)
                # move local k from l1 to L0B for CUBE mmad
                # the shape of local k in l1 is zZ
                # the shape of local k in L0B is nZ
                # the stride between each (16, 16) is 0
                # repeat 32 times
                tik_inst.load2dv1(mat_l0b_lk[0, 0, 0, 0], mat_l1_lk[0, 0, 0, 0], 0,
                                  block_size * size_per_head // (16 * 16), 1, 0, True)
                # matmul q and local dw
                # the shape of global scores in L0C is zN
                tik_inst.mmad(mat_l0c_dq, mat_l0a_ldw, mat_l0b_lk,
                              block_size, block_size, size_per_head, 0)

            # global dq calculation
            # apply tensor in l1 for global dw (64, 256) zZ
            mat_l1_gdw = tik_inst.Tensor("float16",
                                         (block_size // 16,
                                          global_size // 16, 16, 16),
                                         name="mat_l1_gdw",
                                         scope=tik.scope_cbuf)
            # move global dw from ub to l1
            # the shape of global dw in gm is zN
            # the shape of global dw in l1 is zZ
            # the stride between each (16, 16) is 1024*bs-64
            # repeat 8 times
            # LOOP 4 times
            with tik_inst.for_range(0, block_size // 16) as lb:
                tik_inst.data_move(mat_l1_gdw[lb, 0, 0, 0],
                                   mat_ub_lg_d[block_size // 16, lb, 0, 0],
                                   0, global_size // 16, 16, block_size - 16, 0)
            # apply for tensor in ub for dq (64, 128) zN
            mat_ub_dq = tik_inst.Tensor("float32",
                                        (size_per_head // 16,
                                         block_size // 16, 16, 16),
                                        name="mat_ub_dq",
                                        scope=tik.scope_ubuf)
            with tik_inst.new_stmt_scope():
                # apply for tensor in L0A for global dw (64, 256) zZ
                mat_l0a_gdw = tik_inst.Tensor('float16', (block_size // 16, global_size // 16, 16, 16),
                                              name='mat_l0a_gdw', scope=tik.scope_ca)
                # apply for tensor in L0B for global k (256, 128) nZ
                mat_l0b_gk = tik_inst.Tensor('float16', (global_size // 16, size_per_head // 16, 16, 16),
                                             name='mat_l0b_gk', scope=tik.scope_cb)
                # move dw global from l1 to L0A for CUBE mmad
                # the shape of q in l1 is zZ
                # the shape of q in L0A is zZ
                # the stride between each (16, 16) is 0
                # repeat 16 times
                tik_inst.load2dv1(mat_l0a_gdw[0, 0, 0, 0], mat_l1_gdw[0, 0, 0, 0], 0,
                                  block_size * global_size // (16 * 16), 1, 0, False)
                # move local k from l1 to L0B for CUBE mmad
                # the shape of local k in l1 is zZ
                # the shape of local k in L0B is nZ
                # the stride between each (16, 16) is 0
                # repeat 32 times
                tik_inst.load2dv1(mat_l0b_gk[0, 0, 0, 0], mat_l1_gk[0, 0, 0, 0], 0,
                                  global_size * size_per_head // (16 * 16), 1, 0, True)
                # matmul k and local dw
                # the shape of global scores in L0C is zN
                tik_inst.mmad(mat_l0c_dq, mat_l0a_gdw, mat_l0b_gk,
                              block_size, global_size, size_per_head, 1)
                # move dq from l0c to UB
                # the shape of dq in l9c is zN
                # the shape of dq in ub is zN
                # the stride between each (16, 64) is 0
                # repeat 8 times
                tik_inst.data_move(mat_ub_dq[0, 0, 0, 0], mat_l0c_dq[0, 0, 0, 0], 0, size_per_head // 16,
                                   block_size // 16, 0, 0)

            # local dk calculation
            # dk calculation q.T X dw
            # apply for tensor in ub for dw (320, 64) nZ
            mat_ub_lg_d_nz = tik_inst.Tensor("float16",
                                             (block_size // 16, (global_size +
                                                                 block_size) // 16, 16, 16),
                                             name='mat_ub_lg_d_nz',
                                             scope=tik.scope_ubuf)
            # transpose dw from zN to nZ
            with tik_inst.for_range(0, (global_size + block_size) // 16) as lb:
                with tik_inst.for_range(0, block_size // 16) as gb:
                    tik_inst.vtranspose(
                        mat_ub_lg_d_nz[gb, lb, 0, 0], mat_ub_lg_d[lb, gb, 0, 0])

            # apply tensor in l1 for local dw (64, 64) nZ
            mat_l1_ldw_nz = tik_inst.Tensor("float16",
                                            (block_size // 16,
                                             block_size // 16, 16, 16),
                                            name="mat_l1_ldw_nz",
                                            scope=tik.scope_cbuf)
            # move local dw from ub to l1
            # the shape of local dw in ub is nZ
            # the shape of local dw in l1 is nZ
            # the stride between each (16, 64) is 256
            # repeat 4 times
            tik_inst.data_move(mat_l1_ldw_nz[0, 0, 0, 0],
                               mat_ub_lg_d_nz[0, 0, 0, 0],
                               0, block_size // 16, block_size, global_size, 0)
            # apply for tensor in L1 for q (128, 64) nZ
            mat_l1_q_b = tik_inst.Tensor("float16",
                                         (size_per_head // 16,
                                          block_size // 16, 16, 16),
                                         name="mat_l1_q_b",
                                         scope=tik.scope_cbuf)
            # move local q from gm to l1
            # the shape of local q in gm is zN
            # the shape of local dw in l1 is zZ
            # the stride between each (16, 16) is 48
            # repeat 4 times
            # LOOP 8 times
            with tik_inst.for_range(0, size_per_head // 16) as lb:
                tik_inst.load2dv1(mat_l1_q_b[lb, 0, 0, 0],
                                  mat_q[head * size_per_head // 16 + lb,
                                        b * seq_len // 16 + (block * block_size) // 16, 0, 0],
                                  0, block_size // 16, 1, 0, False)
            # apply for tensor in L0C for local dk (128, 64) zN
            mat_l0c_dkl = tik_inst.Tensor("float32",
                                          (block_size // 16,
                                           size_per_head // 16, 16, 16),
                                          name="mat_l0c_dkl",
                                          scope=tik.scope_cc)
            # apply for tensor in ub for local dk (128, 64) zN
            mat_ub_ldk = tik_inst.Tensor("float32",
                                         (block_size // 16,
                                          size_per_head // 16, 16, 16),
                                         name="mat_ub_ldk",
                                         scope=tik.scope_ubuf)
            with tik_inst.new_stmt_scope():
                # apply for tensor in L0A for q (128, 64) zZ
                mat_l0a_q = tik_inst.Tensor('float16', (size_per_head // 16, block_size // 16, 16, 16),
                                            name='mat_l0a_q', scope=tik.scope_ca)
                # apply for tensor in L0B for local dw (64, 64) nZ
                mat_l0b_ldw = tik_inst.Tensor('float16', (block_size // 16, block_size // 16, 16, 16),
                                              name='mat_l0b_ldw', scope=tik.scope_cb)
                # move q from l1 to L0A for CUBE mmad
                # the shape of q in l1 is nZ
                # the shape of q in L0A is zZ
                # the stride between each (16, 16) is 0
                # repeat 4 times
                # LOOP 8 times
                tik_inst.load2dv1(mat_l0a_q[0, 0, 0, 0],
                                  mat_l1_q_b[0, 0, 0, 0],
                                  0, block_size * size_per_head // 256, 1, 0, True)
                # move local dw from l1 to L0B for CUBE mmad
                # the shape of local dw in l1 is nZ
                # the shape of local dw in L0B is nZ
                # the stride between each (16, 16) is 0
                # repeat 32 times
                tik_inst.load2dv1(mat_l0b_ldw[0, 0, 0, 0], mat_l1_ldw_nz[0, 0, 0, 0], 0,
                                  block_size * block_size // (16 * 16), 1, 0, False)
                # matmul q and local dw
                # the shape of local k in L0C is zN
                tik_inst.mmad(mat_l0c_dkl, mat_l0a_q, mat_l0b_ldw,
                              size_per_head, block_size, block_size, 0)
                # move local dk from l0c to UB
                # the shape of local dk in l0C is zN
                # the shape of local dk in UB is zN
                # the stride between each (16, 128) is 0
                # repeat 4 times
                tik_inst.data_move(mat_ub_ldk[0, 0, 0, 0], mat_l0c_dkl[0, 0, 0, 0], 0, block_size // 16,
                                   size_per_head // 16, 0, 0)

            # move global dw from UB to l1
            # apply for tensor in L1 for global dw (64, 256) nZ
            mat_l1_dwg_b = tik_inst.Tensor("float16",
                                           (block_size // 16,
                                            global_size // 16, 16, 16),
                                           name="mat_l1_dwg_b",
                                           scope=tik.scope_cbuf)
            # move global dw from UB to L1
            # the shape of global dw in gm is nZ
            # the shape of global dw in gm is nZ
            # the stride between each (16, 64) is 0
            # repeat 8 times
            tik_inst.data_move(mat_l1_dwg_b[0, 0, 0, 0],
                               mat_ub_lg_d_nz[0, block_size // 16, 0, 0],
                               0, block_size // 16, global_size, block_size, 0)

            with tik_inst.new_stmt_scope():
                # apply for tensor in L0A for q (128, 64) zZ
                mat_l0a_q = tik_inst.Tensor('float16', (size_per_head // 16, block_size // 16, 16, 16),
                                            name='mat_l0a_q', scope=tik.scope_ca)
                # apply for tensor in L0B for local dw (64, 64) nZ
                mat_l0b_gdw = tik_inst.Tensor('float16', (block_size // 16, global_size // 16, 16, 16),
                                              name='mat_l0b_ldw', scope=tik.scope_cb)
                # move q from l1 to L0A for CUBE mmad
                # the shape of q in l1 is nZ
                # the shape of q in L0A is zZ
                # the stride between each (16, 16) is 0
                # repeat 4 times
                # LOOP 8 times
                tik_inst.load2dv1(mat_l0a_q[0, 0, 0, 0],
                                  mat_l1_q_b[0, 0, 0, 0],
                                  0, block_size * size_per_head // 256, 1, 0, True)
                # move local dw from l1 to L0B for CUBE mmad
                # the shape of local dw in l1 is nZ
                # the shape of local dw in L0B is nZ
                # the stride between each (16, 16) is 0
                # repeat 32 times
                tik_inst.load2dv1(mat_l0b_gdw[0, 0, 0, 0], mat_l1_dwg_b[0, 0, 0, 0], 0,
                                  block_size * global_size // (16 * 16), 1, 0, False)
                # matmul q and local dw
                # the shape of local k in L0C is zN
                with tik_inst.if_scope(block == 0):
                    tik_inst.mmad(mat_l0c_dkg, mat_l0a_q, mat_l0b_gdw,
                                  size_per_head, block_size, global_size, 0)
                with tik_inst.else_scope():
                    tik_inst.mmad(mat_l0c_dkg, mat_l0a_q, mat_l0b_gdw,
                                  size_per_head, block_size, global_size, 1)

            # cast dq from 32 to 16
            # apply for tensor in ub for dq (64, 128) zN
            mat_ub_dq_16 = tik_inst.Tensor("float16",
                                           (size_per_head // 16,
                                            block_size // 16, 16, 16),
                                           name="mat_ub_dq_16",
                                           scope=tik.scope_ubuf)
            # apply for tensor in ub for local dk (128, 64) zN
            mat_ub_ldk_16 = tik_inst.Tensor("float16",
                                            (block_size // 16,
                                             size_per_head // 16, 16, 16),
                                            name="mat_ub_ldk_16",
                                            scope=tik.scope_ubuf)
            tik_inst.vec_conv(
                64, "", mat_ub_ldk_16[0, 0, 0, 0], mat_ub_ldk[0, 0, 0, 0], size_per_head * block_size // 64, 4, 8)
            tik_inst.vec_conv(
                64, "", mat_ub_dq_16[0, 0, 0, 0], mat_ub_dq[0, 0, 0, 0], size_per_head * block_size // 64, 4, 8)

            # move dq from UB to gm
            # the shape of dq in UB is zN
            # the shape of dq in gm is zN
            # the stride between each (16, 64) is 0
            # repeat 8 times
            tik_inst.data_move(mat_dq[head * size_per_head // 16,
                                      b * seq_len // 16 + (block * block_size) // 16, 0, 0],
                               mat_ub_dq_16[0, 0, 0,
                                            0], 0, size_per_head // 16, block_size, 0,
                               bs * seq_len - block_size)
            # move local dk from UB to gm
            # the shape of local dk in UB is zN
            # the shape of local dk in gm is zN
            # the stride between each (16, 64) is 0
            # repeat 8 times
            tik_inst.data_move(mat_dk[b * seq_len // 16 + (block * block_size) // 16,
                                      head * size_per_head // 16, 0, 0],
                               mat_ub_ldk_16[0, 0, 0,
                                             0], 0, block_size // 16, size_per_head, 0,
                               heads * size_per_head - size_per_head)
        with tik_inst.for_range(0, global_size // 16) as lb:
            # apply for tensor in ub for global dk (128, 16) zN
            mat_ub_gdk_32 = tik_inst.Tensor("float32",
                                            (1, size_per_head // 16, 16, 16),
                                            name="mat_ub_gdk",
                                            scope=tik.scope_ubuf)
            # apply for tensor in ub for global dk (128, 16) zN
            mat_ub_gdk = tik_inst.Tensor("float16",
                                         (1, size_per_head // 16, 16, 16),
                                         name="mat_ub_gdk",
                                         scope=tik.scope_ubuf)
            # apply for tensor in ub for global dk (128, 16) zN
            mat_ub_ldk2 = tik_inst.Tensor("float16",
                                          (1, size_per_head // 16, 16, 16),
                                          name="mat_ub_ldk2",
                                          scope=tik.scope_ubuf)
            # move global dk from l0c to UB
            # the shape of global dk in l0C is zN
            # the shape of global dk in UB is zN
            # the stride between each (16, 128) is 0
            # repeat 1 times
            tik_inst.data_move(mat_ub_gdk_32[0, 0, 0, 0], mat_l0c_dkg[lb, 0, 0, 0], 0, 1,
                               size_per_head // 16, 0, 0)
            tik_inst.vec_conv(
                64, "", mat_ub_gdk[0, 0, 0, 0], mat_ub_gdk_32[0, 0, 0, 0], size_per_head * 16 // 64, 4, 8)
            # move local dk from gm to UB
            # the shape of local dk in gm is zN
            # the shape of local dk in UB is zN
            # the stride between each (16, 128) is 0
            # repeat 1 times
            tik_inst.data_move(mat_ub_ldk2[0, 0, 0, 0], mat_dk[b * seq_len // 16 + 4 * lb + global_idx,
                                                               head * size_per_head // 16, 0, 0], 0, 1,
                               size_per_head, 0, 0)
            # add local dk and global dk
            mat_ub_dk = tik_inst.Tensor("float16",
                                        (1, size_per_head // 16, 16, 16),
                                        name="mat_ub_dk",
                                        scope=tik.scope_ubuf)
            tik_inst.vec_add(128, mat_ub_dk, mat_ub_ldk2, mat_ub_gdk,
                             size_per_head * 16 // 128, 8, 8, 8)
            # move dk from UB to gm
            # the shape of dk in UB is zN
            # the shape of dk in gm is zN
            # the stride between each (16, 128) is 0
            # repeat 1 times
            tik_inst.data_move(
                mat_dk[b * seq_len // 16 + 4 * lb + global_idx,
                       head * size_per_head // 16, 0, 0],
                mat_ub_dk[0, 0, 0, 0], 0, 1, size_per_head, 0, 0)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[mat_q, mat_k, mat_lc, mat_gc, mat_lc_grad, mat_gc_grad],
                      outputs=[mat_dq, mat_dk])
    return tik_inst
