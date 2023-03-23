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

matmul_dds_op_info = TBERegOp("MatmulDDS") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("matmul_dds.so") \
    .compute_cost(10) \
    .kernel_name("matmul_dds") \
    .partial_flag(True) \
    .attr("bs", "required", "int", "all") \
    .attr("heads", "required", "int", "all") \
    .input(0, "q", False, "required", "all") \
    .input(1, "k", False, "required", "all") \
    .input(2, "local_mask", False, "required", "all") \
    .input(3, "global_mask", False, "required", "all") \
    .output(0, "local_prob", False, "required", "all") \
    .output(1, "global_prob", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default,
                  DataType.F32_Default, DataType.F32_Default,
                  DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(matmul_dds_op_info)
def matmul_dds(q,
               k,
               local_mask,
               global_mask,
               local_prob,
               global_prob,
               bs,
               heads,
               kernel_name="matmul_dds"):
    """
    :param q: the dict of input q (bs*seq_len, embedding_size) zN
    :param k: the dict of input k (bs*seq_len, embedding_size) nZ
    :param bs:  batch size int
    :param heads:  number of heads int
    :param local_mask:  the dict of input mask local  (bs*16*64, 64) zN
    :param global_mask:   the dict of input mask global   (heads*1024, 256)  zN
    :param kernel_name:  dds_softmax
    :return: None
    """

    shape_q = q.get(
        'shape')  # shape_q (embedding_size, bs*seq_length) > (embedding_size//16, bs*seq_length//16, 16, 16) zN
    shape_local_mask = local_mask.get(
        'shape')  # shape_local_mask (16*64, bs*64) > (64, bs*4, 16, 16)    zN
    # sequence length only support 1024 for now
    seq_len = shape_q[1] * shape_q[2] // bs
    # size per head assume 128
    size_per_head = shape_q[0] * shape_q[-1] // heads
    block_size = shape_local_mask[0]  # block size only support 64 for now
    block_num = seq_len // block_size  # block number only support 16 for now
    global_size = seq_len // 4  # global size only support 256 for now

    tik_inst = tik.Tik(tik.Dprofile('v100', 'cloud'))

    mat_q = tik_inst.Tensor("float16", (size_per_head * heads // 16, bs * seq_len // 16, 16, 16),
                            name="mat_q",
                            scope=tik.scope_gm)  # zN
    mat_k = tik_inst.Tensor("float16", (size_per_head * heads // 16, bs * seq_len // 16, 16, 16),
                            name="mat_k",
                            scope=tik.scope_gm)  # nZ
    mat_lm = tik_inst.Tensor("float32", (block_num * block_size // 16, bs * block_size // 16, 16, 16),
                             name="mat_lm",
                             scope=tik.scope_gm)  # zN
    mat_gm = tik_inst.Tensor("float32", (bs * global_size // 16, seq_len // 16, 16, 16),
                             name="mat_gm",
                             scope=tik.scope_gm)  # zN
    mat_lc = tik_inst.Tensor("float16", (bs, heads, block_num, block_size // 16, block_size // 16, 16, 16),
                             name="mat_lc",
                             scope=tik.scope_gm)  # zN
    mat_gc = tik_inst.Tensor("float16", (bs, heads, block_num, global_size // 16, block_size // 16, 16, 16),
                             name="mat_gc",
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
        global_idx = tik_inst.Scalar(dtype="int32")
        global_idx.set_as(3 - (head - 4 * s))
        # apply tensor for global key which is (128, 256) in L1 nZ
        # for each head, global k is the same, put global k in L1 in order of reuse
        mat_l1_gk = tik_inst.Tensor("float16",
                                    (size_per_head // 16,
                                     global_size // 16, 16, 16),
                                    name="mat_l1_gk",
                                    scope=tik.scope_cbuf)
        with tik_inst.for_range(0, size_per_head // 16) as gb:
            # move global key from gm to L1 nZ
            # the shape of k is nZ, move (16, 256) in one loop, the stride between each (16, 16) is 3*(16,16)
            tik_inst.data_move(mat_l1_gk[gb, 0, 0, 0],
                               mat_k[head * size_per_head // 16 + gb,
                                     b * seq_len // 16 + global_idx, 0, 0],
                               0, block_num, 16, 48, 0)

        with tik_inst.for_range(0, block_num) as block:
            # calculate qk matmul block by block

            # apply tensor in l0c for local mask (64, 64) zN
            mat_l0c_l = tik_inst.Tensor("float32", (block_size // 16, block_size // 16, 16, 16),
                                        name='mat_l0c_l',
                                        scope=tik.scope_cc)
            # apply tensor in l0c for global mask (256, 64) zN
            mat_l0c_g = tik_inst.Tensor("float32", (global_size // 16, block_size // 16, 16, 16),
                                        name='mat_l0c_g',
                                        scope=tik.scope_cc)
            # apply tensor in l1 for local k (128, 64) nZ
            mat_l1_lk = tik_inst.Tensor("float16",
                                        (size_per_head // 16,
                                         block_size // 16, 16, 16),
                                        name="mat_l1_lk",
                                        scope=tik.scope_cbuf)
            # apply for tensor in L1 for fp 16 exp result (320, 64) zN
            mat_l1_lg_exp_16 = tik_inst.Tensor("float16", ((global_size + block_size) // 16,
                                                           block_size // 16, 16, 16),
                                               name='mat_l1_lg_exp_16',
                                               scope=tik.scope_cbuf)
            # convert exp out to fp 16
            # apply for tensor in UB for fp 16 exp result (64, 320) zN
            mat_ub_lg_exp_16 = tik_inst.Tensor("float16", ((global_size + block_size) // 16,
                                                           block_size // 16, 16, 16),
                                               name='mat_ub_lg_exp_16',
                                               scope=tik.scope_ubuf)
            # move local k from gm to l1 nZ
            # the shape of local k in gm is nZ
            # the shape of local k in l1 is nZ
            # the stride between each (16, 64) is 1024*bs-64
            # repeat 8 times
            tik_inst.data_move(mat_l1_lk,
                               mat_k[head * size_per_head // 16, b * seq_len // 16 + (
                                   block * block_size) // 16, 0, 0],
                               0, size_per_head // 16, block_size, bs * seq_len - block_size, 0)
            # apply tensor in l1 for q (64, 128) zN
            mat_l1_q = tik_inst.Tensor("float16",
                                       (block_size // 16,
                                        size_per_head // 16, 16, 16),
                                       name="mat_l1_q",
                                       scope=tik.scope_cbuf)
            # move q from gm to l1
            # the shape of local k in gm is zN
            # the shape of local k in l1 is zZ
            # the stride between each (16, 16) is 1024*bs-64
            # repeat 8 times
            # LOOP 4 times
            with tik_inst.for_range(0, block_size // 16) as lb:
                tik_inst.data_move(mat_l1_q[lb, 0, 0, 0],
                                   mat_q[head * size_per_head // 16, b * seq_len // 16 + (
                                       block * block_size) // 16 + lb, 0, 0],
                                   0, size_per_head // 16, 16, bs * seq_len - 16, 0)

            # global
            # apply a new scope
            with tik_inst.new_stmt_scope():
                # apply tensor in ub for global mask (256, 64) zN
                mat_ub_gm = tik_inst.Tensor("float32", (global_size // 16, block_size // 16, 16, 16),
                                            name='mat_ub_gm',
                                            scope=tik.scope_ubuf)
                # move global mask from gm to ub zN
                # the shape of global mask in gm is zN
                # the shape of global mask in UB is zN
                # the stride between each (64, 16) is 960
                # repeat 16 times
                tik_inst.data_move(mat_ub_gm,
                                   mat_gm[b * global_size // 16,
                                          block * block_size // 16, 0, 0],
                                   0, global_size // 16, block_size * 2, seq_len * 2 - block_size * 2, 0)
                # move global mask from ub to l0c for bias add
                # the shape of global mask in ub is zN
                # the shape of global mask in l0c is zN
                # the stride between each (16, 64) is 0
                # repeat 16 times
                tik_inst.data_move(mat_l0c_g[0, 0, 0, 0],
                                   mat_ub_gm[0, 0, 0, 0],
                                   0, global_size // 16, block_size // 16, 0, 0)
                with tik_inst.for_range(0, 4, thread_num=2) as gb:
                    # apply for tensor in L0A for q (64, 128) zZ
                    mat_l0a_g = tik_inst.Tensor('float16',
                                                (block_size // 16, size_per_head //
                                                 (16 * 4), 16, 16),
                                                name='mat_l0a_g', scope=tik.scope_ca)
                    # apply for tensor in L0B for global k (128, 256) nZ
                    mat_l0b_g = tik_inst.Tensor('float16',
                                                (size_per_head // (16 * 4),
                                                 global_size // 16, 16, 16),
                                                name='mat_l0b_g', scope=tik.scope_cb)
                    # move q from l1 to L0A for CUBE mmad
                    # the shape of q in l1 is zZ
                    # the shape of q in L0A is zZ
                    # the stride between each (16, 16) is 0
                    # repeat 32 times
                    with tik_inst.for_range(0, block_size // 16) as bl:
                        tik_inst.load2dv1(mat_l0a_g[bl, 0, 0, 0], mat_l1_q[bl, size_per_head * gb // 64, 0, 0], 0,
                                          16 * size_per_head // (4 * 16 * 16), 1, 0, False)
                    # move global k from l1 to L0B for CUBE mmad
                    # the shape of global k in l1 is nZ
                    # the shape of global k in L0B is nZ
                    # the stride between each (16, 16) is 0
                    # repeat 128 times
                    tik_inst.load2dv1(mat_l0b_g[0, 0, 0, 0], mat_l1_gk[size_per_head * gb // 64, 0, 0, 0], 0,
                                      global_size * size_per_head // (4 * 16 * 16), 1, 0, False)
                    # matmul q and global k
                    # the shape of global scores in L0C is zN
                    tik_inst.mmad(mat_l0c_g, mat_l0a_g, mat_l0b_g,
                                  block_size, size_per_head // 4, global_size, 1)

            # local
            # apply a new scope
            with tik_inst.new_stmt_scope():
                # apply tensor in ub for local mask (64, 64) zN
                mat_ub_lm = tik_inst.Tensor("float32", (block_size // 16, block_size // 16, 16, 16),
                                            name='mat_ub_lm',
                                            scope=tik.scope_ubuf)
                # move local mask from gm to ub zN
                # the shape of local mask in gm is zN
                # the shape of local mask in UB is zN
                # the stride between each (64, 16) is 0
                # repeat 4 times
                tik_inst.data_move(mat_ub_lm,
                                   mat_lm[block * block_size // 16,
                                          b * block_size // 16, 0, 0],
                                   0, block_size // 16, block_size * 2, (bs * block_size - block_size) * 2, 0)
                # move local mask from ub to l0c for bias add
                # the shape of local mask in ub is zN
                # the shape of local mask in l0c is zN
                # the stride between each (16, 64) is 0
                # repeat 4 times
                tik_inst.data_move(mat_l0c_l[0, 0, 0, 0],
                                   mat_ub_lm[0, 0, 0, 0],
                                   0, block_size // 16, block_size // 16, 0, 0)
                with tik_inst.for_range(0, 4, thread_num=2) as gb:
                    # apply for tensor in L0A for q (64, 128) zZ
                    mat_l0a_l = tik_inst.Tensor('float16', (block_size // 16, size_per_head // (16 * 4), 16, 16),
                                                name='mat_l0a_l', scope=tik.scope_ca)
                    # apply for tensor in L0B for local k (128, 64) nZ
                    mat_l0b_l = tik_inst.Tensor('float16', (size_per_head // (16 * 4), block_size // 16, 16, 16),
                                                name='mat_l0b_l', scope=tik.scope_cb)
                    # move q from l1 to L0A for CUBE mmad
                    # the shape of q in l1 is zZ
                    # the shape of q in L0A is zZ
                    # the stride between each (16, 16) is 0
                    # repeat 32 times
                    with tik_inst.for_range(0, block_size // 16) as bl:
                        tik_inst.load2dv1(mat_l0a_l[bl, 0, 0, 0], mat_l1_q[bl, size_per_head * gb // 64, 0, 0], 0,
                                          16 * size_per_head // (4 * 16 * 16), 1, 0, False)
                    # move local k from l1 to L0B for CUBE mmad
                    # the shape of local k in l1 is nZ
                    # the shape of local k in L0B is nZ
                    # the stride between each (16, 16) is 0
                    # repeat 32 times
                    tik_inst.load2dv1(mat_l0b_l[0, 0, 0, 0], mat_l1_lk[size_per_head * gb // 64, 0, 0, 0], 0,
                                      block_size * size_per_head // (16 * 16 * 4), 1, 0, False)
                    # matmul q and local k
                    # the shape of local scores in L0C is (64, 64) zN
                    tik_inst.mmad(mat_l0c_l, mat_l0a_l, mat_l0b_l,
                                  block_size, size_per_head // 4, block_size, 1)

            with tik_inst.new_stmt_scope():
                with tik_inst.for_range(0, block_size // 16, thread_num=2) as gb:
                    mat_ub_lg = tik_inst.Tensor("float32", (1, (block_size + global_size) // 16, 16, 16),
                                                name='mat_ub_lg',
                                                scope=tik.scope_ubuf)
                    tik_inst.data_move(mat_ub_lg[0, 0, 0, 0], mat_l0c_g[0, gb, 0, 0], 0,
                                       global_size // 16, 1, block_size // 16 - 1, 0)
                    tik_inst.data_move(mat_ub_lg[0, global_size // 16, 0, 0], mat_l0c_l[0, gb, 0, 0], 0,
                                       block_size // 16, 1, block_size // 16 - 1, 0)
                    mat_ub_lg_16 = tik_inst.Tensor("float16", (1, (block_size + global_size) // 16, 16, 16),
                                                   name='mat_ub_lg_16',
                                                   scope=tik.scope_ubuf)
                    tik_inst.vec_conv(64, "", mat_ub_lg_16[0, 0, 0, 0],
                                      mat_ub_lg[0, 0, 0, 0],
                                      (block_size + global_size) * 16 // 64, 4, 8)
                    with tik_inst.for_range(0, 16) as lb:
                        mat_ub_lg_lb = tik_inst.Tensor("float16", (block_size + global_size,),
                                                       name='mat_ub_lg_lb',
                                                       scope=tik.scope_ubuf)
                        mat_ub_lg_lb_subs = tik_inst.Tensor("float16", (block_size + global_size,),
                                                            name='mat_ub_lg_lb_subs',
                                                            scope=tik.scope_ubuf)

                        tik_inst.data_move(mat_ub_lg_lb[0], mat_ub_lg_16[0, 0, lb, 0], 0,
                                           (block_size + global_size) // 16, 1, 15, 0)
                        max_value = tik_inst.Scalar("float16",
                                                    name='max_value',
                                                    init_value=0)
                        with tik_inst.for_range(0, (block_size + global_size) // 64) as nb:
                            mat_ub_lg_max = tik_inst.Tensor("float16", (2,),
                                                            name='mat_ub_lg_max',
                                                            scope=tik.scope_ubuf)
                            tik_inst.vcmax(64, mat_ub_lg_max[0], mat_ub_lg_lb[64 * nb], 1,
                                           1, 1, 4)
                            mat_ub_lg_max_sub = tik_inst.Tensor("float16", (2,),
                                                                name='mat_ub_lg_max_sub',
                                                                scope=tik.scope_ubuf)
                            tik_inst.vmuls(
                                2, mat_ub_lg_max_sub[0], mat_ub_lg_max[0], -1.0, 1, 1, 1, 1, 1)
                            block_max_value = tik_inst.Scalar("float16",
                                                              name='block_max_value',
                                                              init_value=0)
                            block_max_value.set_as(mat_ub_lg_max_sub[0])
                            max_value_int8 = tik_inst.Scalar("int8",
                                                             name='max_value_int8',
                                                             init_value=0)
                            max_value_int = tik_inst.Tensor("int8", (1,),
                                                            name='max_value_int',
                                                            scope=tik.scope_ubuf)
                            max_value_fp16 = tik_inst.Tensor("float16", (1,),
                                                             name='max_value_fp16',
                                                             scope=tik.scope_ubuf)
                            max_value_fp16[0].set_as(max_value)
                            block_max_value_int = tik_inst.Tensor("int8", (1,),
                                                                  name='block_max_value_int',
                                                                  scope=tik.scope_ubuf)
                            block_max_value_int8 = tik_inst.Scalar("int8",
                                                                   name='block_max_value_int8',
                                                                   init_value=0)
                            tik_inst.vec_conv(
                                1, "", max_value_int, max_value_fp16[0], 1, 1, 1)
                            tik_inst.vec_conv(
                                1, "", block_max_value_int, mat_ub_lg_max_sub[0], 1, 1, 1)
                            max_value_int8.set_as(max_value_int[0])
                            block_max_value_int8.set_as(block_max_value_int[0])
                            with tik_inst.if_scope(block_max_value_int8 < max_value_int8):
                                max_value.set_as(block_max_value)
                            with tik_inst.else_scope():
                                block_max_value.set_as(max_value)
                        tik_inst.vadds(64, mat_ub_lg_lb_subs[0], mat_ub_lg_lb[0],
                                       max_value, (block_size + global_size) // 64, 1, 1, 4, 4)
                        mat_ub_lg_exp_lb = tik_inst.Tensor("float16", (block_size + global_size,),
                                                           name='mat_ub_lg_exp_lb',
                                                           scope=tik.scope_ubuf)
                        tik_inst.vexp(64, mat_ub_lg_exp_lb[0],
                                      mat_ub_lg_lb_subs[0], (block_size + global_size) // 64, 1, 1, 4, 4)
                        tik_inst.data_move(mat_ub_lg_exp_16[0, gb, lb, 0], mat_ub_lg_exp_lb[0], 0,
                                           (block_size + global_size) // 16, 1, 0, block_size - 1)

            # move exp fp16 from ub to L1 for CUBE mmad
            # the shape of exp fp16 in ub is zN
            # the shape of exp fp16 in L1 is zN
            # the stride between each (16, 16) is 0
            # repeat 4 times
            tik_inst.data_move(mat_l1_lg_exp_16[0, 0, 0, 0], mat_ub_lg_exp_16[0, 0, 0, 0],
                               0, (global_size + block_size) // 16, block_size, 0, 0)
            # apply for tensor in UB for local attention out (64, 64)  zN
            mat_ub_l_out = tik_inst.Tensor("float16", (block_size // 16, block_size // 16, 16, 16),
                                           name='mat_ub_l_out',
                                           scope=tik.scope_ubuf)
            # apply for tensor in UB for global attention out (64, 256)  zN
            mat_ub_g_out = tik_inst.Tensor("float16", (global_size // 16, block_size // 16, 16, 16),
                                           name='mat_ub_g_out',
                                           scope=tik.scope_ubuf)
            # apply tensor in l0c for exp sum (16, 64) zN
            mat_l0c_exp = tik_inst.Tensor("float32", (block_size // 16, 1, 16, 16),
                                          name='mat_l0c_exp',
                                          scope=tik.scope_cc)
            # apply tensor in ub for exp sum (16, 64) zN
            mat_ub_exp_sum = tik_inst.Tensor("float32", (block_size // 16, 1, 16, 16),
                                             name='mat_ub_exp_sum',
                                             scope=tik.scope_ubuf)

            with tik_inst.new_stmt_scope():
                with tik_inst.for_range(0, 4, thread_num=2) as gb:
                    # apply for tensor in L0A for q (64, 128) zZ
                    mat_l0a_ones = tik_inst.Tensor('float16', (1, (global_size + block_size) // 64, 16, 16),
                                                   name='mat_l0a_ones', scope=tik.scope_ca)
                    # apply for tensor in L0B for exp (350, 64) nZ
                    mat_l0b_exp = tik_inst.Tensor('float16',
                                                  ((global_size + block_size) //
                                                   64, block_size // 16, 16, 16),
                                                  name='mat_l0b_exp', scope=tik.scope_cb)
                    # move ones from l1 to L0A for CUBE mmad
                    # the shape of ones in l1 is zZ
                    # the shape of ones in L0A is zZ
                    # the stride between each (16, 16) is 0
                    # repeat 32 times
                    tik_inst.load2dv1(mat_l0a_ones[0, 0, 0, 0], mat_l1_ones[0, 0, 0, 0], 0,
                                      (global_size + block_size) * 16 // (4 * 16 * 16), 1, 0, False)
                    # move global k from l1 to L0B for CUBE mmad
                    # the shape of global k in l1 is nZ
                    # the shape of global k in L0B is nZ
                    # the stride between each (16, 16) is 0
                    # repeat 128 times
                    tik_inst.load2dv1(mat_l0b_exp[0, 0, 0, 0],
                                      mat_l1_lg_exp_16[(global_size + block_size) * gb // 64, 0, 0, 0], 0,
                                      (global_size + block_size) * block_size // (4 * 16 * 16), 1, 0, False)
                    with tik_inst.if_scope(gb == 0):
                        tik_inst.mmad(mat_l0c_exp, mat_l0a_ones, mat_l0b_exp, 16,
                                      (global_size + block_size) // 4, block_size, 0)
                    with tik_inst.else_scope():
                        tik_inst.mmad(mat_l0c_exp, mat_l0a_ones, mat_l0b_exp, 16,
                                      (global_size + block_size) // 4, block_size, 1)

                tik_inst.data_move(mat_ub_exp_sum[0, 0, 0, 0], mat_l0c_exp[0, 0, 0, 0], 0,
                                   block_size // 16, 1, 0, 0)
                # apply for tensor in UB for global prob sum (64,)
                mat_ub_lg_exp_sum = tik_inst.Tensor("float32", (block_size,),
                                                    name='mat_ub_lg_exp_sum',
                                                    scope=tik.scope_ubuf)
                tik_inst.data_move(mat_ub_lg_exp_sum[0], mat_ub_exp_sum[0, 0, 0, 0],
                                   0, block_size // 16, 1 * 2, 15 * 2, 0)
                # apply for tensor in UB for attention prob sum rec (64,)
                mat_ub_exp_sum_rec = tik_inst.Tensor("float32", (block_size,),
                                                     name='mat_ub_exp_sum_rec',
                                                     scope=tik.scope_ubuf)
                mat_ub_exp_sum_rec_16 = tik_inst.Tensor("float16", (block_size,),
                                                        name='mat_ub_exp_sum_rec_16',
                                                        scope=tik.scope_ubuf)
                worker_tensor = tik_inst.Tensor("float32", (block_size * 2,),
                                                name='worker_tensor',
                                                scope=tik.scope_ubuf)
                # calculate attention prob sum vec (64,)
                tik_inst.vec_rec_high_preci(
                    64, mat_ub_exp_sum_rec, mat_ub_lg_exp_sum, worker_tensor, 1, 8, 8)
                tik_inst.vec_conv(block_size, "", mat_ub_exp_sum_rec_16[0],
                                  mat_ub_exp_sum_rec[0],
                                  block_size // 64, 4, 8)
                with tik_inst.for_range(0, block_size) as bbs:
                    # apply for scalar in UB for prob sum rec
                    sum_exp = tik_inst.Scalar("float16",
                                              name='sum_exp',
                                              init_value=0)
                    # set value for scalar prob sum rec
                    sum_exp.set_as(mat_ub_exp_sum_rec_16[bbs])
                    tik_inst.vec_muls(16, mat_ub_l_out[0, bbs // 16, bbs % 16, 0],
                                      mat_ub_lg_exp_16[global_size //
                                                       16, bbs // 16, bbs % 16, 0],
                                      sum_exp, block_size // 16,
                                      block_size, block_size)
                    tik_inst.vec_muls(16, mat_ub_g_out[0, bbs // 16, bbs % 16, 0],
                                      mat_ub_lg_exp_16[0, bbs // 16, bbs %
                                                       16, 0], sum_exp, global_size // 16,
                                      block_size, block_size)
            # move local out from UB to gm
            # the shape of local out in UB is zN
            # the shape of local out in gm is zN
            # the stride between each (16, 64) is 0
            # repeat 4 times
            tik_inst.data_move(mat_lc[b, head, block, 0, 0, 0, 0], mat_ub_l_out[0, 0, 0, 0], 0,
                               block_size // 16, block_size, 0, 0)
            # move global out from UB to gm
            # the shape of global out in UB is zN
            # the shape of global out in gm is zN
            # the stride between each (16, 64) is 0
            # repeat 16 times
            tik_inst.data_move(mat_gc[b, head, block, 0, 0, 0, 0], mat_ub_g_out[0, 0, 0, 0], 0,
                               global_size // 16, block_size, 0, 0)

    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[mat_q, mat_k, mat_lm, mat_gm],
                      outputs=[mat_lc, mat_gc])
    return tik_inst
