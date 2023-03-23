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
"""dsd back impl"""
from __future__ import absolute_import
from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import DataType, TBERegOp, op_info_register

dsd_grad_info = TBERegOp('DSDGrad') \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("dsdbrop.so") \
    .compute_cost(10) \
    .kernel_name("dsdbpropimpl") \
    .partial_flag(True) \
    .input(0, "w1_gm", False, "required", "all") \
    .input(1, "w2_gm", False, "required", "all") \
    .input(2, "v_gm", False, "required", "all") \
    .input(3, "a_gm", False, "required", "all") \
    .input(4, "d_a_gm", False, "required", "all") \
    .output(0, "d_w1_gm", False, "required", "all") \
    .output(1, "d_w2_gm", False, "required", "all") \
    .output(2, "d_v_gm", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default,
                  DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(dsd_grad_info)
def dsdbpropimpl(w1_gm, w2_gm, v_gm, a_gm, d_a_gm, d_w1_gm={}, d_w2_gm={}, d_v_gm={}, kernel_name='dsdbpropimpl'):
    """dsd back impl"""
    if util.get_product_version() == util.VERSION_MINI:
        tik_inst = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_inst = tik.Tik(tik.Dprofile("v100", "cloud"))

    # shape is:(batch_size, head, block_num, block_size//16, 16, head_size//16, 16)
    input_w1_shape = w1_gm.get('shape')
    # shape is:(batch_size, head, block_num, head_size//16, 16, global_size//16, 16)
    input_w2_shape = w2_gm.get('shape')
    # shape is:(batch_size, seq_len//16, 16, head, v_embedding//16, 16)
    input_v_shape = v_gm.get('shape')

    batch_size = input_w1_shape[0]
    head = input_w1_shape[1]
    block_num = input_w1_shape[2]
    block_size = input_w1_shape[4] * 16
    head_size = input_w1_shape[3] * 16
    global_size = input_w2_shape[3] * 16
    v_embedding = input_v_shape[1] * 16 // head
    seq_len = input_v_shape[0] * 16 // batch_size

    block_bite_size = 32

    # 4, 16, 1024//64, 64//16, 64//16, 16*16
    w1_gm = tik_inst.Tensor('float16',
                            (batch_size, head, block_num, head_size //
                             16, block_size // 16, 16, 16),
                            name='w1_gm',
                            scope=tik.scope_gm)
    w2_gm = tik_inst.Tensor('float16',
                            (batch_size, head, block_num, global_size //
                             16, head_size // 16, 16, 16),
                            name='w2_gm',
                            scope=tik.scope_gm)

    v_gm = tik_inst.Tensor('float16',
                           (batch_size * seq_len // 16, head * v_embedding // 16, 16, 16),
                           name='v_gm',
                           scope=tik.scope_gm)

    # zN
    a_gm = tik_inst.Tensor('float16',
                           (batch_size, head, v_embedding //
                            16, seq_len // 16, 16, 16),
                           name='a_gm',
                           scope=tik.scope_gm)
    # zN
    d_a_gm = tik_inst.Tensor('float16',
                             (batch_size, head, v_embedding //
                              16, seq_len // 16, 16, 16),
                             name='d_a_gm',
                             scope=tik.scope_gm)

    # output
    # w-zN
    # 4, 16, 1024//64, 64//16, 64//16, 16*16
    d_w1_gm = tik_inst.Tensor('float16',
                              (batch_size, head, block_num, head_size //
                               16, block_size // 16, 16, 16),
                              name='d_w1_gm',
                              scope=tik.scope_gm)
    d_w2_gm = tik_inst.Tensor('float16',
                              (batch_size, head, block_num, global_size //
                               16, head_size // 16, 16, 16),
                              name='d_w2_gm',
                              scope=tik.scope_gm)

    # v-nZ
    d_v_gm = tik_inst.Tensor('float16',
                             (batch_size * seq_len // 16, head * v_embedding // 16, 16, 16),
                             name='d_v_gm',
                             scope=tik.scope_gm)

    channel_num = batch_size * head
    with tik_inst.for_range(0, channel_num, block_num=channel_num) as channel_idx:
        head_idx = channel_idx // batch_size
        bs_idx = channel_idx % batch_size
        global_idx = 3 - head_idx % 4
        # tensor size // (byte * l0b size * thread)
        cpt_time = 1 if global_size * v_embedding * \
                        4 // (1024 * 64) <= 1 else global_size * v_embedding * 4 // (1024 * 64)
        ub_time = 1 if global_size == 256 else 2

        d_a_l1 = tik_inst.Tensor('float16', (seq_len // 16, v_embedding // 16, 16, 16),
                                 name='d_a_l1', scope=tik.scope_cbuf)

        with tik_inst.for_range(0, v_embedding // 16) as brick_i:
            tik_inst.data_move(d_a_l1[0, brick_i, 0, 0], d_a_gm[bs_idx, head_idx, brick_i, 0, 0, 0], 0,
                               seq_len // 16, 16 * 16 * 2 // block_bite_size,
                               0, (v_embedding // 16 - 1) * 16 * 16 * 2 // block_bite_size)

        # dv
        with tik_inst.for_range(0, block_num, thread_num=2) as w_idx:
            d_v_l0c = tik_inst.Tensor('float32', (v_embedding // 16, head_size // 16, 16, 16),
                                      name='d_v_local_l0c', scope=tik.scope_cc)
            d_v_ub = tik_inst.Tensor('float16', (v_embedding // 16, head_size // 16, 16, 16),
                                     name='d_v_ub', scope=tik.scope_ubuf)
            d_v_ub_32 = tik_inst.Tensor('float32', (v_embedding // 16, head_size // 16, 16, 16),
                                        name='d_v_ub_32', scope=tik.scope_ubuf)

            d_v_global_32_l0c = tik_inst.Tensor('float32', (v_embedding // 16, 1, 16, 16),
                                                name='d_v_global_32_l0c', scope=tik.scope_cc)

            d_v_global_32_ub = tik_inst.Tensor('float32', (v_embedding // 16, 1, 16, 16),
                                               name='d_v_global_32_ub', scope=tik.scope_ubuf)

            # d_v_local
            with tik_inst.new_stmt_scope():
                w_local_l1 = tik_inst.Tensor('float16', (head_size // 16, block_size // 16, 16, 16),
                                             name='w_local_l1', scope=tik.scope_cbuf)
                w_local_l0a = tik_inst.Tensor('float16', (head_size // 16, block_size // 16, 16, 16),
                                              name='w_local_l0a', scope=tik.scope_ca)

                d_a_l0b = tik_inst.Tensor('float16', (block_size // 16, v_embedding // 16, 16, 16),
                                          name='d_a_l0b', scope=tik.scope_cb)

                tik_inst.data_move(w_local_l1[0, 0, 0, 0], w1_gm[bs_idx, head_idx, w_idx, 0, 0, 0, 0], 0,
                                   1, (block_size * head_size * 2) // block_bite_size,
                                   0, 0)

                tik_inst.load2dv1(d_a_l0b[0, 0, 0, 0], d_a_l1[w_idx * block_size // 16, 0, 0, 0], 0,
                                  (block_size * v_embedding) // (16 * 16), 1, 0, True)

                tik_inst.load2dv1(w_local_l0a[0, 0, 0, 0], w_local_l1[0, 0, 0, 0],
                                  0, (head_size * block_size) // (16 * 16),
                                  1, 0, True)

                tik_inst.mmad(d_v_l0c, w_local_l0a, d_a_l0b,
                              head_size, block_size, v_embedding, 0)

                tik_inst.data_move(d_v_ub_32[0, 0, 0, 0], d_v_l0c[0, 0, 0, 0], 0,
                                   1, (v_embedding * head_size) * 4 // 1024, 0, 0)

            # d_v_global
            with tik_inst.new_stmt_scope():
                w_global_l1 = tik_inst.Tensor('float16', (1, head_size // 16, 16, 16),
                                              name='w_global_l1', scope=tik.scope_cbuf)
                w_global_l0a = tik_inst.Tensor('float16', (1, head_size // 16, 16, 16),
                                               name='w_global_l0a', scope=tik.scope_ca)

                d_a_l0b = tik_inst.Tensor('float16', (head_size // 16, v_embedding // 16, 16, 16),
                                          name='d_a_l0b', scope=tik.scope_cb)
                with tik_inst.for_range(0, block_num, thread_num=2) as w_idx_1:
                    tik_inst.load2dv1(d_a_l0b[0, 0, 0, 0], d_a_l1[w_idx_1 * (block_size // 16), 0, 0, 0], 0,
                                      (head_size * v_embedding) // (16 * 16), 1, 0, True)

                    tik_inst.data_move(w_global_l1[0, 0, 0, 0], w2_gm[bs_idx, head_idx, w_idx_1, w_idx, 0, 0, 0], 0,
                                       head_size // 16, 16 * 16 * 2 // block_bite_size,
                                       0, 0)
                    tik_inst.load2dv1(w_global_l0a[0, 0, 0, 0], w_global_l1[0, 0, 0, 0], 0,
                                      16 * head_size // (16 * 16),
                                      1, 0, True)

                    # shape: d_v_l0c = (v_embedding // 16, head_size // 16, 16, 16)
                    with tik_inst.if_scope(w_idx_1 == 0):
                        tik_inst.mmad(d_v_global_32_l0c, w_global_l0a, d_a_l0b,
                                      16, head_size, v_embedding, 0)
                    with tik_inst.else_scope():
                        tik_inst.mmad(d_v_global_32_l0c, w_global_l0a, d_a_l0b,
                                      16, head_size, v_embedding, 1)

                tik_inst.data_move(d_v_global_32_ub[0, 0, 0, 0], d_v_global_32_l0c[0, 0, 0, 0], 0,
                                   1, v_embedding * 16 * 4 // 1024, 0, 0)

                with tik_inst.for_range(0, 4) as cpt_i:
                    tik_inst.vadd(64, d_v_ub_32[0, global_idx, cpt_i * 4, 0], d_v_ub_32[0, global_idx, cpt_i * 4, 0],
                                  d_v_global_32_ub[0, 0,
                                                   cpt_i * 4, 0], v_embedding // 16,
                                  1, 1, 1,
                                  head_size * 16 * 4 // block_bite_size, head_size * 16 * 4 // block_bite_size,
                                  16 * 16 * 4 // block_bite_size)

                tik_inst.vconv(64, '', d_v_ub[0, 0, 0, 0], d_v_ub_32[0, 0, 0, 0],
                               v_embedding * head_size // 64, 1, 1, 4, 8)

                with tik_inst.for_range(0, head_size // 16) as h_idx:
                    with tik_inst.for_range(0, v_embedding // 16) as v_idx:
                        tik_inst.vtranspose(
                            d_v_ub[v_idx, h_idx, 0, 0], d_v_ub[v_idx, h_idx, 0, 0])
                    tik_inst.data_move(d_v_gm[bs_idx * seq_len // 16 + w_idx * (block_size // 16) + h_idx,
                                              head_idx * v_embedding // 16, 0, 0],
                                       d_v_ub[0, h_idx, 0, 0], 0,
                                       v_embedding // 16, 16 * 16 * 2 // block_bite_size,
                                       (head_size // 16 - 1) * 16 * 16 * 2 // 32, 0)

        with tik_inst.new_stmt_scope():
            with tik_inst.for_range(0, block_num, thread_num=2) as w_idx:
                d_local_l0a = tik_inst.Tensor('float16', (block_size // 16, v_embedding // 16, 16, 16),
                                              name='d_local_l0a', scope=tik.scope_ca)

                v_local_l1 = tik_inst.Tensor('float16', (v_embedding // 16, head_size // 16, 16, 16),
                                             name='v_local_l1', scope=tik.scope_cbuf)
                v_local_l0b = tik_inst.Tensor('float16', (v_embedding // 16, head_size // 16, 16, 16),
                                              name='v_local_l0b', scope=tik.scope_cb)

                # d_w_local
                d_w_local_l0c = tik_inst.Tensor('float32', (head_size // 16, block_size // 16, 16, 16),
                                                name='d_w_local_l0c', scope=tik.scope_cc)

                d_w_local_ub_32 = tik_inst.Tensor('float32', (head_size // 16, block_size // 16, 16, 16),
                                                  name='d_w_local_ub', scope=tik.scope_ubuf)

                d_w_local_ub = tik_inst.Tensor('float16', (head_size // 16, block_size // 16, 16, 16),
                                               name='d_w_local_ub', scope=tik.scope_ubuf)

                tik_inst.load2dv1(d_local_l0a[0, 0, 0, 0], d_a_l1[w_idx * (block_size // 16), 0, 0, 0],
                                  0, (block_size * v_embedding) // (16 * 16), 1, 0, False)

                # shape is: v_gm = (batch_size, seq_len // 16, head, v_embedding // 16, 16, 16)
                # shape is: v_local_l1 = (v_embedding//16, head_size//16, 16, 16)
                with tik_inst.for_range(0, head_size // 16) as brick_i:
                    tik_inst.data_move(v_local_l1[0, brick_i, 0, 0],
                                       v_gm[bs_idx * seq_len // 16 + w_idx *
                                            (head_size // 16) + brick_i, head_idx * v_embedding // 16, 0, 0],
                                       0, v_embedding // 16, 16 * 16 * 2 // block_bite_size,
                                       0, (head_size // 16 - 1) * 16 * 16 * 2 // block_bite_size)

                tik_inst.load2dv1(v_local_l0b[0, 0, 0, 0], v_local_l1[0, 0, 0, 0],
                                  0, v_embedding * head_size // (16 * 16), 1, 0, True)

                # dw
                tik_inst.mmad(d_w_local_l0c, d_local_l0a, v_local_l0b,
                              block_size, v_embedding, head_size, 0)

                tik_inst.data_move(d_w_local_ub_32[0, 0, 0, 0], d_w_local_l0c[0, 0, 0, 0], 0,
                                   1, head_size * block_size * 4 // 1024,
                                   0, 0)

                tik_inst.vconv(64, '', d_w_local_ub[0, 0, 0, 0], d_w_local_ub_32[0, 0, 0, 0],
                               head_size * block_size // 64, 1, 1, 4, 8)

                tik_inst.data_move(d_w1_gm[bs_idx, head_idx, w_idx, 0, 0, 0, 0], d_w_local_ub[0, 0, 0, 0], 0,
                                   1, head_size * block_size * 2 // block_bite_size,
                                   0, 0)

        # calculate d_w_global
        with tik_inst.new_stmt_scope():
            # load2d permute
            v_global_l1 = tik_inst.Tensor('float16', (v_embedding // 16, global_size // 16, 16, 16),
                                          name='v_global_l1', scope=tik.scope_cbuf)
            with tik_inst.for_range(0, block_num) as w_idx:
                tik_inst.data_move(v_global_l1[0, w_idx, 0, 0],
                                   v_gm[bs_idx * seq_len // 16 + (
                                       w_idx * (
                                           block_size // 16) + global_idx), head_idx * v_embedding // 16, 0, 0],
                                   0, v_embedding // 16, 16 * 16 * 2 // block_bite_size,
                                   0, (global_size // 16 - 1) * 16 * 16 * 2 // block_bite_size)

            with tik_inst.for_range(0, block_num * ub_time, thread_num=2) as w_idx:
                d_global_l0a = tik_inst.Tensor('float16', (head_size // (16 * ub_time),
                                                           v_embedding // (16 * cpt_time), 16, 16),
                                               name='d_global_l0a', scope=tik.scope_ca)

                v_global_l0b = tik_inst.Tensor('float16', (v_embedding // (16 * cpt_time),
                                                           global_size // 16, 16, 16),
                                               name='v_global_l0b', scope=tik.scope_cb)

                # d_w_global，小z大n
                d_w_global_l0c = tik_inst.Tensor('float32', (global_size // 16, head_size // (16 * ub_time), 16, 16),
                                                 name='d_w_global_l0c', scope=tik.scope_cc)
                d_w_global_ub = tik_inst.Tensor('float16', (global_size // 16,
                                                            head_size // (16 * ub_time), 16, 16),
                                                name='d_w_global_ub', scope=tik.scope_ubuf)
                d_w_global_ub_32 = tik_inst.Tensor('float32', (global_size // 16,
                                                               head_size // (16 * ub_time), 16, 16),
                                                   name='d_w_global_ub_32', scope=tik.scope_ubuf)

                with tik_inst.for_range(0, cpt_time) as cpt_idx:
                    tik_inst.load2dv1(v_global_l0b[0, 0, 0, 0],
                                      v_global_l1[cpt_idx * v_embedding //
                                                  (16 * cpt_time), 0, 0, 0], 0,
                                      global_size * v_embedding // (16 * 16 * cpt_time), 1, 0, True)
                    with tik_inst.for_range(0, head_size // (16 * ub_time)) as brick_i:
                        tik_inst.load2dv1(d_global_l0a[brick_i, 0, 0, 0],
                                          d_a_l1[w_idx * (block_size // (16 * ub_time)) + brick_i,
                                                 cpt_idx * v_embedding // (16 * cpt_time), 0, 0],
                                          0, (16 * v_embedding) // (16 * 16 * cpt_time), 1, 0, False)

                    # shape is: (head_size, global_size) =  (head_size, v_embedding//cpttime) *
                    # shape is: (v_embedding//cpttime, global_size)
                    with tik_inst.if_scope(cpt_idx == 0):
                        tik_inst.mmad(d_w_global_l0c, d_global_l0a, v_global_l0b,
                                      head_size // ub_time, v_embedding // cpt_time, global_size, 0)
                    with tik_inst.else_scope():
                        tik_inst.mmad(d_w_global_l0c, d_global_l0a, v_global_l0b,
                                      head_size // ub_time, v_embedding // cpt_time, global_size, 1)

                tik_inst.data_move(d_w_global_ub_32[0, 0, 0, 0], d_w_global_l0c[0, 0, 0, 0], 0,
                                   1, head_size * global_size * 4 // (1024 * ub_time),
                                   0, 0)

                # shape is: global_size // 16, head_size // 16, 16, 16)
                rpt_time = global_size // (16 * 8)
                with tik_inst.for_range(0, rpt_time) as conv_i:
                    tik_inst.vconv(64, '',
                                   d_w_global_ub[conv_i * global_size //
                                                 (16 * rpt_time), 0, 0, 0],
                                   d_w_global_ub_32[conv_i * global_size //
                                                    (16 * rpt_time), 0, 0, 0],
                                   global_size * head_size // (64 * rpt_time * ub_time), 1, 1, 4, 8)

                with tik_inst.if_scope(ub_time == 1):
                    tik_inst.data_move(d_w2_gm[bs_idx, head_idx, w_idx, 0, 0, 0, 0], d_w_global_ub[0, 0, 0, 0], 0,
                                       1, head_size * global_size *
                                       2 // (block_bite_size),
                                       0, 0)
                with tik_inst.else_scope():
                    w_idx_i = w_idx // 2
                    h_idx = (w_idx % 2) * 2  # 0/2

                    with tik_inst.for_range(0, head_size // (16 * ub_time)) as m_idx:
                        tik_inst.data_move(d_w2_gm[bs_idx, head_idx, w_idx_i, 0, h_idx + m_idx, 0, 0],
                                           d_w_global_ub[0, m_idx, 0, 0], 0,
                                           global_size // 16, 16 * 16 * 2 // block_bite_size,
                                           (head_size // (16 * ub_time) - 1) *
                                           16 * 16 * 2 // block_bite_size,
                                           (head_size // 16 - 1) * 16 * 16 * 2 // block_bite_size)

    tik_inst.BuildCCE(kernel_name=kernel_name,
                      inputs=[w1_gm, w2_gm, v_gm, a_gm, d_a_gm],
                      outputs=[d_w1_gm, d_w2_gm, d_v_gm])
    return tik_inst
