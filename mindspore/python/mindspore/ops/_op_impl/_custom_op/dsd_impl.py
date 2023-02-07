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
""" dense sparse to densne matmul"""
from __future__ import absolute_import
from te import tik
from tbe.tvm.topi.cce import util
from mindspore.ops.op_info_register import DataType, TBERegOp, op_info_register

dsd_matmul_info = TBERegOp('DSDMatmul') \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("dsdmatmul.so") \
    .compute_cost(10) \
    .kernel_name("dsd_matmul") \
    .partial_flag(True) \
    .input(0, "input_w1", False, "required", "all") \
    .input(1, "input_w2", False, "required", "all") \
    .input(2, "input_v", False, "required", "all") \
    .output(0, "output_y", False, "required", "all") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .get_op_info()


@op_info_register(dsd_matmul_info)
def dsd_matmul(input_w1, input_w2, input_v, output_y={}, kernel_name='dsd_matmul'):
    """ dense sparse to densne matmul"""
    if util.get_product_version() == util.VERSION_MINI:
        tik_inst = tik.Tik(tik.Dprofile("v100", "mini"))
    else:
        tik_inst = tik.Tik(tik.Dprofile("v100", "cloud"))

    # shape is: (batch_size, head, block_num, block_size//16, 16, head_size//16, 16)
    input_w1_shape = input_w1.get('shape')
    # shape is: (batch_size, head, block_num, head_size//16, 16, global_size//16, 16)
    input_w2_shape = input_w2.get('shape')
    input_v_shape = input_v.get('shape')

    batch_size = input_w1_shape[0]
    head = input_w1_shape[1]
    block_num = input_w1_shape[2]
    block_size = input_w1_shape[4] * 16
    head_size = input_w1_shape[3] * 16
    global_size = input_w2_shape[3] * 16
    v_embedding = input_v_shape[1] * 16 // head
    seq_len = input_v_shape[0] * 16 // batch_size

    block_bite_size = 32
    cpt_time = seq_len // 512

    w1_gm = tik_inst.Tensor('float16', (batch_size, head, block_num, head_size //
                                        16, block_size // 16, 16, 16), name='w1_gm', scope=tik.scope_gm)
    w2_gm = tik_inst.Tensor('float16', (batch_size, head, block_num, global_size //
                                        16, head_size // 16, 16, 16), name='w2_gm', scope=tik.scope_gm)
    #
    v_gm = tik_inst.Tensor('float16', (batch_size * seq_len // 16,
                                       head * v_embedding // 16, 16, 16), name='v_gm', scope=tik.scope_gm)
    # zN
    output_gm = tik_inst.Tensor('float16', (batch_size, head, v_embedding // 16, seq_len // 16, 16, 16),
                                name='output_gm',
                                scope=tik.scope_gm)

    channel_num = batch_size * head
    with tik_inst.for_range(0, channel_num, block_num=channel_num) as channel_idx:
        head_idx = channel_idx // batch_size
        bs_idx = channel_idx % batch_size
        output_l0c = tik_inst.Tensor("float32", (v_embedding // 16, block_size // 16, 16, 16), name='output_l0c',
                                     scope=tik.scope_cc)
        output_ub_32 = tik_inst.Tensor('float32', (v_embedding // 16, block_size // 16, 16, 16), name='output_ub_32',
                                       scope=tik.scope_ubuf)
        output_ub = tik_inst.Tensor('float16', (v_embedding // 16, block_size // 16, 16, 16), name='output_ub',
                                    scope=tik.scope_ubuf)
        # zZ
        w1_l1 = tik_inst.Tensor(
            'float16', (block_size // 16, head_size // 16, 16, 16), name='w1_l1', scope=tik.scope_cbuf)
        # nZ
        v_local_l1 = tik_inst.Tensor(
            'float16', (head_size // 16, v_embedding // 16, 16, 16), name='v_local_l1', scope=tik.scope_cbuf)
        # zZ
        w2_l1 = tik_inst.Tensor('float16', (head_size // 16, global_size // (16 * cpt_time), 16, 16),
                                name='w2_l1', scope=tik.scope_cbuf)
        # nZ
        # use same v_global
        v_global_l1 = tik_inst.Tensor('float16', (global_size // 16, v_embedding // 16, 16, 16),
                                      name='v_global_l1', scope=tik.scope_cbuf)
        # global v
        global_idx = 3 - head_idx % 4
        tik_inst.data_move(v_global_l1[0, 0, 0, 0], v_gm[bs_idx * seq_len // 16 + global_idx,
                                                         head_idx * v_embedding // 16, 0, 0], 0, seq_len // (4 * 16),
                           16 * v_embedding * 2 // block_bite_size,
                           (4 * head * v_embedding * 16 - 16 * v_embedding) * 2 // block_bite_size, 0)
        # every block size is 64, the output of the local and global is (1024,128) Zn
        with tik_inst.for_range(0, block_num, thread_num=2) as w_idx:
            # global
            with tik_inst.new_stmt_scope():
                w2_l0a = tik_inst.Tensor('float16', (head_size // 16, global_size // (cpt_time * 16), 16, 16),
                                         name='w2_l0a', scope=tik.scope_ca)
                v_global_l0b = tik_inst.Tensor('float16', (global_size // (cpt_time * 16), v_embedding // 16, 16, 16),
                                               name='v_global_l0b', scope=tik.scope_cb)
                with tik_inst.for_range(0, cpt_time) as cpt_idx:
                    with tik_inst.for_range(0, head_size // 16) as brick_i:
                        tik_inst.data_move(w2_l1[brick_i, 0, 0, 0],
                                           w2_gm[bs_idx, head_idx, w_idx, cpt_idx *
                                                 global_size // (16 * cpt_time), brick_i, 0, 0], 0,
                                           global_size // (16 * cpt_time), 16 * 16 * 2 // block_bite_size,
                                           (block_size // 16 - 1) * 16 * 16 * 2 // block_bite_size, 0)
                    tik_inst.load2dv1(
                        w2_l0a[0, 0, 0, 0], w2_l1[0, 0, 0, 0], 0, block_size * global_size // (cpt_time * 16 * 16), 1,
                        0)

                    tik_inst.load2dv1(v_global_l0b[0, 0, 0, 0], v_global_l1[cpt_idx * global_size // (
                        16 * cpt_time), 0, 0, 0], 0, global_size * v_embedding // (16 * 16 * cpt_time), 1, 0)

                    with tik_inst.if_scope(cpt_idx == 0):
                        tik_inst.mmad(output_l0c, w2_l0a, v_global_l0b,
                                      block_size, global_size // cpt_time, v_embedding, 0)
                    with tik_inst.else_scope():
                        tik_inst.mmad(output_l0c, w2_l0a, v_global_l0b,
                                      block_size, global_size // cpt_time, v_embedding, 1)
            # local
            with tik_inst.new_stmt_scope():
                w1_l0a = tik_inst.Tensor('float16', (block_size // 16, head_size // 16, 16, 16),
                                         name='w1_l0a', scope=tik.scope_ca)
                v_local_l0b = tik_inst.Tensor('float16', (head_size // 16, v_embedding // 16, 16, 16),
                                              name='v_local_l0b', scope=tik.scope_cb)
                tik_inst.data_move(v_local_l1[0, 0, 0, 0],
                                   v_gm[bs_idx * seq_len // 16 + w_idx * 4, head_idx *
                                        v_embedding // 16, 0, 0], 0, block_size // 16,
                                   16 * v_embedding * 2 // block_bite_size,
                                   16 * (head - 1) * v_embedding * 2 // block_bite_size, 0)
                tik_inst.load2dv1(v_local_l0b[0, 0, 0, 0], v_local_l1[0, 0, 0, 0], 0,
                                  head_size * v_embedding // (16 * 16), 1, 0)
                # w
                with tik_inst.for_range(0, block_size // 16) as brick_i:
                    tik_inst.data_move(w1_l1[brick_i, 0, 0, 0], w1_gm[bs_idx, head_idx, w_idx, 0, brick_i, 0, 0], 0,
                                       head_size // 16, (16 * 16 * 2) // block_bite_size,
                                       (block_size // 16 - 1) * 16 * 16 * 2 // block_bite_size, 0)
                tik_inst.load2dv1(w1_l0a[0, 0, 0, 0], w1_l1[0, 0, 0, 0], 0, block_size * head_size // (16 * 16), 1, 0)
                tik_inst.mmad(output_l0c, w1_l0a, v_local_l0b,
                              block_size, head_size, v_embedding, 1)
                tik_inst.data_move(output_ub_32[0, 0, 0, 0], output_l0c[0, 0, 0, 0], 0,
                                   1, block_size * v_embedding * 4 // 1024, 0, 0)
                tik_inst.vconv(64, '', output_ub[0, 0, 0, 0], output_ub_32[0, 0, 0, 0],
                               v_embedding * block_size // 64, 1, 1, 4, 8)
                tik_inst.data_move(output_gm[bs_idx, head_idx, 0, w_idx * (block_size // 16), 0, 0],
                                   output_ub[0, 0, 0, 0],
                                   0, v_embedding // 16, 16 * block_size * 2 // block_bite_size, 0,
                                   (seq_len - block_size) * 16 * 2 // block_bite_size)
    tik_inst.BuildCCE(kernel_name=kernel_name, inputs=[w1_gm, w2_gm, v_gm],
                      outputs=[output_gm])
    return tik_inst
