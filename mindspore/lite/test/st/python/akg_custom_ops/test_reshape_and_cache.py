# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
Test ReshapeAndCache plugin custom ops.
"""
import os
import logging
from typing import List
import numpy as np
import mindspore_lite as mslite
from mindspore import nn, ops
from mindspore import Tensor, Parameter, context, export
from mindspore.ops.auto_generate.gen_ops_prim import ReshapeAndCache


class ReshapeAndCacheNet(nn.Cell):
    """
    ReshapeAndCacheNet.
    """

    def __init__(self, num_blocks, block_size, kv_head, head_dim):
        super().__init__()
        self.key_cache = Parameter(
            np.zeros((num_blocks, block_size, kv_head, head_dim), dtype=np.float16),
            name="key_cache",
        )
        self.value_cache = Parameter(
            np.zeros((num_blocks, block_size, kv_head, head_dim), dtype=np.float16),
            name="value_cache",
        )
        self.reshape_and_cache = ReshapeAndCache()
        self.depends = ops.Depend()

    def construct(self, key, value, slot_mapping):
        out_key = self.reshape_and_cache(
            key, value, self.key_cache, self.value_cache, slot_mapping
        )
        out_key_cache = self.depends(self.key_cache, out_key)
        out_value_cache = self.depends(self.value_cache, out_key)
        return out_key, out_key_cache, out_value_cache


def export_model() -> str:
    """
    Export model with fixed shape.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    num_tokens = 512
    num_head = 40
    head_dim = 128
    block_size = 16
    num_blocks = 128

    key = Tensor(np.ones((num_tokens, num_head, head_dim), dtype=np.float16))
    value = Tensor(np.ones((num_tokens, num_head, head_dim), dtype=np.float16))
    slot_mapping = Tensor(np.ones((num_tokens,), dtype=np.int32))

    file_name = "reshape_and_cache"
    net = ReshapeAndCacheNet(num_blocks, block_size, num_head, head_dim)
    export(net, key, value, slot_mapping, file_name=file_name, file_format="MINDIR")
    model_name = file_name + ".mindir"
    assert os.path.exists(model_name)
    return model_name


def ref_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    """
    Implement reshape_and_cache with numpy.
    """
    _, block_size, _, _ = key_cache.shape
    key_out = np.zeros_like(key)
    for i, slot_idx in enumerate(slot_mapping):
        if slot_idx == -1:  # skip special pad slot index -1.
            continue
        block_index = slot_idx // block_size
        block_offset = slot_idx % block_size
        key_cache[block_index, block_offset, :, :] = key[i, :, :]
        value_cache[block_index, block_offset, :, :] = value[i, :, :]
        key_out[i, :, :] = key[i, :, :]
    return (key_out, key_cache, value_cache)


def create_input(cache_shape: List[int], update_shape: List[int], with_pad_slot: bool):
    """
    Create numpy input data for ReshapeAndCache op.
    """
    key = np.random.rand(*update_shape).astype(np.float16)
    value = np.random.rand(*update_shape).astype(np.float16)

    key_cache = np.zeros(cache_shape).astype(np.float16)
    value_cache = np.zeros(cache_shape).astype(np.float16)

    num_blocks = cache_shape[0]
    block_size = cache_shape[1]
    total_num_slots = num_blocks * block_size

    num_tokens = update_shape[0]
    if with_pad_slot:
        # construct a slot mapping case like: [x, ..., z, -1, ..., -1]
        num_valid_token = num_tokens // 2
        num_pad_token = num_tokens - num_valid_token
        slot_mapping = np.random.choice(
            np.arange(0, total_num_slots), size=num_valid_token, replace=False
        ).astype(np.int32)
        pad_slots = np.array([-1 for _ in range(num_pad_token)], dtype=np.int32)
        slot_mapping = np.concatenate((slot_mapping, pad_slots), axis=0)
    else:
        slot_mapping = np.random.choice(
            np.arange(0, total_num_slots), size=num_tokens, replace=False
        ).astype(np.int32)

    return key, value, key_cache, value_cache, slot_mapping


def create_golden_data(with_pad_slot: bool):
    """
    Create golden data for ReshapeAndCache op.
    """
    num_tokens = 512
    num_blocks = 128
    block_size = 16
    num_head = 40
    head_dim = 128

    cache_shape = [num_blocks, block_size, num_head, head_dim]
    update_shape = [num_tokens, num_head, head_dim]
    key, value, key_cache, value_cache, slot_mapping = create_input(
        cache_shape, update_shape, with_pad_slot
    )
    logging.info("slot_mapping shape: %s, data:\n%s", slot_mapping.shape, slot_mapping)

    # generate golden output with numpy op implement.
    key_golden, key_cahce_gloden, value_cache_golden = ref_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping
    )

    inputs = [key, value, slot_mapping]
    ref_outputs = [key_golden, key_cahce_gloden, value_cache_golden]

    return inputs, ref_outputs


def do_mslite_infer(model_file, in_tensors):
    """
    Do model inference with mslite.
    """
    lite_context = mslite.Context()
    lite_context.target = ["ascend"]
    lite_context.ascend.device_id = 2
    lite_context.ascend.provider = "ge"
    lite_context.ascend.rank_id = 0
    model = mslite.Model()

    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "ascend_akg.ini")
    print(f"Use config file: {config_path}")
    model.build_from_file(
        model_file, mslite.ModelType.MINDIR, lite_context, config_path=config_path
    )

    out_tensors = model.predict(in_tensors)
    np_output = [tensor.get_data_to_numpy() for tensor in out_tensors]
    return np_output


def inference_model(mindir_model: str, with_pad_slot: bool):
    """
    Inference model
    """
    inputs, ref_outputs = create_golden_data(with_pad_slot)
    # 运行昇腾算子
    in_tensors = [mslite.Tensor(x) for x in inputs]
    ascend_outputs = do_mslite_infer(mindir_model, in_tensors)

    for i, ascend_output in enumerate(ascend_outputs):
        is_close = np.allclose(ref_outputs[i], ascend_output, rtol=1e-3, atol=1e-03)
        logging.info("ref_outputs %d:\n%s", i, ref_outputs[i])
        logging.info("ascend_outputs %d:\n%s", i, ascend_output)
        logging.info("ascend output %d is equal to ref output: %s", i, is_close)
        assert is_close


def test_reshape_and_cache_fixed_shape():
    """
    Test ReshapAndCache of fixed shape.
    """
    model_path = export_model()
    print(f"reshape_and_cache_dynamic_shape st : export success to path: {model_path}")
    logging.info(
        "reshape_and_cache_dynamic_shape st : export success to path: %s", model_path
    )

    inference_model(model_path, with_pad_slot=False)
    print("reshape_and_cache_dynamic_shape st : inference success.")


def test_reshape_and_cache_skip_pad_slot():
    """
    Test ReshapAndCache with skipping index: -1.
    """
    model_path = export_model()
    print(f"reshape_and_cache_skip_pad_slot st : export success to path: {model_path}")
    logging.info(
        "reshape_and_cache_skip_pad_slot st : export success to path: %s", model_path
    )

    inference_model(model_path, with_pad_slot=True)
    print("reshape_and_cache_skip_pad_slot st : inference success.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        filename="./test.log",
        filemode="w",
    )
    test_reshape_and_cache_fixed_shape()
    test_reshape_and_cache_skip_pad_slot()
