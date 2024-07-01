# Copyright 2022 Huawei Technologies Co., Ltd
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
"""eval script"""
import os
import numpy as np
import time
import mindspore.context as context
from mindspore import Tensor, nn, load_checkpoint
from tests.st.mindscience.mindsponge.mindsponge.cell.amp import amp_convert
from tests.st.mindscience.mindsponge.mindsponge.common.config_load import load_config
from tests.mark_utils import arg_mark

from data import Feature
from model import MegaFold, compute_confidence


def fold_infer(mixed_precision, crop_size, is_ge_only=False):
    '''mega fold inference'''
    data_config = "./config/data.yaml"
    model_config = "./config/model.yaml"
    checkpoint_path = "/home/workspace/mindspore_ckpt/ckpt/MEGA_Fold_1.ckpt"
    data_cfg = load_config(data_config)
    model_cfg = load_config(model_config)
    data_cfg.eval.crop_size = crop_size
    model_cfg.seq_length = data_cfg.eval.crop_size
    slice_key = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[slice_key]
    model_cfg.slice = slice_val
    megafold = MegaFold(model_cfg, mixed_precision=mixed_precision)
    if is_ge_only:
        context.set_context(jit_level="O2")
    load_checkpoint(checkpoint_path, megafold)
    fp32_white_list = (nn.Softmax, nn.LayerNorm)
    amp_convert(megafold, fp32_white_list)
    time_list = []
    raw_feature = np.load("/home/workspace/mindspore_dataset/mindsponge_data/pkl/raw_feature.npy", allow_pickle=True)
    raw_feature = raw_feature.item()
    ori_res_length = raw_feature['msa'].shape[1]
    processed_feature = Feature(data_cfg, raw_feature)
    feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                               mixed_precision=mixed_precision)
    prev_pos = Tensor(prev_pos)
    prev_msa_first_row = Tensor(prev_msa_first_row)
    prev_pair = Tensor(prev_pair)
    for i in range(2):
        feat_i = [Tensor(x[i]) for x in feat]
        t_start = time.time()
        result = megafold(*feat_i, prev_pos, prev_msa_first_row, prev_pair)
        t_end = time.time()
        time_list.append(t_end - t_start)
        prev_pos, prev_msa_first_row, prev_pair, predicted_lddt_logits = result
    predicted_lddt_logits = predicted_lddt_logits.asnumpy()[:ori_res_length]
    confidence, _ = compute_confidence(predicted_lddt_logits, return_lddt=True)
    return confidence, time_list


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_910B_Ascend_fold():
    """
    Feature: 910B Megaflod
    Description: test train and eval
    Expectation: success
    """
    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "SATURATION_MODE"
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        memory_optimize_level="O1",
                        max_call_depth=6000)
    mixed_precision = 1
    crop_size = 512
    confidence, time_list = fold_infer(mixed_precision, crop_size)
    compile_time, exectue_time = time_list
    compile_time = compile_time - exectue_time
    os.environ.pop("MS_ASCEND_CHECK_OVERFLOW_MODE")
    assert confidence > 0.9
    assert compile_time < 500
    assert exectue_time < 100

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_910A_Ascend_fold():
    """
    Feature: 910A Megaflod
    Description: test train and eval
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        memory_optimize_level="O1",
                        max_call_depth=6000)
    context.set_context(jit_level="O2")
    mixed_precision = 1
    crop_size = 1024
    confidence, time_list = fold_infer(mixed_precision, crop_size, True)
    compile_time, exectue_time = time_list
    compile_time = compile_time - exectue_time
    assert confidence > 0.9
    assert compile_time < 500
    assert exectue_time < 100
