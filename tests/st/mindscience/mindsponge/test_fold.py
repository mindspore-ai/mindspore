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
import pytest
import mindspore.context as context
from mindspore import Tensor, nn, load_checkpoint
from mindspore.common import mutable
from tests.st.mindscience.mindsponge.mindsponge.cell.amp import amp_convert
from tests.st.mindscience.mindsponge.mindsponge.common.config_load import load_config

from data import Feature
from model import MegaFold, compute_confidence

from module.lr import cos_decay_lr
from module.fold_wrapcell import TrainOneStepCell, WithLossCell


context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    memory_optimize_level="O1",
                    max_call_depth=6000)

FP32_WHITE_LIST = (nn.Softmax, nn.LayerNorm)

def fold_infer(crop_size):
    '''mega fold inference'''
    mixed_precision = 1
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
    load_checkpoint(checkpoint_path, megafold)
    amp_convert(megafold, FP32_WHITE_LIST)
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


def train_step(data, mixed_precision):
    '''mega fold train'''
    config = load_config("./config/training.yaml")
    config.model.is_training = True
    network = MegaFold(config.model, mixed_precision)
    amp_convert(network, FP32_WHITE_LIST)
    train_config = config.train
    lr = cos_decay_lr(start_step=train_config.start_step, lr_init=0.0,
                      lr_min=train_config.lr_min, lr_max=train_config.lr_max,
                      decay_steps=train_config.total_steps,
                      warmup_steps=train_config.warmup_steps)
    opt = nn.Adam(params=network.trainable_params(), learning_rate=lr, eps=1e-6)
    net_with_criterion = WithLossCell(network, config)
    train_net = TrainOneStepCell(net_with_criterion, opt, sens=train_config.loss_scale,
                                 gradient_clip_value=train_config.gradient_clip)

    num_recycle = 3
    train_net.phase = 'forward'
    feature_list = ['target_feat', 'msa_feat', 'msa_mask', 'seq_mask', 'aatype',
                    'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions',
                    'template_mask', 'template_pseudo_beta_mask', 'template_pseudo_beta',
                    'extra_msa', 'extra_has_deletion', 'extra_deletion_value', 'extra_msa_mask',
                    'residx_atom37_to_atom14', 'atom37_atom_exists', 'residue_index',
                    'prev_pos', 'prev_msa_first_row', 'prev_pair']

    label_list = ["pseudo_beta", "pseudo_beta_mask", "all_atom_mask", "true_msa",
                  "bert_mask", "residx_atom14_to_atom37", "restype_atom14_bond_lower_bound",
                  "restype_atom14_bond_upper_bound", "atomtype_radius",
                  "backbone_affine_tensor", "backbone_affine_mask", "atom14_gt_positions",
                  "atom14_alt_gt_positions", "atom14_atom_is_ambiguous", "atom14_gt_exists",
                  "atom14_atom_exists", "atom14_alt_gt_exists", "all_atom_positions",
                  "rigidgroups_gt_frames", "rigidgroups_gt_exists", "rigidgroups_alt_gt_frames",
                  "torsion_angles_sin_cos", "use_clamped_fape", "filter_by_solution", "chi_mask"]

    recycle_feature_name = feature_list[:-3]
    prev_pos = Tensor(data['prev_pos'])
    prev_msa_first_row = Tensor(data['prev_msa_first_row'])
    prev_pair = Tensor(data['prev_pair'])

    for step in range(200):
        train_net.set_train(False)
        train_net.add_flags_recursive(train_backward=False)
        for recycle in range(num_recycle - 1):
            inputs = {}
            for key in recycle_feature_name:
                inputs[key] = Tensor(data[key][recycle])
            inputs['prev_pos'] = prev_pos
            inputs['prev_msa_first_row'] = prev_msa_first_row
            inputs['prev_pair'] = prev_pair
            inputs = mutable(inputs)
            feat = []
            for key in feature_list:
                feat.append(inputs[key])
            prev_pos, prev_msa_first_row, prev_pair, _ = network(*feat)
        inputs = {}
        for key in feature_list[:-3]:
            inputs[key] = Tensor(data[key][num_recycle - 1])
        inputs['prev_pos'] = prev_pos
        inputs['prev_msa_first_row'] = prev_msa_first_row
        inputs['prev_pair'] = prev_pair
        for key in label_list:
            inputs[key] = Tensor(data[key])
        train_net.add_flags_recursive(train_backward=True)
        train_net.phase = 'train'
        keys = feature_list + label_list
        feat = []
        for key in keys:
            feat.append(inputs.get(key))
        feat = mutable(feat)
        loss = train_net(*feat)
        step += 1
    return loss[0]

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_910B_Ascend_fold():
    """
    Feature: 910B Megaflod
    Description: test train and eval
    Expectation: success
    """
    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "SATURATION_MODE"
    crop_size = 1536
    confidence, time_list = fold_infer(crop_size)
    compile_time, exectue_time = time_list
    compile_time = compile_time - exectue_time
    os.environ.pop("MS_ASCEND_CHECK_OVERFLOW_MODE")
    assert confidence > 0.9
    assert compile_time < 500
    assert exectue_time < 100

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_910B_Ascend_fold_256():
    """
    Feature: 910B Megaflod
    Description: test train and eval
    Expectation: success
    """
    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "SATURATION_MODE"
    crop_size = 256
    confidence, time_list = fold_infer(crop_size)
    compile_time, exectue_time = time_list
    compile_time = compile_time - exectue_time
    os.environ.pop("MS_ASCEND_CHECK_OVERFLOW_MODE")
    assert confidence > 0.9
    assert compile_time < 350
    assert exectue_time < 2.5

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_910B_Ascend_fold_512():
    """
    Feature: 910B Megaflod
    Description: test train and eval
    Expectation: success
    """
    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "SATURATION_MODE"
    crop_size = 512
    confidence, time_list = fold_infer(crop_size)
    compile_time, exectue_time = time_list
    compile_time = compile_time - exectue_time
    os.environ.pop("MS_ASCEND_CHECK_OVERFLOW_MODE")
    assert confidence > 0.9
    assert compile_time < 350
    assert exectue_time < 8

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_910A_Ascend_fold():
    """
    Feature: 910A Megaflod
    Description: test train and eval
    Expectation: success
    """
    crop_size = 1024
    confidence, time_list = fold_infer(crop_size)
    compile_time, exectue_time = time_list
    compile_time = compile_time - exectue_time
    assert confidence > 0.9
    assert compile_time < 500
    assert exectue_time < 100

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_910B_Ascend_train_fold():
    """
    Feature: 910B Megaflod train
    Description: test train and eval
    Expectation: success
    """
    os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "SATURATION_MODE"
    train_data = np.load("/home/workspace/mindspore_dataset/mindsponge_data/train_data.npy",
                         allow_pickle=True)
    train_data = train_data.item()
    loss = train_step(train_data, 1)
    os.environ.pop("MS_ASCEND_CHECK_OVERFLOW_MODE")
    assert loss < 18
