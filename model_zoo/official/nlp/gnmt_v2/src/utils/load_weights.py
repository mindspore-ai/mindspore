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
"""Weight loader."""

import numpy as np

from mindspore.train.serialization import load_checkpoint


def load_infer_weights(config):
    """
    Load weights from ckpt or npz.

    Args:
        config: Config.

    Returns:
        dict, weights.
    """
    model_path = config.existed_ckpt
    if model_path.endswith(".npz"):
        ms_ckpt = np.load(model_path)
        is_npz = True
    else:
        ms_ckpt = load_checkpoint(model_path)
        is_npz = False
    weights = {}
    for param_name in ms_ckpt:
        infer_name = param_name.replace("gnmt.gnmt.", "")
        if infer_name.startswith("embedding_lookup."):
            if is_npz:
                weights[infer_name] = ms_ckpt[param_name]
            else:
                weights[infer_name] = ms_ckpt[param_name].data.asnumpy()
            infer_name = "beam_decoder.decoder." + infer_name
            if is_npz:
                weights[infer_name] = ms_ckpt[param_name]
            else:
                weights[infer_name] = ms_ckpt[param_name].data.asnumpy()
            continue
        elif not infer_name.startswith("gnmt_encoder"):
            if infer_name.startswith("gnmt_decoder."):
                infer_name = infer_name.replace("gnmt_decoder.", "decoder.")
            infer_name = "beam_decoder.decoder." + infer_name

        if is_npz:
            weights[infer_name] = ms_ckpt[param_name]
        else:
            weights[infer_name] = ms_ckpt[param_name].data.asnumpy()
    return weights
