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
"""export checkpoint file into air, mindir and onnx models"""
import numpy as np
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper
from src.gat import GAT
from mindspore import Tensor, context, load_checkpoint, export


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def export_gat():
    """ export_gat """
    if config.dataset == "citeseer":
        feature_size = [1, 3312, 3703]
        biases_size = [1, 3312, 3312]
        num_classes = 6
    else:
        feature_size = [1, 2708, 1433]
        biases_size = [1, 2708, 2708]
        num_classes = 7

    hid_units = config.hid_units
    n_heads = config.n_heads

    feature = np.random.uniform(0.0, 1.0, size=feature_size).astype(np.float32)
    biases = np.random.uniform(0.0, 1.0, size=biases_size).astype(np.float64)

    feature_size = feature.shape[2]
    num_nodes = feature.shape[1]

    gat_net = GAT(feature_size,
                  num_classes,
                  num_nodes,
                  hid_units,
                  n_heads,
                  attn_drop=0.0,
                  ftr_drop=0.0)

    gat_net.set_train(False)
    load_checkpoint(config.ckpt_file, net=gat_net)
    gat_net.add_flags_recursive(fp16=True)

    export(gat_net, Tensor(feature), Tensor(biases), file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_gat()
