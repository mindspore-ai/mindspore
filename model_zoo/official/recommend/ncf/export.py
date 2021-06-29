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
"""ncf export file"""
import os
import numpy as np

from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export

import src.constants as rconst
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from ncf import NCFModel, PredictWithSigmoid


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    '''run export.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    if config.device_target == "Ascend":
        context.set_context(device_id=config.device_id)

    topk = rconst.TOP_K
    num_eval_neg = rconst.NUM_EVAL_NEGATIVES

    if config.dataset == "ml-1m":
        num_eval_users = 6040
        num_eval_items = 3706
    elif config.dataset == "ml-20m":
        num_eval_users = 138493
        num_eval_items = 26744
    else:
        raise ValueError("not supported dataset")

    ncf_net = NCFModel(num_users=num_eval_users,
                       num_items=num_eval_items,
                       num_factors=config.num_factors,
                       model_layers=config.layers,
                       mf_regularization=0,
                       mlp_reg_layers=[0.0, 0.0, 0.0, 0.0],
                       mf_dim=16)

    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(ncf_net, param_dict)

    network = PredictWithSigmoid(ncf_net, topk, num_eval_neg)

    users = Tensor(np.zeros([config.eval_batch_size, 1]).astype(np.int32))
    items = Tensor(np.zeros([config.eval_batch_size, 1]).astype(np.int32))
    masks = Tensor(np.zeros([config.eval_batch_size, 1]).astype(np.float32))

    input_data = [users, items, masks]
    export(network, *input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == "__main__":
    run_export()
