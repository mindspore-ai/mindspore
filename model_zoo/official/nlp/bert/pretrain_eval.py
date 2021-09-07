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

"""
Bert evaluation script.
"""

import os
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.utils import BertMetric
from src.model_utils.config import config as cfg, bert_net_cfg
from src.bert_for_pre_training import BertPretrainEval
from src.dataset import create_eval_dataset


def bert_predict():
    '''
    Predict function
    '''
    devid = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
    dataset = create_eval_dataset(cfg.batch_size, 1, data_dir=cfg.eval_data_dir)
    net_for_pretraining = BertPretrainEval(bert_net_cfg)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    return model, dataset, net_for_pretraining


def MLM_eval():
    '''
    Evaluate function
    '''
    _, dataset, net_for_pretraining = bert_predict()
    net = Model(net_for_pretraining, eval_network=net_for_pretraining, eval_indexes=[0, 1],
                metrics={'name': BertMetric(cfg.batch_size)})
    res = net.eval(dataset, dataset_sink_mode=False)
    print("==============================================================")
    for _, v in res.items():
        print("Accuracy is: ", v)
    print("==============================================================")


if __name__ == "__main__":
    DEVICE_ID = 0
    os.environ['DEVICE_ID'] = str(DEVICE_ID)
    MLM_eval()
