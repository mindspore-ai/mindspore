# Copyright 2020-21 Huawei Technologies Co., Ltd
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
"""Warpctc evaluation"""
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.loss import CTCLoss
from src.dataset import create_dataset
from src.crnn import crnn
from src.metric import CRNNAccuracy
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id


set_seed(1)


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)


@moxing_wrapper(pre_process=None)
def crnn_eval():
    if config.device_target == 'Ascend':
        device_id = get_device_id()
        context.set_context(device_id=device_id)

    config.batch_size = 1
    max_text_length = config.max_text_length
    # input_size = config.input_size
    # create dataset
    dataset = create_dataset(name=config.eval_dataset,
                             dataset_path=config.eval_dataset_path,
                             batch_size=config.batch_size,
                             is_training=False,
                             config=config)
    # step_size = dataset.get_dataset_size()
    loss = CTCLoss(max_sequence_length=config.num_step,
                   max_label_length=max_text_length,
                   batch_size=config.batch_size)
    net = crnn(config, full_precision=config.device_target != 'Ascend')
    # load checkpoint
    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)
    # define model
    model = Model(net, loss_fn=loss, metrics={'CRNNAccuracy': CRNNAccuracy(config)})
    # start evaluation
    res = model.eval(dataset, dataset_sink_mode=config.device_target == 'Ascend')
    print("result:", res, flush=True)

if __name__ == '__main__':
    crnn_eval()
