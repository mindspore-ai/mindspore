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
"""test ShuffleNetV1"""
import time
from mindspore import context, nn
from mindspore.train.model import Model
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.shufflenetv1 import ShuffleNetV1 as shufflenetv1
from src.dataset import create_dataset
from src.crossentropysmooth import CrossEntropySmooth
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id


set_seed(1)


@moxing_wrapper(pre_process=None)
def test():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                        device_id=get_device_id())

    # create dataset
    dataset = create_dataset(config.eval_dataset_path, do_train=False, device_num=1, rank=0)
    # step_size = dataset.get_dataset_size()

    # define net
    net = shufflenetv1(model_size=config.model_size, n_class=config.num_classes)

    # load checkpoint
    param_dict = load_checkpoint(config.ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss
    loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=config.label_smooth_factor,
                              num_classes=config.num_classes)

    # define model
    eval_metrics = {'Loss': nn.Loss(), 'Top_1_Acc': nn.Top1CategoricalAccuracy(),
                    'Top_5_Acc': nn.Top5CategoricalAccuracy()}
    model = Model(net, loss_fn=loss, metrics=eval_metrics)

    # eval model
    start_time = time.time()
    res = model.eval(dataset, dataset_sink_mode=True)
    log = "result:" + str(res) + ", ckpt:'" + config.ckpt_path + "', time: " + str(
        (time.time() - start_time) * 1000)
    print(log)
    filename = './eval_log.txt'
    with open(filename, 'a') as file_object:
        file_object.write(log + '\n')


if __name__ == '__main__':
    test()
