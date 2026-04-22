# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

import os
import json
import time
import numpy as np
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import Tensor
from mindspore import context
from mindspore.train import Model
from mindspore.train.callback import Callback
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from tests.mark_utils import arg_mark

class Net(nn.Cell):
    """simple network"""

    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.ones([1]).astype(np.int32)), name='w')

    def construct(self, *dataset):
        '''calculation '''
        return dataset[0] + self.w


class TimeMonitor(Callback):
    """Time Monitor."""

    def __init__(self, dataset_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.epoch_mseconds = list()
        self.per_step_mseconds = list()

    def epoch_begin(self, run_context):
        '''begin time '''
        self.epoch_time = time.time()
        print(run_context)

    def epoch_end(self, run_context):
        '''end time '''
        epoch_msecond = (time.time() - self.epoch_time) * 1000
        self.epoch_mseconds.append(epoch_msecond)
        per_step_msecond = epoch_msecond / self.dataset_size
        self.per_step_mseconds.append(per_step_msecond)
        print("Epoch time: {:5.3f}, per step time: {:5.3f}".format(epoch_msecond, per_step_msecond), flush=True)
        print(run_context)


def _set_context(train_mode, device, distribute):
    """
    setting context
    :param train_mode: train mode, context.GRAPH_MODE or context.PYNATIVE_MODE
    :param device: device where the code will be run, Ascend or GPU or CPU
    :param distribute: run distribute or not, True or False
    :return: None
    """
    # set context
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=train_mode, device_target=device)
    context.set_context(device_id=device_id)

class MyData():
    '''test autotune dataset '''

    @staticmethod
    def __getitem__(item):
        return np.random.randn(10, 10, 3).astype(np.float32), np.array([1]).astype(np.float32), np.array([10]).astype(
            np.float32)

    @staticmethod
    def __len__():
        return 100


def func_test(x, y):
    '''
    used by map
    '''
    return x + y


def create_dataset(num_parallel_workers=8, prefetch_size=8, autotune=False, json_name=None, interval=1,
                   is_offload=False, ir_patch=None, batch_size=None):
    '''create dataset '''
    ds.config.set_prefetch_size(prefetch_size)
    ds.config.set_enable_autotune(autotune, json_name)
    ds.config.set_autotune_interval(interval)

    ds.config.set_auto_offload(is_offload)

    if ir_patch is not None:
        context.set_context(save_graphs=True, save_graphs_path=ir_patch)

    column_names = ["image", "label", "title"]
    sampler = ds.RandomSampler(False, 80)
    dataset = ds.GeneratorDataset(MyData(), column_names, num_parallel_workers=num_parallel_workers,
                                  sampler=sampler)

    randomcrop_op = vision.RandomCrop(9)
    randominvert_op = vision.RandomInvert(0.5)
    verticalflip_op = vision.RandomVerticalFlip(0.5)

    dataset = dataset.map(operations=[randomcrop_op, randominvert_op], input_columns="image", output_columns="image",
                          num_parallel_workers=num_parallel_workers)

    dataset = dataset.map(operations=verticalflip_op, input_columns="image", output_columns="image",
                          num_parallel_workers=num_parallel_workers)

    dataset = dataset.map(operations=func_test, input_columns=["image", "label"], output_columns=["image"],
                          num_parallel_workers=num_parallel_workers)
    dataset = dataset.project(["image", "title"])

    horizontalflip_op = vision.RandomHorizontalFlip(0.5)
    dataset = dataset.map(operations=horizontalflip_op, input_columns=["image"], output_columns=["image"],
                          num_parallel_workers=num_parallel_workers)

    dataset = dataset.batch(batch_size, num_parallel_workers=num_parallel_workers)
    return dataset


def run_net(json_name):
    """ run generator dataset with net """
    _set_context(train_mode=1, device="CPU", distribute=False)

    dataset = create_dataset(8, 4, True, json_name, 1, False, None, 4)

    model = Model(Net())
    call_back = []
    dataset_size = dataset.get_dataset_size()
    call_back.append(TimeMonitor(dataset_size))
    model.train(epoch=10, train_dataset=dataset, callbacks=call_back, dataset_sink_mode=False, sink_size=25)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_func_autotune_generatordataset_cpu():
    """
    Feature: Test autotune with generator
    Description: Test autotune with genenrator
    Expectation: Success
    """
    json_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), "generator_json")
    if os.path.exists(json_name + "_0.json"):
        os.remove(json_name + "_0.json")

    run_net(json_name)

    assert os.path.exists(json_name + "_{}.json".format("0"))
    file = open(json_name + "_{}.json".format("0"), "r", encoding="utf-8")
    json_file = json.load(file)
    file.close()

    assert list(json_file.keys()) == ['remark', 'summary', 'tree'], "the key of json is wrong."
    assert len(json_file["summary"]) == 6, "len(map) is wrong."
    assert "Op(ID:" in json_file["summary"][0]
    assert "(num_parallel_workers:" in json_file["summary"][0]
    assert "prefetch_size:" in json_file["summary"][0]

    if os.path.exists(json_name + "_0.json"):
        os.remove(json_name + "_0.json")


if __name__ == "__main__":
    test_func_autotune_generatordataset_cpu()
