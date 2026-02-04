# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test callback."""
import time
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.callback import Callback
from tests.st.frontend.dataset.animal import create_animal_dataset
from tests.st.frontend.networks.network import resnet50


class CheckCallback(Callback):
    def __init__(self):
        super().__init__()
        self.begin_count = 0
        self.epoch_begin_count = 0
        self.epoch_end_count = 0
        self.step_begin_count = 0
        self.step_end_count = 0
        self.end_count = 0

    def begin(self, run_context):
        self.begin_count += 1

    def epoch_begin(self, run_context):
        self.epoch_begin_count += 1

    def epoch_end(self, run_context):
        self.epoch_end_count += 1

    def step_begin(self, run_context):
        self.step_begin_count += 1

    def step_end(self, run_context):
        self.step_end_count += 1

    def end(self, run_context):
        self.end_count += 1


class CallbackFactory:
    def __init__(self, epoch_size=2, batch_size=32, num_classes=12):
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def image_data_proc(self):
        dataset = create_animal_dataset(epoch_size=self.epoch_size, batch_size=self.batch_size)
        self.num_classes = dataset.num_classes()
        return dataset

    def me_train_dataset_with_callback(self, callback_func):
        dataset = self.image_data_proc()
        net = resnet50(self.num_classes)
        net.set_train()
        loss = SoftmaxCrossEntropyWithLogits(sparse=False)
        opt = Momentum(learning_rate=0.1, momentum=0.9,
                       params=filter(lambda x: x.requires_grad, net.get_parameters()))
        model = Model(net, loss, opt)
        start_time = int(round(time.time() * 1000))
        model.train(self.epoch_size, dataset, callbacks=callback_func, dataset_sink_mode=True)
        end_time = int(round(time.time() * 1000))
        print("----finish model train-----")
        return end_time - start_time


def test_callback_basic_6_insertion_point_check():
    """
    Feature: test callback.
    Description: test callback.
    Expectation: the result match with expected result.
    """
    fact = CallbackFactory(epoch_size=2, batch_size=32)
    checkcallback_cb = CheckCallback()
    fact.me_train_dataset_with_callback(callback_func=[checkcallback_cb])
    assert checkcallback_cb.begin_count == 1
    assert checkcallback_cb.epoch_begin_count == 2
    assert checkcallback_cb.epoch_end_count == 2
    assert checkcallback_cb.step_begin_count == 2
    assert checkcallback_cb.step_end_count == 2
    assert checkcallback_cb.end_count == 1
