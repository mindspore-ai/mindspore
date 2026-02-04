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
"""Model train utils."""

from typing import Iterator, Tuple

import numpy as np
from mindspore import log as logger
from mindspore.nn import Momentum, SoftmaxCrossEntropyWithLogits
from mindspore.train import Model


def create_train_model(network, amp_level="O0", metrics=None, loss_scale_manager=None,
                       loss="default", opt=None):
    """
    Create a MindSpore model for training.

    Args:
        network: MindSpore network to train.
        amp_level (str): AMP level, defaults to "O0".
        metrics: Training metrics.
        loss_scale_manager: Loss scale manager passed to `Model`.
        loss: Loss function or literal "default" to use softmax CE loss.
        opt: Optimizer instance. When None a Momentum optimizer is created.

    Returns:
        Model: Configured MindSpore `Model` instance.
    """
    logger.info("MindSporeTest::create a model with amp_level=%s", amp_level)
    loss_fn = loss
    if loss_fn == "default":
        loss_fn = SoftmaxCrossEntropyWithLogits(reduction="mean")

    optimizer = opt
    if optimizer is None:
        optimizer = Momentum(learning_rate=0.01, momentum=0.9, params=network.get_parameters())

    model = Model(network=network, loss_fn=loss_fn, optimizer=optimizer, amp_level=amp_level,
                  metrics=metrics, loss_scale_manager=loss_scale_manager)
    return model


class FakeDataInitMode:
    """Initialization modes for `GeneratorFakeData`."""

    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3


class GeneratorFakeData:
    """
    A fake dataset that yields randomly generated data, mirroring PyNative behaviour.

    Args:
        size (int): Total number of samples (default: 1024).
        batch_size (int): Batch size per iteration (default: 32).
        image_size (tuple): Shape of each sample (default: (3, 224, 224)).
        num_classes (int): Number of target classes (default: 10).
        random_offset (int): Offset added to the RNG seed (default: 0).
        use_parallel (bool): Kept for compatibility, no-op in the migrated version.
        fakedata_mode (FakeDataInitMode): Controls data initialisation strategy.
        dtype (numpy.dtype): Data type used for images and labels (default: np.float32).
    """

    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0, use_parallel=False,
                 fakedata_mode=FakeDataInitMode.OnesInit, dtype=np.float32):
        del use_parallel  # parallel context management is not required in migrated tests
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = dtype
        self.label_data_type = dtype
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode

        self.total_batch_size = self.rank_batch_size * self.rank_size
        assert self.size % self.total_batch_size == 0

        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def get_repeat_count(self):
        return 1

    def set_image_data_type(self, data_type):
        self.image_data_type = data_type

    def set_label_data_type(self, data_type):
        self.label_data_type = data_type

    def set_label_onehot(self, is_onehot=True):
        self.is_onehot = is_onehot

    def create_tuple_iterator(self, num_epochs=-1, do_copy=False):
        del num_epochs, do_copy
        return self

    def __next__(self):
        batch_index = self.batch_index
        self.batch_index += 1
        if batch_index * self.total_batch_size >= self.size:
            raise StopIteration

        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        if self.fakedata_mode == FakeDataInitMode.OnesInit:
            img = np.ones(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.ZerosInit:
            img = np.zeros(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.UniqueInit:
            total_size = 1
            for dim in self.total_batch_data_size:
                total_size *= dim
            img = np.reshape(np.arange(total_size) * 0.0001, self.total_batch_data_size)
        else:
            img = np.random.randn(*self.total_batch_data_size)
        target = np.random.randint(0, self.num_classes, size=(self.rank_size, self.rank_batch_size))
        np.random.set_state(rng_state)
        img = img[self.rank_id]
        target = target[self.rank_id]
        img_ret = img.astype(self.image_data_type)
        target_ret = target.astype(self.label_data_type)
        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_classes))
            target_onehot[np.arange(self.rank_batch_size), target] = 1
            target_ret = target_onehot.astype(self.label_data_type)
        return img_ret, target_ret

    def __len__(self):
        return self.size // self.total_batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0
