# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
'''Remove after MindData merge to MindSpore '''
import numpy as np

from mindspore import Tensor


class MindData:
    """ Stub for MindData """

    def __init__(self, size=1, batch_size=None, repeat_count=1,
                 np_types=None, output_shapes=None, input_indexs=()):
        self._size = size
        self._batch_size = batch_size
        self._repeat_count = repeat_count
        self._np_types = np_types
        self._output_shapes = output_shapes
        self._input_indexs = input_indexs
        self._iter_num = 0
        self._global_step = 0

    def get_dataset_size(self):
        return self._size

    def get_repeat_count(self):
        return self._repeat_count

    def get_batch_size(self):
        return self._batch_size

    def output_types(self):
        return self._np_types

    def output_shapes(self):
        return self._output_shapes

    @property
    def input_indexs(self):
        return self._input_indexs

    def device_que(self, send_epoch_end=True, create_data_info_queue=False, queue_name=""):
        self.queue_name = '6ba41974-209e-11ea-88b0-a24efeb2c736'
        self.send_epoch_end = send_epoch_end
        return self

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        return self.__iter__()

    def send(self, num_epochs=-1):
        pass

    def stop_send(self):
        pass

    def release(self):
        pass

    def continue_send(self):
        pass

    def get_data_info(self):
        pass

    def get_mbuf_queue_size(self):
        pass

    def get_send_info(self):
        pass

    def __len__(self):
        return self._size

    def __iter__(self):
        return self

    def __next__(self):
        if self._size < self._iter_num:
            raise StopIteration
        self._iter_num += 1
        next_value = []
        for shape, typ in zip(self._output_shapes, self._np_types):
            next_value.append(Tensor(np.ndarray(shape, typ)))

        return tuple(next_value)

    def next(self):
        return self.__next__()

    def reset(self):
        self._iter_num = 0

    def get_init_step(self):
        return self._global_step
