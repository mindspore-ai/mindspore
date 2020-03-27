# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================
import mindspore.dataset as ds


def test_basic():
    ds.config.load('../data/dataset/declient.cfg')

    # assert ds.config.get_rows_per_buffer() == 32
    assert ds.config.get_num_parallel_workers() == 4
    # assert ds.config.get_worker_connector_size() == 16
    assert ds.config.get_prefetch_size() == 16
    assert ds.config.get_seed() == 5489

    # ds.config.set_rows_per_buffer(1)
    ds.config.set_num_parallel_workers(2)
    # ds.config.set_worker_connector_size(3)
    ds.config.set_prefetch_size(4)
    ds.config.set_seed(5)

    # assert ds.config.get_rows_per_buffer() == 1
    assert ds.config.get_num_parallel_workers() == 2
    # assert ds.config.get_worker_connector_size() == 3
    assert ds.config.get_prefetch_size() == 4
    assert ds.config.get_seed() == 5


if __name__ == '__main__':
    test_basic()
