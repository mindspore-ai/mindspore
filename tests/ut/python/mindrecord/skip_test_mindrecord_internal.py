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
# ============================================================================
"""
test mindrecord internal func
"""
from multiprocessing import cpu_count

from mindspore.mindrecord import MAX_CONSUMER_COUNT


def test_c_layer_thread_num_with_python_layer():
    assert cpu_count() == MAX_CONSUMER_COUNT()


if __name__ == "__main__":
    test_c_layer_thread_num_with_python_layer()
