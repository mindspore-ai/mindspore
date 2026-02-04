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
# ==============================================================================
"""
Test dataset with send and recv
"""

import os

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_dataset_send_recv():
    """
    Feature: Dataset with send & recv
    Description: send to recv
    Expectation: Success
    """
    # dataset send & recv
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True dataset_send_recv.py")
    assert return_code == 0


if __name__ == '__main__':
    test_dataset_send_recv()
