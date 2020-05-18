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
@File  : test_parse.py
@Date  : 2019-11-05 14:49
@Desc  :
"""
import os
from mindspore._extends.parse import dump_obj
from mindspore._extends.parse import load_obj


def test_load_dump():
    data = (1, 3, 2, 7, 9)
    file_name = dump_obj(data, "./")
    obj = load_obj("./" + file_name)
    os.remove(f'./{file_name}')
    assert data == obj
