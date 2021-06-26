# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================

"""Local adapter"""

import os

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv('RANK_ID', '0')
    return int(global_rank_id)


def get_job_id():
    return 'Local Job'
