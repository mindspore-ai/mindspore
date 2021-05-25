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

"""Device adapter for ModelArts"""

from .config import config
if config.enable_modelarts:
    from .moxing_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
else:
    from .local_adapter import get_device_id, get_device_num, get_rank_id, get_job_id

__all__ = [
    'get_device_id', 'get_device_num', 'get_job_id', 'get_rank_id'
]
