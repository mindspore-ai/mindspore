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
"""Utils for parameter server training mode"""

from mindspore._c_expression import get_ps_mode_rank

def _get_ps_mode_rank():
    ps_rank = get_ps_mode_rank()
    if ps_rank == -1:
        raise RuntimeError("The parameter server mode training is not launched yet.")
    return ps_rank
