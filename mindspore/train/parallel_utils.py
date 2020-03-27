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
"""Parallel utils"""

__all__ = ["ParallelMode"]


class ParallelMode:
    """
    Parallel mode options.

    There are five kinds of parallel modes, "STAND_ALONE", "DATA_PARALLEL",
    "HYBRID_PARALLEL", "SEMI_AUTO_PARALLEL" and "AUTO_PARALLEL". Default: "STAND_ALONE".

        - STAND_ALONE: Only one processor working.
        - DATA_PARALLEL: Distributing the data across different processors.
        - HYBRID_PARALLEL: Achieving data parallelism and model parallelism manually.
        - SEMI_AUTO_PARALLEL: Achieving data parallelism and model parallelism by setting parallel strategies.
        - AUTO_PARALLEL: Achieving parallelism automatically.

    MODE_LIST: The list for all supported parallel modes.
    """

    STAND_ALONE = "stand_alone"
    DATA_PARALLEL = "data_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    SEMI_AUTO_PARALLEL = "semi_auto_parallel"
    AUTO_PARALLEL = "auto_parallel"
    MODE_LIST = [STAND_ALONE, DATA_PARALLEL, HYBRID_PARALLEL, SEMI_AUTO_PARALLEL, AUTO_PARALLEL]
