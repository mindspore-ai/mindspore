# Copyright 2021 Huawei Technologies Co., Ltd
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
Utils for testing offline debugger.
"""

import os
import tempfile
import numpy as np


def build_dump_structure(path, tensor_name_list, tensor_list, net_name, tensor_info_list):
    """Build dump file structure from tensor_list."""
    temp_dir = tempfile.mkdtemp(prefix=net_name, dir=path)
    for tensor_name, tensor, tensor_info in zip(tensor_name_list, tensor_list, tensor_info_list):
        slot = str(tensor_info.slot)
        iteration = str(tensor_info.iteration)
        rank_id = str(tensor_info.rank_id)
        root_graph_id = str(tensor_info.root_graph_id)
        is_output = str(tensor_info.is_output)
        path = os.path.join(temp_dir, "rank_" + rank_id, net_name, root_graph_id, iteration)
        os.makedirs(path, exist_ok=True)
        if is_output == "True":
            file = tempfile.mkstemp(prefix=tensor_name, suffix=".output." + slot +
                                    ".DefaultFormat.npy", dir=path)
        else:
            file = tempfile.mkstemp(prefix=tensor_name, suffix=".input." + slot +
                                    ".DefaultFormat.npy", dir=path)
        full_path = file[1]
        np.save(full_path, tensor)
    return temp_dir
