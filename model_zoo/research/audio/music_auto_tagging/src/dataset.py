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
'''python dataset.py'''

import os
import mindspore.dataset as ds


def create_dataset(base_path, filename, batch_size, columns_list,
                   num_consumer):
    """Create dataset"""

    path = os.path.join(base_path, filename)
    dtrain = ds.MindDataset(path, columns_list, num_consumer)
    dtrain = dtrain.shuffle(buffer_size=dtrain.get_dataset_size())
    dtrain = dtrain.batch(batch_size, drop_remainder=True)

    return dtrain
