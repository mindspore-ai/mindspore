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
"""Generate the lineage event which conform to proto format."""
from __future__ import absolute_import

import time

from mindspore.train.lineage_pb2 import LineageEvent


def serialize_to_lineage_event(name, value):
    """Serialize value to lineage event."""
    event = LineageEvent()
    event.wall_time = time.time()
    content = _get_lineage_content(name, event)
    content.ParseFromString(value)
    return event.SerializeToString()


def _get_lineage_content(name, event):
    if name == 'dataset_graph':
        return event.dataset_graph
    if name == 'eval_lineage':
        return event.evaluation_lineage
    if name == 'train_lineage':
        return event.train_lineage
    if name == 'custom_lineage_data':
        return event.user_defined_info
    raise KeyError(f'No such field in LineageEvent')
