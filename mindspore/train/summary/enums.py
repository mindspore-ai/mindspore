# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Summary's enumeration file."""
from enum import Enum


class BaseEnum(Enum):
    """The base enum class."""

    @classmethod
    def to_list(cls):
        """Converts the enumeration into a list."""
        return [member.value for member in cls.__members__.values()]


class PluginEnum(BaseEnum):
    """The list of plugins currently supported by the summary."""
    GRAPH = 'graph'
    SCALAR = 'scalar'
    IMAGE = 'image'
    TENSOR = 'tensor'
    HISTOGRAM = 'histogram'
    TRAIN_LINEAGE = 'train_lineage'
    EVAL_LINEAGE = 'eval_lineage'
    CUSTOM_LINEAGE_DATA = 'custom_lineage_data'
    DATASET_GRAPH = 'dataset_graph'
    EXPLAINER = 'explainer'


class WriterPluginEnum(Enum):
    """The list of extra plugins."""
    EXPORTER = 'exporter'
    EXPLAINER = 'explainer'
    SUMMARY = 'summary'
    LINEAGE = 'lineage'


class ModeEnum(BaseEnum):
    """The modes currently supported by the summary."""
    TRAIN = 'train'
    EVAL = 'eval'
