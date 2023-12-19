# Copyright 2023 Huawei Technologies Co., Ltd
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
"""abstract class for graph modifier"""

from abc import ABC, abstractmethod


from mslite_bench.utils import InferLogger

class ABCGraphModifier(ABC):
    """abstract class for graph modifier"""
    def __init__(self):
        self.blocks_sorted = self._sorted_blocks()
        self.logger = InferLogger().logger

    @property
    def sorted_blocks(self):
        """sorted blocks list based on feed-froward network"""
        return self.blocks_sorted

    @abstractmethod
    def extract_model(self,
                      save_path,
                      input_names=None,
                      output_names=None):
        """ extract sub model based on input and output tensor names"""
        raise NotImplementedError

    @abstractmethod
    def _all_node_names(self):
        """return all node names in network"""
        raise NotImplementedError

    @abstractmethod
    def _sorted_blocks(self):
        """get sorted blocks based on feed-foward network"""
        raise NotImplementedError
