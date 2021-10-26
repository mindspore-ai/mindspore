# Copyright 2019 Huawei Technologies Co., Ltd
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
This module is to  write data into mindrecord.
"""
import mindspore._c_mindrecord as ms
from mindspore import log as logger
from .common.exceptions import MRMIndexGeneratorError, MRMGenerateIndexError

__all__ = ['ShardIndexGenerator']


class ShardIndexGenerator:
    """
    Wrapper class which is represent ShardIndexGenerator class in c++ module.

    The class would generate db files for accelerating reading.

    Args:
        path (str): Absolute path of MindRecord File.
        append (bool): If True, open existed MindRecord Files for appending, or create new MindRecord Files.

    Raises:
        MRMIndexGeneratorError: If failed to create index generator.
    """
    def __init__(self, path, append=False):
        self._generator = ms.ShardIndexGenerator(path, append)
        if not self._generator:
            logger.critical("Failed to create index generator.")
            raise MRMIndexGeneratorError

    def build(self):
        """
        Build index generator.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMGenerateIndexError: If failed to build index generator.
        """
        ret = self._generator.build()
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to build index generator.")
            raise MRMGenerateIndexError
        return ret

    def write_to_db(self):
        """
        Create index field in table for reading data.

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            MRMGenerateIndexError: If failed to write to database.
        """
        ret = self._generator.write_to_db()
        if ret != ms.MSRStatus.SUCCESS:
            logger.critical("Failed to write to database.")
            raise MRMGenerateIndexError
        return ret
