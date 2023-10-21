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
# ==============================================================================
"""Efficient line-based file reading.
"""
import os

from mindspore import log as logger
from ..core.validator_helpers import check_filename, check_uint64, check_value


class LineReader:
    """
    Line-based file reader.

    Cache the line-based meta data of the file in advance to achieve random-access reading of each file line.

    Args:
        filename (str): Filename to be read.

    Raises:
        TypeError: If `filename` is not of type int.
        RuntimeError: If `filename` does not exist or is not a regular file.

    Examples:
        >>> from mindspore.dataset import LineReader
        >>>
        >>> reader = LineReader("/path/to/txt/or/csv/file")
        >>> # Read the first line of csv file
        >>> reader.readline(1)
        >>> # Return the row size in csv file
        >>> reader.len()
        >>> # Close the handle
        >>> reader.close()
    """

    def __init__(self, filename):
        check_filename(filename)
        self.filename = os.path.realpath(filename)

        if not os.path.exists(self.filename):
            raise RuntimeError("The input file [{}] does not exist.".format(filename))

        if not os.path.isfile(self.filename):
            raise RuntimeError("The input file [{}] is not a regular file.".format(filename))

        # get the line offsets
        self.offsets = [0]
        with open(self.filename, mode='rb') as fo:
            while fo.readline():
                self.offsets.append(fo.tell())

        # pop the last empty line offset
        self.offsets.pop()
        if not self.offsets:
            logger.warning("The input file [{}] is empty.".format(filename))

        # will be init in readline
        self.fo_handle = None

    def __getitem__(self, line):
        """Read specified line content"""
        return self.readline(line)

    def __len__(self):
        """Get the total number of lines in the file"""
        return self.len()

    def __del__(self):
        """Close the file when object released"""
        self.close()

    def len(self):
        """Get the total number of lines in the current file."""
        return len(self.offsets)

    def readline(self, line):
        """
        Reads the contents of the specified line.

        Args:
            line (int): The line number to be read, with a starting line number of 1.

        Returns:
            str, the contents of the corresponding line, without line break characters.

        Raises:
            TypeError: If `line` is not of type int.
            ValueError: If `line` exceeds the total number of lines in the file.
        """
        check_uint64(line, "line")
        check_value(line, [1, len(self.offsets)], "line")
        if self.fo_handle is None:
            self.fo_handle = open(self.filename, mode="rt")

        self.fo_handle.seek(self.offsets[line - 1])
        content = self.fo_handle.readline()

        # remove the line break character
        if content.endswith("\r\n"):
            content = content[:-2]
        elif content.endswith("\n"):
            content = content[:-1]
        elif content.endswith("\r"):
            content = content[:-1]
        return content

    def close(self):
        """Close the file handle."""
        if self.fo_handle is None:
            return
        self.fo_handle.close()
        self.fo_handle = None
