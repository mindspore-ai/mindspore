# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Data access adapter module for supporting both file and IO object access in safetensors loading.

This module provides an adapter pattern implementation that allows the PySafeSlice class
to work with both local files (using memory mapping) and IO objects (like BytesIO)
without changing the core logic. The adapter abstracts the differences between file-based
and IO-based data access, enabling lazy loading for files while fully loading IO objects.
"""
import io
import mmap
from abc import ABC, abstractmethod
from typing import Union, Any, IO


class DataAccessAdapter(ABC):
    """
    Abstract base class for data access adapters.

    This adapter abstracts the differences between file-based and IO-based data access,
    allowing PySafeSlice to work with both local files and IO objects.
    """

    def __init__(self, file_source: IO[bytes]):
        """
        Initialize the data access adapter.

        Args:
            file_source: Either an opened file or an IO object
        """
        self.buffer = file_source

    def close(self) -> None:
        """
        Close any resources held by the adapter.
        """

    @property
    @abstractmethod
    def data_source(self) -> bytes:
        """
        Get the data source for tensor access.

        Returns:
            The bytes data source for tensor access
        """


class FileDataAccessAdapter(DataAccessAdapter):
    """
    Data access adapter for local file access using memory mapping.

    This adapter provides efficient access to safetensors files using memory mapping,
    enabling lazy loading and minimal memory usage.
    """

    def __init__(self, filename: str):
        """
        Initialize the file adapter.

        Args:
            filename: Path to the safetensors file
        """
        # File must remain open for mmap and will be closed by close() method
        # pylint: disable=consider-using-with
        fd = open(filename, "rb")
        super().__init__(fd)
        self.__file_mmap = mmap.mmap(self.buffer.fileno(), 0, access=mmap.ACCESS_COPY)

    def close(self) -> None:
        """
        Close the file and memory mapping resources.
        """
        self.buffer.close()

    @property
    def data_source(self) -> Any:
        """
        Get the memory-mapped file as data source.

        Returns:
            The memory-mapped file content as bytes
        """
        return self.__file_mmap


class IODataAccessAdapter(DataAccessAdapter):
    """
    Data access adapter for IO objects like BytesIO.

    This adapter loads the entire content into memory since IO objects don't support
    memory mapping, and provides tensor access from the loaded buffer.
    """

    def __init__(self, io_object: IO[bytes]):
        """
        Initialize the IO adapter.

        Args:
            io_object: An IO object like BytesIO containing the safetensors data
        """
        super().__init__(io_object)

        # Store the current position
        original_pos = io_object.tell()
        # Read all data
        self.__buffer_data = io_object.read()
        # Restore original position
        io_object.seek(original_pos)

    @property
    def data_source(self) -> bytes:
        """
        Get the in-memory buffer as data source.

        Returns:
            The in-memory buffer data as bytes
        """
        return self.__buffer_data


def create_data_access_adapter(source: Union[str, IO[bytes]]) -> DataAccessAdapter:
    """
    Factory method to create the appropriate data access adapter based on the input source.

    Args:
        source: Either a file path string or an IO object (like BytesIO)

    Returns:
        An instance of the appropriate DataAccessAdapter subclass
    """
    if isinstance(source, str):
        return FileDataAccessAdapter(source)
    if isinstance(source, io.BytesIO):
        return IODataAccessAdapter(source)
    raise TypeError(f"Unsupported source type: {type(source)}. "
                    f"Expected str or io.BytesIO.")
