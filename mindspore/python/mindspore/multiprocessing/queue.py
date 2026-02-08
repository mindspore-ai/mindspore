# Copyright 2025 Huawei Technologies Co., Ltd
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
""" multiprocessiong queue """

import io
import pickle
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler


class IpcConnectionWrapper:
    """use ForkingPickler to serialize and deserialize object."""

    def __init__(self, connection):
        self.connection = connection

    def send(self, obj):
        """Serialize and send obj."""
        bytes_buffer = io.BytesIO()
        ForkingPickler(bytes_buffer, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(bytes_buffer.getvalue())

    def recv(self):
        """Receive and deserialize obj."""
        bytes_buffer = self.recv_bytes()
        return pickle.loads(bytes_buffer)

    def __getattr__(self, name):
        if "connection" in self.__dict__:
            return getattr(self.connection, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute 'connection'")


class Queue(multiprocessing.queues.Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reader: IpcConnectionWrapper = IpcConnectionWrapper(self._reader)
        self._writer: IpcConnectionWrapper = IpcConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv
