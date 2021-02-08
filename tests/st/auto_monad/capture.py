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
# ==============================================================================
import os
import sys
import tempfile
from contextlib import contextmanager

class Capture():
    def start(self):
        self._old_stdout = sys.stdout
        self._stdout_fd = self._old_stdout.fileno()
        self._saved_stdout_fd = os.dup(self._stdout_fd)
        self._file = sys.stdout = tempfile.TemporaryFile(mode='w+t')
        self.output = ''
        os.dup2(self._file.fileno(), self._stdout_fd)

    def stop(self):
        os.dup2(self._saved_stdout_fd, self._stdout_fd)
        os.close(self._saved_stdout_fd)
        sys.stdout = self._old_stdout
        self._file.seek(0)
        self.output = self._file.read()
        self._file.close()

@contextmanager
def capture(cap):
    cap.start()
    try:
        yield cap
    finally:
        cap.stop()

def check_output(output, patterns):
    assert output, "Capture output failed!"
    for pattern in patterns:
        assert output.find(pattern) != -1, "Unexpected output:\n" + output + "\n--- pattern ---\n" + pattern
