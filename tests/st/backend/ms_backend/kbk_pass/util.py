# Copyright 2024 Huawei Technologies Co., Ltd
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
import glob
import shutil
import random
import re
from contextlib import contextmanager
import mindspore as ms

class Capture():
    def __init__(self, opt_pass_name, operator_name):
        self.pass_name = opt_pass_name
        self.op_name = operator_name
        self.pass_output = None
        self.matched_file = None
        # since testcases are run simultaneously, need to keep IR paths different
        self.ir_path = f'IRs-{opt_pass_name}-{random.randint(0, 65535)}'.upper()

    def start(self):
        ms.set_context(save_graphs=True, save_graphs_path=self.ir_path)

    def stop(self):
        pattern = r'.+\d_' + self.pass_name
        # check and read opt pass output file
        for file_name in glob.iglob(f'{self.ir_path}/**/*.ir', recursive=True):
            if not re.match(pattern, file_name):
                continue
            self.matched_file = file_name
            with open(file_name, mode='r') as f:
                text = f.read()
                if self.op_name in text:
                    self.pass_output = text
                    break
            if self.pass_output:
                break
        # remove ir path recursively
        shutil.rmtree(self.ir_path, ignore_errors=True)

    def check_output(self, patterns):
        assert self.pass_output, f"Can not find ir file with name containing '" + \
                self.pass_name + "' and with operator '" + self.op_name + "' in it"
        index = 0
        for pattern in patterns:
            index = self.pass_output.find(pattern, index)
            assert index != -1, "Unexpected output:\n" + self.pass_output + "\n--- from file ---\n" \
                    + self.matched_file + "\n--- pattern ---\n" + pattern


@contextmanager
def capture(cap):
    cap.start()
    try:
        yield cap
    finally:
        cap.stop()
