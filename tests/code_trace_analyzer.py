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
"""Code Trace Analyzer utils."""
import os
import time
import types
import re
import inspect
from mindspore import nn


class CodeTraceAnalyzer:
    """
    Code Trace Analyzer.

    Args:
        obj (Cell, Function): The obj.
        save_graphs_path (str): The path of saved ir files.
        ir_file (str): The ir files to be read. Default: execute.
    """

    def __init__(self, obj, save_graphs_path, ir_file="execute"):
        self.obj = obj
        self.save_graphs_path = save_graphs_path
        self.ir_content = ""
        self.ir_file_name = ir_file
        self.code_lines = 0
        self.ignore_code_lines = 0
        self.traced_code_lines = 0
        self.not_traced_codes = []
        self.accuracy = 0.0
        self.analyzed = False
        self.dir_black_lists = ["mindspore/nn", "mindspore/ops", "mindspore/train"]
        self.cost_time = 0
        self.extra_fns = []

    @staticmethod
    def skip_check(line):
        line_strip = line.strip()
        if not line_strip:  # blank line
            return True
        if line_strip[0] == '#':  # comment
            return True
        if line_strip[0] == '@':  # decorator
            return True
        if len(line_strip) > 3 and line_strip[0:4] == "def ":  # function define
            return True

        return False

    def add_functions(self, *functions):
        """Add more functions those are not top functions to analyze."""
        for fn in functions:
            if not isinstance(fn, (types.FunctionType, types.MethodType)):
                raise ValueError(f"{fn} must be a Function")
            self.extra_fns.append(fn)

    def analyze(self):
        """Start to analyze the code trace accuracy and return the accuracy."""
        if self.analyzed:
            raise ValueError(f"analyze() can only call once.")

        start_time = time.time()
        self._read_ir_content()
        if isinstance(self.obj, nn.Cell):
            self._check_net(self.obj)
        elif isinstance(self.obj, (types.FunctionType, types.MethodType)):
            self._check_function(self.obj)
        else:
            raise ValueError(f"Obj {self.obj} muse be a Cell or Function.")

        for fn in self.extra_fns:
            self._check_function(fn)

        self.analyzed = True
        self.accuracy = self.traced_code_lines / (self.code_lines - self.ignore_code_lines)
        self.cost_time = time.time() - start_time
        return self.accuracy

    def report_analysis(self):
        """Report the analysis."""
        if not self.analyzed:
            print("Please run analyze() success first.")
            return

        print("\n------Code Trace Analysis-------")
        print(f"The code trace accuracy is {self.accuracy}")
        print(
            f"All of code lines is {self.code_lines}, ignored code lines is {self.ignore_code_lines}. "
            f"And there are {self.traced_code_lines} of codes appeared in the ir file: {self.ir_file_name}")
        print(f"#analyze() cost time: {self.cost_time}s")
        if self.not_traced_codes:
            print(f"Below codes are not traced in ir file:")
            for index, line in enumerate(self.not_traced_codes):
                print(f"[{index}] {line}")

    def _read_ir_content(self):
        """Get the content of the last ir file"""
        ir_files = map(lambda f: os.path.join(self.save_graphs_path, f),
                       filter(lambda f: re.match(rf'\d+_{self.ir_file_name}_\d+.ir', f),
                              os.listdir(self.save_graphs_path)))
        file_name = max(ir_files, key=os.path.getctime)
        with open(os.path.join(file_name), 'r') as f:
            self.ir_content = f.read()
        self.ir_file_name = file_name

    def _check_lines(self, fn):
        fn_file_name: str = fn.__code__.co_filename
        for item in self.dir_black_lists:
            if item in fn_file_name:
                return

        lines, offset = inspect.getsourcelines(fn)
        for index, line in enumerate(lines):
            line = line.replace('\n', '').replace('\r', '')
            if self.skip_check(line):
                continue

            self.code_lines += 1

            if "<IGNORE>" in line:
                self.ignore_code_lines += 1
                continue

            line = f"In file {fn_file_name}:{offset + index}/{line}/"
            if line in self.ir_content:
                self.traced_code_lines += 1
            else:
                self.not_traced_codes.append(line)

    def _check_net(self, cell):
        """Recursively check the cell and its sub cell except the mindspore inner cells"""
        fn = getattr(cell, "construct")
        fn = inspect.unwrap(fn)
        self._check_lines(fn)

        for sub_cell in cell.cells():
            self._check_net(sub_cell)

    def _check_function(self, fn):
        """"Only Check the given function"""
        fn = inspect.unwrap(fn)
        self._check_lines(fn)
