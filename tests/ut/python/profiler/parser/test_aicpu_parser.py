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
"""Test the aicpu parser."""
import os
import tempfile
import shutil

from mindspore.profiler.parser.aicpu_data_parser import DataPreProcessParser


class TestAicpuParser:
    """Test the class of Aicpu Parser."""
    def setup_class(self):
        """Initialization before test case execution."""


    def teardown_method(self) -> None:
        """Clear output file."""
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

    def test_aicpu_parser_binary(self):
        """Test the class of aicpu binary data Parser."""
        self.profiling_dir = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                           '../../../data/profiler_data/'
                                                           'JOB_AICPU/data'))
        self.expect_dir = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                        '../../../data/profiler_data/'
                                                        'JOB_AICPU/expect'))
        self.output_path = tempfile.mkdtemp(prefix='output_data_preprocess_aicpu_')
        self.output_file = os.path.join(self.output_path, 'output_data_preprocess_aicpu_0.txt')
        self.expect_file = os.path.join(self.expect_dir, 'output_data_preprocess_aicpu_0.txt')
        self.op_task_dict = {
            "22_2": "Default/network-_VirtualDatasetCell/_backbone-WithLossCell/_backbone-AlexNet/dropout-Dropout/"
                    "DropoutGenMask-op281",
            "22_4": "Default/network-_VirtualDatasetCell/_backbone-WithLossCell/_backbone-AlexNet/dropout-Dropout/"
                    "DropoutGenMask-op280"
        }

        data = DataPreProcessParser(self.profiling_dir, self.output_file, self.op_task_dict)
        data.execute()
        with open(self.expect_file, 'r') as fp:
            expect_result = fp.read()
        with open(self.output_file, 'r') as fp:
            result = fp.read()
        assert expect_result == result

    def test_aicpu_parser_txt(self):
        """Test the class of aicpu txt data Parser."""
        self.profiling_dir = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                           '../../../data/profiler_data/'
                                                           'JOB_AICPU/data_txt'))
        self.expect_dir = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                        '../../../data/profiler_data/'
                                                        'JOB_AICPU/expect_txt'))
        self.output_path = tempfile.mkdtemp(prefix='output_data_preprocess_aicpu_')
        self.output_file = os.path.join(self.output_path, 'output_data_preprocess_aicpu_0.txt')
        self.expect_file = os.path.join(self.expect_dir, 'output_data_preprocess_aicpu_0.txt')

        data = DataPreProcessParser(self.profiling_dir, self.output_file, None)
        data.execute()
        with open(self.expect_file, 'r') as fp:
            expect_result = fp.read()
        with open(self.output_file, 'r') as fp:
            result = fp.read()
        assert expect_result == result
