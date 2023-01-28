# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import sys
import shutil
from tests.security_utils import security_off_wrap
import pytest


def cleanup():
    data_path = os.path.join(os.getcwd(), "data")
    kernel_meta_path = os.path.join(os.getcwd(), "kernel_data")
    cache_path = os.path.join(os.getcwd(), "__pycache__")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    if os.path.exists(kernel_meta_path):
        shutil.rmtree(kernel_meta_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)


class CheckProfilerFiles:
    def __init__(self, device_id, rank_id, profiler_path, device_target):
        """Arges init."""
        self.device_id = device_id
        self.rank_id = rank_id
        self.profiler_path = profiler_path
        self.device_target = device_target
        if device_target == "Ascend":
            self._check_d_profiling_file()
        elif device_target == "GPU":
            self._check_gpu_profiling_file()
        else:
            self._check_cpu_profiling_file()

    def _check_gpu_profiling_file(self):
        """Check gpu profiling file."""
        op_detail_file = self.profiler_path + f'gpu_op_detail_info_{self.device_id}.csv'
        op_type_file = self.profiler_path + f'gpu_op_type_info_{self.device_id}.csv'
        activity_file = self.profiler_path + f'gpu_activity_data_{self.device_id}.csv'
        timeline_file = self.profiler_path + f'gpu_timeline_display_{self.device_id}.json'
        getnext_file = self.profiler_path + f'minddata_getnext_profiling_{self.device_id}.txt'
        pipeline_file = self.profiler_path + f'minddata_pipeline_raw_{self.device_id}.csv'
        framework_file = self.profiler_path + f'gpu_framework_{self.device_id}.txt'

        gpu_profiler_files = (op_detail_file, op_type_file, activity_file,
                              timeline_file, getnext_file, pipeline_file, framework_file)
        for file in gpu_profiler_files:
            assert os.path.isfile(file)

    def _check_d_profiling_file(self):
        """Check Ascend profiling file."""
        aicore_file = self.profiler_path + f'aicore_intermediate_{self.rank_id}_detail.csv'
        step_trace_file = self.profiler_path + f'step_trace_raw_{self.rank_id}_detail_time.csv'
        timeline_file = self.profiler_path + f'ascend_timeline_display_{self.rank_id}.json'
        aicpu_file = self.profiler_path + f'aicpu_intermediate_{self.rank_id}.csv'
        minddata_pipeline_file = self.profiler_path + f'minddata_pipeline_raw_{self.rank_id}.csv'
        queue_profiling_file = self.profiler_path + f'device_queue_profiling_{self.rank_id}.txt'
        memory_file = self.profiler_path + f'memory_usage_{self.rank_id}.pb'

        d_profiler_files = (aicore_file, step_trace_file, timeline_file, aicpu_file,
                            minddata_pipeline_file, queue_profiling_file, memory_file)
        for file in d_profiler_files:
            assert os.path.isfile(file)

    def _check_cpu_profiling_file(self):
        """Check cpu profiling file."""
        op_detail_file = self.profiler_path + f'cpu_op_detail_info_{self.device_id}.csv'
        op_type_file = self.profiler_path + f'cpu_op_type_info_{self.device_id}.csv'
        timeline_file = self.profiler_path + f'cpu_op_execute_timestamp_{self.device_id}.txt'

        cpu_profiler_files = (op_detail_file, op_type_file, timeline_file)
        for file in cpu_profiler_files:
            assert os.path.isfile(file)


class TestEnvEnableProfiler:
    profiler_path = os.path.join(os.getcwd(), f'data/profiler/')
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0

    @classmethod
    def setup_class(cls):
        """Run begin all test case start."""
        cleanup()

    @staticmethod
    def teardown():
        """Run after each test case end."""
        cleanup()

    @pytest.mark.level2
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_cpu_profiler(self):
        if sys.platform != 'linux':
            return
        status = os.system(
            """export MS_PROFILER_OPTIONS='{"start":true}';
               python ./run_net.py --target=CPU --mode=0;
            """
        )
        CheckProfilerFiles(self.device_id, self.rank_id, self.profiler_path, "CPU")
        assert status == 0

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_gpu_profiler(self):
        root_status = os.system("whoami | grep root")
        cuda_status = os.system("nvcc -V | grep 'release 10'")
        if root_status and not cuda_status:
            return
        status = os.system(
            """export MS_PROFILER_OPTIONS='{"start":true, "profile_memory":true, "sync_enable":true}';
               python ./run_net.py --target=GPU --mode=0;
            """
        )
        CheckProfilerFiles(self.device_id, self.rank_id, self.profiler_path, "GPU")
        assert status == 0

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_gpu_profiler_pynative(self):
        """
        Feature: profiler support GPU pynative mode.
        Description: profiling l2 GPU pynative mode data, analyze performance issues.
        Expectation: No exception.
        """
        root_status = os.system("whoami | grep root")
        cuda_status = os.system("nvcc -V | grep 'release 10'")
        if root_status and not cuda_status:
            return
        status = os.system(
            """export MS_PROFILER_OPTIONS='{"start":true, "sync_enable":true}';
               python ./run_net.py --target=GPU --mode=1;
            """
        )
        CheckProfilerFiles(self.device_id, self.rank_id, self.profiler_path, "GPU")
        assert status == 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_ascend_profiler(self):
        status = os.system(
            """export MS_PROFILER_OPTIONS='{"start":true, "profile_memory":true}';
               python ./run_net.py --target=Ascend --mode=0;
            """
        )
        CheckProfilerFiles(self.device_id, self.rank_id, self.profiler_path, "Ascend")
        assert status == 0
