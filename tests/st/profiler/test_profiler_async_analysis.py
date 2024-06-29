# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
import shutil
import csv
from tests.mark_utils import arg_mark


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
    def __init__(self, device_id, rank_id, profiler_path, device_target, profile_framework='all'):
        """Args init."""
        self.device_id = device_id
        self.rank_id = rank_id
        self.profiler_path = os.path.join(profiler_path, 'profiler')
        self.device_target = device_target
        if device_target == "Ascend":
            self._check_d_profiling_file()
        self._check_host_profiling_file(profile_framework=profile_framework)

    def _check_d_profiling_file(self):
        """Check Ascend profiling file."""
        aicore_file = f'aicore_intermediate_{self.rank_id}_detail.csv'
        timeline_file = f'ascend_timeline_display_{self.rank_id}.json'
        aicpu_file = f'aicpu_intermediate_{self.rank_id}.csv'
        minddata_pipeline_file = f'minddata_pipeline_raw_{self.rank_id}.csv'
        queue_profiling_file = f'device_queue_profiling_{self.rank_id}.txt'

        d_profiler_files = (aicore_file, timeline_file, aicpu_file, minddata_pipeline_file, queue_profiling_file)
        for _file in d_profiler_files:
            result_file = os.path.join(self.profiler_path, _file)
            assert os.path.isfile(result_file)

    def _check_host_profiling_file(self, profile_framework='all'):
        host_dir = os.path.join(self.profiler_path, 'host_info')
        if profile_framework is None:
            assert not os.path.exists(host_dir)
            return
        timeline_file = os.path.join(host_dir, f'timeline_{self.rank_id}.json')
        if profile_framework in ['all', 'time']:
            assert os.path.isfile(timeline_file)
        else:
            assert not os.path.exists(timeline_file)
        csv_file = os.path.join(host_dir, f'host_info_{self.rank_id}.csv')
        assert os.path.isfile(csv_file)
        with open(csv_file, 'r') as f:
            f_reader = csv.reader(f)
            header = next(f_reader)
            assert header == ['tid', 'pid', 'parent_pid', 'module_name', 'event', 'stage', 'level', 'start_end',
                              'custom_info', 'memory_usage(kB)', 'time_stamp(us)']
            for row in f_reader:
                assert len(row) == 11


class TestProfilerAsyncAnalysis:
    profiler_path = os.path.join(os.getcwd(), f'data')
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

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
    def test_ascend_profiler(self):
        status = os.system(
            """python ./run_net_with_profiler.py --target=Ascend --mode=0 --output_path=%s""" % self.profiler_path
        )
        CheckProfilerFiles(self.device_id, self.rank_id, self.profiler_path, "Ascend")
        assert status == 0
