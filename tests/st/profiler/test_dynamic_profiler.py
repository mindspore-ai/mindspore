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
# ============================================================================
import os
import shutil
import tempfile
from tests.mark_utils import arg_mark


def cleanup():
    kernel_meta_path = os.path.join(os.getcwd(), "kernel_data")
    cache_path = os.path.join(os.getcwd(), "__pycache__")
    if os.path.exists(kernel_meta_path):
        shutil.rmtree(kernel_meta_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)


class TestProfiler:

    def setup(self):
        """Run begin each test case start."""
        cleanup()
        self.data_path = tempfile.mkdtemp(prefix='profiler_data', dir='/tmp')

    def teardown(self):
        """Run after each test case end."""
        cleanup()
        if os.path.exists(self.data_path):
            shutil.rmtree(self.data_path)

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_ascend_profiler(self):
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        status = os.system(
            f"""
               python ./run_net_with_dynamic_profiler.py --target=Ascend --mode=0 --output_path={self.data_path};
            """
        )
        assert status == 0
        for i in range(5, 16, 5):
            profiler_path = os.path.join(self.data_path, str(i), str(rank_id), 'profiler')
            assert os.path.exists(profiler_path)
