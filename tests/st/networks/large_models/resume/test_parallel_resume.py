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
"""
Test module for testing resume training from specified checkpoint.
How to run this:
    pytest tests/st/networks/large_models/resume/test_parallel_resume.py
"""

import os
import shutil

from tests.mark_utils import arg_mark
from resume_train_utils import extract_loss_values


def remove_folder(folder_path):
    """
    If the folder exists, delete it and all its contents.

    Args:
      folder_path: The path to the folder to delete.
    """
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Directory '{folder_path}' has been removed.")
        except OSError as e:
            print(f"Remove directory '{folder_path}' failed: {e}")
    else:
        print(f"Directory '{folder_path}' is not exist.")


class TestResumeTraining:
    """A test class for testing pipeline."""

    @arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
    def test_train(self):
        """
        Feature: Trainer.train()
        Description: Test parallel trainer for train.
        Expectation: AssertionError
        """
        ascend_home_path = os.getenv('ASCEND_HOME_PATH')
        if not ascend_home_path:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        ret = os.system(f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 && "
                        f"bash {sh_path}/msrun_launch.sh 4")

        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log/worker_3.log -C 3")
        assert ret == 0

        loss = extract_loss_values("msrun_log/worker_3.log")

        resume_start = 256
        train_middle = 128
        for i in range(0, 10):
            assert abs(loss[resume_start + i] - loss[train_middle + i]) < 0.005

        remove_folder("./output/test_resume_parallel/checkpoint")
