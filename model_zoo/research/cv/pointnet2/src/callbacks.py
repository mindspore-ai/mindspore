# Copyright 2021 Huawei Technologies Co., Ltd
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
"""callbacks"""

import moxing as mox
from mindspore.train.callback import Callback


class MoxCallBack(Callback):
    """Mox training files from online"""

    def __init__(self, local_train_url, train_url, mox_freq):
        super(MoxCallBack, self).__init__()
        self.local_train_url = local_train_url
        self.train_url = train_url
        self.mox_freq = mox_freq

    def epoch_end(self, run_context):
        """Mox files at the end of each epoch"""
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.mox_freq == 0:
            mox.file.copy_parallel(self.local_train_url, self.train_url)
