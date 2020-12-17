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
"""Face Recognition Callback."""
import time
from mindspore.train.callback import Callback

class ProgressMonitor(Callback):
    '''ProgressMonitor'''
    def __init__(self, reid_args):
        super(ProgressMonitor, self).__init__()
        self.epoch_start_time = 0
        self.step_start_time = 0
        self.globe_step_cnt = 0
        self.local_step_cnt = 0
        self.reid_args = reid_args
        self._dataset_size = reid_args.steps_per_epoch

    def begin(self, run_context):
        self.run_context_ = run_context

        if not self.reid_args.epoch_cnt:
            self.reid_args.logger.info('start network train...')

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        if int(cb_params.cur_step_num / self._dataset_size) != self.reid_args.epoch_cnt:
            self.reid_args.logger.info('epoch end, local passed')
            self.reid_args.epoch_cnt += 1

    def step_begin(self, run_context):
        self.run_context_ = run_context

        self.step_start_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        time_used = time.time() - self.step_start_time
        cur_lr = self.reid_args.lrs[cb_params.cur_step_num]
        fps_mean = self.reid_args.per_batch_size * self.reid_args.log_interval * self.reid_args.world_size / time_used
        self.reid_args.logger.info('epoch[{}], iter[{}], loss:{}, cur_lr:{:.6f}, mean_fps:{:.2f} imgs/sec'.format(
            self.reid_args.epoch_cnt, cb_params.cur_step_num, cb_params.net_outputs, cur_lr, fps_mean))
        self.step_start_time = time.time()

    def end(self, run_context):
        self.run_context_ = run_context

        self.reid_args.logger.info('end network train...')
