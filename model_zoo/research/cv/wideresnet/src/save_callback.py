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
"""save best ckpt"""

from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback import Callback
from src.config import config_WideResnet as cfg


class SaveCallback(Callback):
    """
    save best ckpt
    """
    def __init__(self, model, eval_dataset, ckpt_path, modelart):
        super(SaveCallback, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.ckpt_path = ckpt_path
        self.acc = 0.96
        self.cur_acc = 0.0
        self.modelart = modelart

    def step_end(self, run_context):
        """
        step end
        """
        cb_params = run_context.original_args()
        result = self.model.eval(self.eval_dataset)
        self.cur_acc = result['accuracy']
        print("cur_acc is", self.cur_acc)

        if result['accuracy'] > self.acc:
            self.acc = result['accuracy']
            file_name = "WideResNet_best" + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            if self.modelart:
                import moxing as mox
                mox.file.copy_parallel(src_url=cfg.save_checkpoint_path, dst_url=self.ckpt_path)
            print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
