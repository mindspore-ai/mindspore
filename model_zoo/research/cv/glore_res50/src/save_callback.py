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
"""define savecallback, save best model while training."""
from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class SaveCallback(Callback):
    """
    define savecallback, save best model while training.
    """
    def __init__(self, model_save, eval_dataset_save, save_file_path):
        super(SaveCallback, self).__init__()
        self.model = model_save
        self.eval_dataset = eval_dataset_save
        self.acc = 0.78
        self.save_path = save_file_path

    def step_end(self, run_context):
        """
        eval and save model while training.
        """
        cb_params = run_context.original_args()

        result = self.model.eval(self.eval_dataset)
        print(result)
        if result['Accuracy'] > self.acc:
            self.acc = result['Accuracy']
            file_name = self.save_path + str(self.acc) + ".ckpt"
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
