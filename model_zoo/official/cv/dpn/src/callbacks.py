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
import os
from mindspore.train.callback import Callback
from mindspore.train.serialization import save_checkpoint


class SaveCallback(Callback):
    """
    Evaluating on eval_dataset after each epoch.
    And it will save the parameters if the accuracy is better.
    """

    def __init__(self, model, eval_dataset, ckpt_path):
        super(SaveCallback, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.acc = 0.2
        self.ckpt_path = ckpt_path

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        result = self.model.eval(self.eval_dataset)
        print("epoch", epoch_num, " top_1_accuracy:", result['top_1_accuracy'])
        if result['top_1_accuracy'] > self.acc:
            self.acc = result['top_1_accuracy']
            file_name = "max.ckpt"
            file_name = os.path.join(self.ckpt_path, file_name)
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the accuracy is", self.acc)
