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
"""Callbacks file"""
from mindspore.train.callback import Callback

class EvalCallBack(Callback):
    """
    Monitor the loss in evaluate.
    """
    def __init__(self, model, eval_dataset, metric, eval_file_path="./eval.log"):
        super(EvalCallBack, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.metric = metric
        self.metric.clear()
        self.eval_file_path = eval_file_path
        self.run_context = None

    def epoch_end(self, run_context):
        self.run_context = run_context
        self.metric.clear()
        out = self.model.eval(self.eval_dataset)

        eval_file = open(self.eval_file_path, "a+")
        eval_file.write("EvalCallBack: HR = {}, NDCG = {}\n".format(out['ncf'][0], out['ncf'][1]))
        eval_file.close()
        print("EvalCallBack: HR = {}, NDCG = {}".format(out['ncf'][0], out['ncf'][1]))
