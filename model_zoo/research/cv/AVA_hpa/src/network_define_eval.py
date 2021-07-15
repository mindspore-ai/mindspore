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
"""define evaluation network"""
import time
import numpy as np
import mindspore.nn as nn
from mindspore.train.callback import Callback
import src.eval_metrics as eval_metrics



class EvalCallBack(Callback):
    """eval callback"""
    def __init__(self, model, eval_dataset, eval_per_epoch, epoch_per_eval, logger):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval
        self.logger = logger

    def epoch_end(self, run_context):
        """record evaluation results"""
        cb_param = run_context.original_args()
        epoch_idx = (cb_param.cur_step_num - 1) // cb_param.batch_num + 1
        if epoch_idx % self.eval_per_epoch == 0:
            start = time.time()
            output = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            print(output)
            lab_f1_macro, lab_f1_micro, lab_auc = output['results_return']
            end = time.time()
            self.logger.info("the {} epoch's Eval result: "
                             "f1_macro {}, f1_micro {}, auc {},"
                             "eval cost {:.2f} s".format(
                                 epoch_idx, lab_f1_macro, lab_f1_micro, lab_auc, end - start))

            self.epoch_per_eval["epoch"].append(epoch_idx)
            self.epoch_per_eval["f1_macro"].append(lab_f1_macro)
            self.epoch_per_eval["f1_micro"].append(lab_f1_micro)
            self.epoch_per_eval["auc"].append(lab_auc)


class EvalCell(nn.Cell):
    """eval cell"""
    def __init__(self, network, loss):
        super(EvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self.criterion = loss

    def construct(self, data, label, nslice):
        outputs = self._network(data)
        return outputs, label, nslice


class EvalMetric(nn.Metric):
    """eval metric"""
    def __init__(self, path):
        super(EvalMetric, self).__init__()
        self.clear()
        self.path = path

    def clear(self):
        """clear variables"""
        self.np_label = []
        self.np_pd = []
        self.np_score = []

        self.np_label_each_label = {}
        self.np_pd_each_label = {}
        self.np_score_each_label = {}
        self.label_num = 0

        self.cnt = 0

    def update(self, *inputs):
        """update middle variables"""
        val_predict = []
        cur = 0
        numpy_predict = inputs[0].asnumpy()
        label = inputs[1].asnumpy()
        nslice = inputs[2].asnumpy()

        self.label_num = label.shape[1]
        for i in range(len(label)):
            sample_bag_predict = np.mean(numpy_predict[int(cur): int(cur) + nslice[i]], axis=0)
            cur = cur + nslice[i]
            val_predict.append(sample_bag_predict)

        self.cnt = self.cnt + 1
        # save middle result
        val_pd = eval_metrics.threshold_tensor_batch(val_predict)
        self.np_pd.append(val_pd)
        self.np_score.append(val_predict)
        self.np_label.append(label)

        length = len(self.np_label_each_label)
        if length == 0:
            for i in range(self.label_num):
                self.np_label_each_label[i] = []
                self.np_pd_each_label[i] = []
                self.np_score_each_label[i] = []
        length = len(label)
        for i in range(length):
            for j in range(self.label_num):
                if label[i][j] == 1:
                    self.np_label_each_label[j].append(label[i].reshape(1, -1))
                    self.np_score_each_label[j].append(val_predict[i].reshape(1, -1))
                    self.np_pd_each_label[j].append(val_pd[i].reshape(1, -1))

    def eval(self):
        """eval final results"""
        self.np_label = np.concatenate(self.np_label)
        self.np_pd = np.concatenate(self.np_pd)
        self.np_score = np.concatenate(self.np_score)
        lab_f1_macro, lab_f1_micro, lab_auc = eval_metrics.np_metrics(self.np_label, self.np_pd, score=self.np_score,
                                                                      path=self.path)
        return lab_f1_macro, lab_f1_micro, lab_auc
