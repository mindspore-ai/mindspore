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
"""postprocess"""
import os
import numpy as np

from mindspore import Tensor
from src.metrics import NCFMetric
from model_utils.config import config

def get_acc():
    """calculate accuracy"""
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    ncf_metric = NCFMetric()
    rst_path = config.post_result_path
    file_num = len(os.listdir(rst_path))//3
    bs = config.eval_batch_size
    for i in range(file_num):
        indice_name = os.path.join(rst_path, "ncf_bs" + str(bs) + "_" + str(i) + "_0.bin")
        item_name = os.path.join(rst_path, "ncf_bs" + str(bs) + "_" + str(i) + "_1.bin")
        weight_name = os.path.join(rst_path, "ncf_bs" + str(bs) + "_" + str(i) + "_2.bin")

        batch_indices = np.fromfile(indice_name, np.int32).reshape(1600, 10)
        batch_items = np.fromfile(item_name, np.int32).reshape(1600, 100)
        metric_weights = np.fromfile(weight_name, bool)
        ncf_metric.update(Tensor(batch_indices), Tensor(batch_items), Tensor(metric_weights))

    out = ncf_metric.eval()

    eval_file_path = os.path.join(config.output_path, config.eval_file_name)
    eval_file = open(eval_file_path, "a+")
    eval_file.write("EvalCallBack: HR = {}, NDCG = {}\n".format(out[0], out[1]))
    eval_file.close()
    print("EvalCallBack: HR = {}, NDCG = {}".format(out[0], out[1]))
    print("=" * 100 + "Eval Finish!" + "=" * 100)

if __name__ == '__main__':
    get_acc()
