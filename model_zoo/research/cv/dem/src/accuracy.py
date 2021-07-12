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
"""
utils, will be used in train.py
"""
import numpy as np
from src import kNN
from src import kNN_cosine

def compute_accuracy_att(net, pred_len, test_att_0, test_visual_0, test_id_0, test_label_0):
    att_pred_0 = net(test_att_0)
    outpred = [0] * pred_len
    test_label_0 = test_label_0.astype("float32")
    for i in range(pred_len):
        outputLabel = kNN.kNNClassify(test_visual_0[i, :], att_pred_0.asnumpy(), test_id_0, 1)
        outpred[i] = outputLabel
    outpred = np.array(outpred)
    acc_0 = np.equal(outpred, test_label_0).mean()
    return acc_0

def compute_accuracy_word(net, pred_len, test_word_0, test_visual_0, test_id_0, test_label_0):
    word_pred_0 = net(test_word_0)
    outpred = [0] * pred_len
    test_label_0 = test_label_0.astype("float32")
    for i in range(pred_len):
        outputLabel = kNN_cosine.kNNClassify(test_visual_0[i, :], word_pred_0.asnumpy(), test_id_0, 1)
        outpred[i] = outputLabel
    outpred = np.array(outpred)
    acc_0 = np.equal(outpred, test_label_0).mean()
    return acc_0

def compute_accuracy_fusion(net, pred_len, test_att_0, test_word_0, test_visual_0, test_id_0, test_label_0):
    fus_pred_0 = net(test_att_0, test_word_0)
    outpred = [0] * pred_len
    test_label_0 = test_label_0.astype("float32")
    for i in range(pred_len):
        outputLabel = kNN_cosine.kNNClassify(test_visual_0[i, :], fus_pred_0.asnumpy(), test_id_0, 1)
        outpred[i] = outputLabel
    outpred = np.array(outpred)
    acc_0 = np.equal(outpred, test_label_0).mean()
    return acc_0
