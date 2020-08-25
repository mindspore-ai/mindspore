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
"""
Area under cure metric
"""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from mindspore.nn.metrics import Metric


def groupby_df_v1(test_df, gb_key):
    """
    groupby_df_v1
    """
    data_groups = test_df.groupby(gb_key)
    return data_groups


def _compute_metric_v1(batch_groups, topk):
    """
    _compute_metric_v1
    """
    results = []
    for df in batch_groups:
        df = df.sort_values(by="preds", ascending=False)
        if df.shape[0] > topk:
            df = df.head(topk)
        preds = df["preds"].values
        labels = df["labels"].values
        if np.sum(labels) > 0:
            results.append(average_precision_score(labels, preds))
        else:
            results.append(0.0)
    return results


def mean_AP_topk(batch_labels, batch_preds, topk=12):
    """
    mean_AP_topk
    """
    def ap_score(label, y_preds, topk):
        ind_list = np.argsort(y_preds)[::-1]
        ind_list = ind_list[:topk]
        if label not in set(ind_list):
            return 0.0
        rank = list(ind_list).index(label)
        return 1.0 / (rank + 1)

    mAP_list = []
    for label, preds in zip(batch_labels, batch_preds):
        mAP = ap_score(label, preds, topk)
        mAP_list.append(mAP)
    return mAP_list


def new_compute_mAP(test_df, gb_key="display_ids", top_k=12):
    """
    new_compute_mAP
    """
    total_start = time.time()
    display_ids = test_df["display_ids"]
    labels = test_df["labels"]
    predictions = test_df["preds"]

    test_df.sort_values(by=[gb_key], inplace=True, ascending=True)
    display_ids = test_df["display_ids"]
    labels = test_df["labels"]
    predictions = test_df["preds"]

    _, display_ids_idx = np.unique(display_ids, return_index=True)

    preds = np.split(predictions.tolist(), display_ids_idx.tolist()[1:])
    labels = np.split(labels.tolist(), display_ids_idx.tolist()[1:])

    def pad_fn(ele_l):
        res_list = ele_l + [0.0 for i in range(30 - len(ele_l))]
        return res_list

    preds = list(map(lambda x: pad_fn(x.tolist()), preds))
    labels = [np.argmax(l) for l in labels]

    result_list = []

    batch_size = 100000
    for idx in range(0, len(labels), batch_size):
        batch_labels = labels[idx:idx + batch_size]
        batch_preds = preds[idx:idx + batch_size]
        meanAP = mean_AP_topk(batch_labels, batch_preds, topk=top_k)
        result_list.extend(meanAP)
    mean_AP = np.mean(result_list)
    print("compute time: {}".format(time.time() - total_start))
    print("mean_AP: {}".format(mean_AP))
    return mean_AP


class AUCMetric(Metric):
    """
    AUCMetric
    """
    def __init__(self):
        super(AUCMetric, self).__init__()
        self.index = 1

    def clear(self):
        """Clear the internal evaluation result."""
        self.true_labels = []
        self.pred_probs = []
        self.display_id = []

    def update(self, *inputs):
        """
        update
        """
        all_predict = inputs[1].asnumpy()  # predict
        all_label = inputs[2].asnumpy()  # label
        all_display_id = inputs[3].asnumpy()  # label
        self.true_labels.extend(all_label.flatten().tolist())
        self.pred_probs.extend(all_predict.flatten().tolist())
        self.display_id.extend(all_display_id.flatten().tolist())

    def eval(self):
        """
        eval
        """
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError(
                'true_labels.size() is not equal to pred_probs.size()')

        result_df = pd.DataFrame({
            "display_ids": self.display_id,
            "preds": self.pred_probs,
            "labels": self.true_labels,
        })
        auc = roc_auc_score(self.true_labels, self.pred_probs)

        MAP = new_compute_mAP(result_df, gb_key="display_ids", top_k=12)
        print("Eval result:"  + " auc: {}, map: {}".format(auc, MAP))
        return auc
