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
"""Utils for NAML."""
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from mindspore import Tensor

from .dataset import create_eval_dataset, EvalNews, EvalUsers, EvalCandidateNews

def get_metric(args, mindpreprocess, news_encoder, user_encoder, metric):
    """Calculate metrics."""
    start = time.time()
    news_dict = {}
    user_dict = {}
    dataset = create_eval_dataset(mindpreprocess, EvalNews, batch_size=args.batch_size)
    dataset_size = dataset.get_dataset_size()
    iterator = dataset.create_dict_iterator(output_numpy=True)
    for count, data in enumerate(iterator):
        news_vector = news_encoder(Tensor(data["category"]), Tensor(data["subcategory"]),
                                   Tensor(data["title"]), Tensor(data["abstract"])).asnumpy()
        for i, nid in enumerate(data["news_id"]):
            news_dict[str(nid[0])] = news_vector[i]
        print(f"===Generate News vector==== [ {count} / {dataset_size} ]", end='\r')
    print(f"===Generate News vector==== [ {dataset_size} / {dataset_size} ]")
    dataset = create_eval_dataset(mindpreprocess, EvalUsers, batch_size=args.batch_size)
    dataset_size = dataset.get_dataset_size()
    iterator = dataset.create_dict_iterator(output_numpy=True)
    for count, data in enumerate(iterator):
        browsed_news = []
        for newses in data["history"]:
            news_list = []
            for nid in newses:
                news_list.append(news_dict[str(nid[0])])
            browsed_news.append(np.array(news_list))
        browsed_news = np.array(browsed_news)
        user_vector = user_encoder(Tensor(browsed_news)).asnumpy()
        for i, uid in enumerate(data["uid"]):
            user_dict[str(uid)] = user_vector[i]
        print(f"===Generate Users vector==== [ {count} / {dataset_size} ]", end='\r')
    print(f"===Generate Users vector==== [ {dataset_size} / {dataset_size} ]")
    dataset = create_eval_dataset(mindpreprocess, EvalCandidateNews, batch_size=args.batch_size)
    dataset_size = dataset.get_dataset_size()
    iterator = dataset.create_dict_iterator(output_numpy=True)
    for count, data in enumerate(iterator):
        pred = np.dot(
            np.stack([news_dict[str(nid)] for nid in data["candidate_nid"]], axis=0),
            user_dict[str(data["uid"])]
        )
        metric.update(pred, data["labels"])
        print(f"===Click Prediction==== [ {count} / {dataset_size} ]", end='\r')
    print(f"===Click Prediction==== [ {dataset_size} / {dataset_size} ]")
    auc = metric.eval()
    total_cost = time.time() - start
    print(f"Eval total cost: {total_cost} s")
    return auc

def process_data(args):
    word_embedding = np.load(args.embedding_file)
    _, h = word_embedding.shape
    if h < args.word_embedding_dim:
        word_embedding = np.pad(word_embedding, ((0, 0), (0, args.word_embedding_dim - 300)), 'constant',
                                constant_values=0)
    elif h > args.word_embedding_dim:
        word_embedding = word_embedding[:, :args.word_embedding_dim]
    print("Load word_embedding", word_embedding.shape)
    return Tensor(word_embedding.astype(np.float32))

def AUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def MRR(y_true, y_pred):
    index = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, index)
    score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(score) / np.sum(y_true)

def DCG(y_true, y_pred, n):
    index = np.argsort(y_pred)[::-1]
    y_true = np.take(y_true, index[:n])
    score = (2 ** y_true - 1) / np.log2(np.arange(len(y_true)) + 2)
    return np.sum(score)

def nDCG(y_true, y_pred, n):
    return DCG(y_true, y_pred, n) / DCG(y_true, y_true, n)

class NAMLMetric:
    """
    Metric method
    """
    def __init__(self):
        super(NAMLMetric, self).__init__()
        self.AUC_list = []
        self.MRR_list = []
        self.nDCG5_list = []
        self.nDCG10_list = []

    def clear(self):
        """Clear the internal evaluation result."""
        self.AUC_list = []
        self.MRR_list = []
        self.nDCG5_list = []
        self.nDCG10_list = []

    def update(self, predict, y_true):
        predict = predict.flatten()
        y_true = y_true.flatten()
        self.AUC_list.append(AUC(y_true, predict))
        self.MRR_list.append(MRR(y_true, predict))
        self.nDCG5_list.append(nDCG(y_true, predict, 5))
        self.nDCG10_list.append(nDCG(y_true, predict, 10))

    def eval(self):
        auc = np.mean(self.AUC_list)
        print('AUC:', auc)
        print('MRR:', np.mean(self.MRR_list))
        print('nDCG@5:', np.mean(self.nDCG5_list))
        print('nDCG@10:', np.mean(self.nDCG10_list))
        return auc
