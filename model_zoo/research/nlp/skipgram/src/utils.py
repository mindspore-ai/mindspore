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
utility function (used in eval.py)
"""

import numpy as np


def cal_top_k_similar(target_embs, emb_matrix, k=1):
    """Return ids of the most similar word of embedding in target_embs
    """
    cosine_projection = np.dot(target_embs, emb_matrix.T)  # sample_num * vocab_size
    target_norms = np.linalg.norm(target_embs, axis=1).reshape(-1, 1)
    emb_norms = np.linalg.norm(emb_matrix, axis=1).reshape(1, -1)
    cosine_similarity = cosine_projection / (np.dot(target_norms, emb_norms))
    top_k_similar = np.argsort(-cosine_similarity, axis=1)
    top_k_similar = top_k_similar[:, :k]
    return top_k_similar


def get_w2v_emb(net, id2word):
    """get word2vec embeddings according to net parameters and dictionary that maps id to word.
    """
    w2v_emb = dict()
    parameters = [item for item in net.c_emb.get_parameters()]
    emb_mat = parameters[0].asnumpy()

    for wid, emb in enumerate(emb_mat):
        word = id2word[wid]
        w2v_emb[word] = emb
    return w2v_emb
